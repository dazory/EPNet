import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import cv2
from collections import deque

import torch
import os
import argparse
import sys


module_path = os.path.abspath('/ws/external')
if module_path not in sys.path:
    sys.path.append(module_path)
from lib.net.point_rcnn import PointRCNN
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.utils.bbox_transform import decode_bbox_target
import torch.nn.functional as F
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils



np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description = "arg parser")
# Please set below arguments
parser.add_argument('--cfg_file', type = str, default = '/ws/external/tools/cfgs/Carlav2_LI_Fusion_with_attention_use_ce_loss.yaml', help = 'specify the config for evaluation')
parser.add_argument('--output_dir', type = str, default = None, help = 'specify an output directory if needed')
parser.add_argument("--ckpt_dir", type = str, default = None, help = "specify a ckpt directory to be evaluated if needed")
parser.add_argument("--ckpt_id", type = int, default = 50)

parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--debug', action='store_true', help='debug mode')

# Please set as default
parser.add_argument("--eval_mode", type = str, default = 'rcnn_online', required = True, help = "specify the evaluation mode")
parser.add_argument('--test', action = 'store_true', default = False, help = 'evaluate without ground truth')
parser.add_argument('--eval_all', action = 'store_true', default = True, help = 'whether to evaluate all checkpoints')
parser.add_argument("--ckpt", type = str, default = None, help = "specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type = str, default = None, help = "specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type = str, default = None, help = "specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size for evaluation')
parser.add_argument('--workers', type = int, default = 0, help = 'number of workers for dataloader')
parser.add_argument("--extra_tag", type = str, default = 'default', help = "extra tag for multiple evaluation")

parser.add_argument('--save_result', action = 'store_true', default = False, help = 'save evaluation results to files')
parser.add_argument('--save_rpn_feature', action = 'store_true', default = False,
                    help = 'save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action = 'store_true', default = True,
                    help = 'sample to the same number of points')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'ignore the checkpoint smaller than this epoch')
parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
parser.add_argument("--rcnn_eval_roi_dir", type = str, default = None,
                    help = 'specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type = str, default = None,
                    help = 'specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest = 'set_cfgs', default = None, nargs = argparse.REMAINDER,
                    help = 'set extra config keys if needed')

parser.add_argument('--model_type', type = str, default = 'base', help = 'model type')

args = parser.parse_args()


class EPNetCarla:
    
    def __init__(self):
        '''
        필요한 것
        input_data = { 'pts_input': inputs }
        input_data['pts_origin_xy']
        input_data['img'] = img
        
        '''
        rospy.init_node('test_epnet')
        
        self.img_sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/camera1/image_color", Image, self.callback_image)
        self.lidar_sub = rospy.Subscriber("/carla/ego_vehicle/lidar/lidar1/point_cloud", PointCloud2, self.callback_lidar)
        self.input_data = None
        self.lidar_seq = 0
        self.camera_seq = 0
        self.output_num = 0
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.npoints = 16384
        
        self.input_deque = deque(maxlen=100)
        
        # self.pred_pub = rospy.Publisher('prediction', String, queue_size=10) # String --> CUSTOM MSG TYPE
        
        # self.model = model
        self.calib = { 'P2': np.array([624.0000000000001, 0.0 ,624.0 ,0.0, 0.0 ,624.0000000000001 ,192.0 ,0.0 ,0.0, 0.0 ,1.0 ,0.0], dtype = np.float32).reshape(3, 4),
            'P3': np.array([624.0000000000001, 0.0 ,624.0 ,0.0, 0.0 ,624.0000000000001 ,192.0 ,0.0 ,0.0, 0.0 ,1.0 ,0.0], dtype = np.float32).reshape(3, 4),
            'R0': np.array([1.0, 0.0 ,0.0 ,0.0 ,1.0 ,0.0 ,0.0 ,0.0 ,1.0], dtype = np.float32).reshape(3, 3),
            'Tr_velo2cam': np.array([0.0, -1.0, 0.0, 0.0 ,0.0, 0.0 ,-1.0 ,-0.0, 1.0, 0.0 ,0.0 ,0.0], dtype = np.float32).reshape(3, 4) }

        self.output_data = dict()
        self.model = None

    def corners3d_to_img_boxes(self, corners3d):
        """
                :param corners3d: (N, 8, 3) corners in rect coordinate
                :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
                :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
                """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.calib['P2'].T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def save_kitti_format(self, bbox3d, scores, classes, img_shape):
        def cls_id_to_type(cls_id):
            type_to_id = {'Background': 0, 'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4, 'Person_sitting': 5}
            for type, id in type_to_id.items():
                if id == cls_id:
                    return type
            return -1

        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
        img_boxes, _ = self.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

        output_data = dict()
        for k in range(bbox3d.shape[0]):
            output_data[k] = dict()
            if cls_id_to_type(classes[k].item()) == 'Background':
                continue
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cls_id_to_type(classes[k].item()), alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2],
                   img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]))#, file=f)
            output_data[k].update({'type': cls_id_to_type(classes[k].item())})
            output_data[k].update({'alpha': alpha})
            output_data[k].update({'bbox': img_boxes[k, 0:4]})
            output_data[k].update({'dimensions': bbox3d[k, 3:6]})
            output_data[k].update({'location': bbox3d[k, 0:3]})
            output_data[k].update({'rotation_y': bbox3d[k, 6]})
            output_data[k].update({'score': scores[k]})

        self.output_data = output_data

    def callback_image(self, img):
        self.camera_seq += 1
        start = rospy.Time.now().to_sec()
        
        if self.input_data is not None and self.camera_seq == self.lidar_seq:
            self.output_num += 1
            cv_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)[:,:,:3]
            
            im = cv_image[...,::-1].copy()
            im = im / 255.0
            im -= self.mean
            im /= self.std
            imback = np.zeros([384, 1280, 3], dtype=np.float32)
            imback[:im.shape[0], :im.shape[1], :] = im
            
            img_shape = (cv_image.shape[0], cv_image.shape[1], 3)
            
            pts_lidar = self.input_data['pts_input']
            if torch.is_tensor(pts_lidar):
                pts_lidar = pts_lidar.cpu().detach().numpy()
            
            pts_rect, pts_intensity = pts_lidar[:, 0:3], pts_lidar[:, 3] # pts_rect = N x 3
            
            # 아래 코드는 lidar_to_rect
            pts_lidar_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype = np.float32))) # pts_rect_hom N x 4
            pts_rect = np.dot(pts_lidar_hom, np.dot(self.calib['Tr_velo2cam'].T, self.calib['R0'].T))
            # 여기까지 해야 PTS_RECT 랑 pts_intensity 가 생김
            
            
            #이제 여기서 rect_to_img
            pts_rect_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype = np.float32)))
            pts_2d_hom = np.dot(pts_rect_hom, self.calib['P2'].T) # N x 4 * 4 * 3
            
            pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T # 이거 중요 
            pts_rect_depth = pts_2d_hom[:, 2] - self.calib['P2'].T[3, 2] # 이거 중요

            # 아래는 valid_flag
            val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
            val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
            val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
            pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
            
            
            x_range, y_range, z_range = np.array([[-40, 40], [-1, 3], [0, 70.4]])
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
            
            pts_rect = pts_rect[pts_valid_flag][:, 0:3]
            
            pts_intensity = pts_intensity[pts_valid_flag]
            pts_origin_xy = pts_img[pts_valid_flag]
            
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace = False)
                
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis = 0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            
            else:
                choice = np.arange(0, len(pts_rect), dtype = np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace = False)
                    choice = np.concatenate((choice, extra_choice), axis = 0)
                np.random.shuffle(choice)
            
            
            ret_pts_origin_xy = pts_origin_xy[choice, :]
            
            self.input_data['img'] = imback # get_image_rgb_with_normal 일 때만
            self.input_data['pts_origin_xy'] = ret_pts_origin_xy
            self.input_data['pts_input'] = self.input_data['pts_input'][choice, : ]

            # print("CAM SEQ:", self.camera_seq)
            print("OUT SEQ:", self.output_num)

            self.eval()

            # for key, value in self.output_data.items():
            #     print(f"{key}, {value}")

        final = rospy.Time.now().to_sec() - start
        
        if  final > 0.1:
            print("final: ", final)

    def callback_lidar(self, lidar):
        # start = rospy.Time.now().to_sec()
        self.lidar_seq += 1
        # lidar = point_cloud2.read_points_list(lidar)
        lidar_numpy = np.frombuffer(lidar.data, dtype=np.float32).reshape(-1, 4)
        # print(lidar_numpy)
        # print(lidar_numpy.shape)
        # lidar_numpy = np.array([np.array([ld.x, ld.y, ld.z, ld.intensity], dtype=np.float32) for ld in lidar])
        self.input_data = {'pts_input': lidar_numpy} # 라이다는 여기만 바꿔주기
        # self.input_deque[self.lidar_seq] = self.input_data
        print("LIDAR SEQ:", self.lidar_seq)
        print("output", self.output_data)

    def eval(self,):
        np.random.seed(666)
        batch_size = 1

        # model inference
        input_data = self.input_data.copy()
        if (input_data is not None) and (self.model is not None):
            print(f">>> model inference")
            self.model.eval()

            pts_input = torch.from_numpy(input_data['pts_input'])[:, :3].unsqueeze(dim=0).cuda()
            pts_origin_xy = torch.from_numpy(input_data['pts_origin_xy']).unsqueeze(dim=0).cuda()
            img = torch.from_numpy(input_data['img']).permute(2, 0, 1).unsqueeze(dim=0).cuda()
            input_data = dict(pts_input=pts_input, pts_origin_xy=pts_origin_xy, img=img)

            ret_dict = self.model(input_data)

            roi_boxes3d = ret_dict['rois']  # (B, M, 7)

            rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
            rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

            if cfg.USE_IOU_BRANCH:
                rcnn_iou_branch = ret_dict['rcnn_iou_branch'].view(batch_size, -1, ret_dict['rcnn_iou_branch'].shape[
                    1])
                rcnn_iou_branch = torch.max(rcnn_iou_branch,
                                            rcnn_iou_branch.new().resize_(rcnn_iou_branch.shape).fill_(1e-4))
                rcnn_cls = rcnn_iou_branch * rcnn_cls

            # bounding box regression
            anchor_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

            pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                              anchor_size=anchor_size,
                                              loc_scope=cfg.RCNN.LOC_SCOPE,
                                              loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                              num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                              get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                              loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                              get_ry_fine=True).view(batch_size, -1, 7)

            # scoring
            if rcnn_cls.shape[2] == 1:
                raw_scores = rcnn_cls  # (B, M, 1)

                norm_scores = torch.sigmoid(raw_scores)
                pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
            else:
                max_rcnn_cls = rcnn_cls.max(dim=-1, keepdim=True)
                raw_scores = max_rcnn_cls.values
                pred_classes = max_rcnn_cls.indices
                cls_norm_scores = F.softmax(rcnn_cls, dim=-1)  # (2, 100, 3)
                norm_scores = cls_norm_scores.max(dim=-1, keepdim=True).values

            # scores thresh
            inds = norm_scores > cfg.RCNN.SCORE_THRESH

            for k in range(batch_size):
                cur_inds = inds[k].view(-1)
                if cur_inds.sum() == 0:
                    continue

                fg_inds = pred_classes[k].squeeze() > 0
                cur_inds = cur_inds * fg_inds

                pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
                raw_scores_selected = raw_scores[k, cur_inds]
                norm_scores_selected = norm_scores[k, cur_inds]
                pred_classes_selected = pred_classes[k, cur_inds]

                # NMS thresh
                # rotated nms
                boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
                keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
                pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
                scores_selected = raw_scores_selected[keep_idx]
                pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().detach().numpy(), scores_selected.cpu().detach().numpy()
                pred_classes_selected = pred_classes_selected[keep_idx]

                _, _, width, height = input_data['img'].shape
                image_shape = height, width, 3

                self.save_kitti_format(pred_boxes3d_selected, scores_selected, pred_classes_selected, image_shape)
        else:
            return

    def run(self):
        rospy.spin()


if __name__ == "__main__":

    epnetcarla = EPNetCarla()

    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rcnn_online':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    with torch.no_grad():
        assert os.path.exists(ckpt_dir), '%s' % ckpt_dir

        if args.model_type == 'base':
            classes = ('Background', 'Car', 'Pedestrian')
            num_classes = classes.__len__()
            model = PointRCNN(num_classes=num_classes, use_xyz=True, mode='TEST')

        model.cuda()

        # load checkpoint
        cur_epoch_id = args.ckpt_id
        cur_ckpt = f"{ckpt_dir}/checkpoint_epoch_{cur_epoch_id}.pth"
        train_utils.load_checkpoint(model, filename=cur_ckpt)

        epnetcarla.model = model

        print(f"START RUN!!")
        epnetcarla.run()


