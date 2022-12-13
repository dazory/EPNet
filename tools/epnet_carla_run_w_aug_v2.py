import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray
import cv2
from epnet_carla.msg import kittiMsg, kittiMsgs
from collections import deque

import torch
import os
import argparse
import sys

module_path = os.path.abspath('/ws/epnet_carla_ws/src/EPNet')
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath('/ws/epnet_carla_ws/src/epnet_carla')
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.net.point_rcnn import PointRCNN
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, cfg_from_list
import torch.nn.functional as F
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from epnet_corruption import gen_corruption_v2, gen_corruption
import torchvision
from torch_augmentation import *
import cv2


# from PIL import Image
np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")
# Please set below arguments
parser.add_argument('--cfg_file', type=str,
                    default='/ws/external/tools/cfgs/Carlav2_ros_LI_Fusion_with_attention_use_ce_loss.yaml',
                    help='specify the config for evaluation')
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
parser.add_argument("--ckpt_id", type=int, default=50)

parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--debug', action='store_true', help='debug mode')

# Please set as default
parser.add_argument("--eval_mode", type=str, default='rcnn_online', required=True, help="specify the evaluation mode")
parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
parser.add_argument('--eval_all', action='store_true', default=True, help='whether to evaluate all checkpoints')
parser.add_argument("--ckpt", type=str, default=None, help="specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type=str, default=None, help="specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")

parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                    help='save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action='store_true', default=True,
                    help='sample to the same number of points')
parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')
parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')

parser.add_argument('--model_type', type=str, default='base', help='model type')

args = parser.parse_args()


class EPNetCarla:

    def __init__(self):
        '''
        D�\ �
        input_data = { 'pts_input': inputs }
        input_data['pts_origin_xy']
        input_data['img'] = img

        '''
        rospy.init_node('test_epnet')

        # self.img_sub = rospy.Subscriber("/carla/ego_vehicle/front_cam/image", Image, self.callback_image)
        self.img_sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.callback_image)

        self.lidar_sub = rospy.Subscriber("/carla/ego_vehicle/lidar/lidar1/point_cloud", PointCloud2,
                                          self.callback_lidar)
        # self.lidar_sub = rospy.Subscriber("/carla/ego_vehicle/lidar", PointCloud2, self.callback_lidar)

        self.pred_pub = rospy.Publisher('/carla/prediction', kittiMsgs, queue_size=100)  # String --> CUSTOM MSG TYPE


        self.input_data = None
        self.lidar_seq = 0
        self.camera_seq = 0
        self.data_flag = 0
        self.output_num = 0

        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()

        self.npoints = 16384

        # self.transform_list = torchvision.transforms.Compose(
        #     [torchvision.transforms.ToPILImage(),
        #      torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        #      torchvision.transforms.ToTensor(),
        #      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #
        #      ])

        # self.input_deque = deque(maxlen=10000)
        # self.rate = rospy.Rate(100)

        self.calib = {
            'P2': torch.tensor([624.0000000000001, 0.0, 624.0, 0.0, 0.0, 624.0000000000001, 192.0, 0.0, 0.0, 0.0, 1.0,
                                0.0]).cuda().reshape(3, 4),
            'P3': torch.tensor([624.0000000000001, 0.0, 624.0, 0.0, 0.0, 624.0000000000001, 192.0, 0.0, 0.0, 0.0, 1.0,
                                0.0]).cuda().reshape(3, 4),
            'R0': torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).cuda().reshape(3, 3),
            'Tr_velo2cam': torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.0, 1.0, 0.0, 0.0, 0.0]).cuda().reshape(
                3, 4)}

        self.output_data = dict()
        self.model = None
        self.model_time_mean = 0.0
        self.total_time_mean = 0.0

        loc_scope, loc_bin_size = cfg.RCNN.LOC_SCOPE, cfg.RCNN.LOC_BIN_SIZE,
        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
        xz_bin_ind = torch.arange(per_loc_bin_num).float()
        xz_bin_center = xz_bin_ind * loc_bin_size + loc_bin_size / 2 - loc_scope  # num_bin
        self.xz_bin_center = xz_bin_center.to('cuda')

        loc_scope, loc_bin_size = cfg.RCNN.LOC_SCOPE, cfg.RCNN.LOC_BIN_SIZE,
        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
        xz_bin_ind = torch.arange(per_loc_bin_num).float()
        xz_bin_center = xz_bin_ind * loc_bin_size + loc_bin_size / 2 - loc_scope  # num_bin
        self.xz_bin_center = xz_bin_center.to('cuda')

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

        assert os.path.exists(ckpt_dir), '%s' % ckpt_dir
        if args.model_type == 'base':
            classes = ('Background', 'Car', 'Pedestrian')
            num_classes = classes.__len__()
            self.model = PointRCNN(num_classes=num_classes, use_xyz=True, mode='TEST')
        self.model.cuda()

        # load checkpoint
        cur_epoch_id = args.ckpt_id
        cur_ckpt = f"{ckpt_dir}/checkpoint_epoch_{cur_epoch_id}.pth"
        train_utils.load_checkpoint(self.model, filename=cur_ckpt)
        print(f"cur_ckpt: {cur_ckpt}")

        self.anchor_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()


    def decode_bbox_target(self, roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, ):
        anchor_size = self.anchor_size

        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2

        # recover xz localization
        assert cfg.TRAIN.BBOX_AVG_BY_BIN == cfg.TEST.BBOX_AVG_BY_BIN

        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2

        pred_x_bin = F.softmax(pred_reg[:, x_bin_l: x_bin_r], 1)  # N x num_bin
        pred_z_bin = F.softmax(pred_reg[:, z_bin_l: z_bin_r], 1)

        xz_bin_center = self.xz_bin_center

        pred_x_abs = xz_bin_center
        pred_z_abs = xz_bin_center

        # if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        pred_x_reg = pred_reg[:, x_res_l: x_res_r] * loc_bin_size  # N x num_bin
        pred_z_reg = pred_reg[:, z_res_l: z_res_r] * loc_bin_size

        pred_x_abs = pred_x_abs + pred_x_reg
        pred_z_abs = pred_z_abs + pred_z_reg

        pos_x = (pred_x_abs * pred_x_bin).sum(dim=1)
        pos_z = (pred_z_abs * pred_z_bin).sum(dim=1)

        # recover y localization
        # if not get_y_by_bin:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

        # recover ry rotation
        ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

        # if not cfg.TEST.RY_WITH_BIN:
        ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
        ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
        # if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4

        # recover size
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3

        size_res_norm = pred_reg[:, size_res_l: size_res_r]
        hwl = size_res_norm * anchor_size + anchor_size

        # shift to original coords
        roi_center = roi_box3d[:, 0:3]
        shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)),
                                    dim=1)
        ret_box3d = shift_ret_box3d
        if roi_box3d.shape[1] == 7:
            roi_ry = roi_box3d[:, 6]
            ret_box3d = self.rotate_pc_along_y_torch(shift_ret_box3d, -roi_ry)
            ret_box3d[:, 6] += roi_ry
        ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
        return ret_box3d

    def rotate_pc_along_y_torch(self, pc, rot_angle):
        cosa = torch.cos(rot_angle).view(-1, 1)
        sina = torch.sin(rot_angle).view(-1, 1)

        raw_1 = torch.cat([cosa, -sina], dim=1)
        raw_2 = torch.cat([sina, cosa], dim=1)

        R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)
        pc_temp = pc[:, [0, 2]].reshape(-1, 1, 2)  # (N, 1, 2)

        pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
        return pc

    def corners3d_to_img_boxes(self, corners3d):
        """
                :param corners3d: (N, 8, 3) corners in rect coordinate
                :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
                :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
                """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.calib['P2'].permute(1,0).cpu().detach().numpy())  # (N, 8, 3)

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
                   bbox3d[k, 6], scores[k]))  # , file=f)

            output_data[k].update({'type': cls_id_to_type(classes[k].item())})
            output_data[k].update({'alpha': alpha})
            output_data[k].update({'bbox': img_boxes[k, 0:4]})
            output_data[k].update({'dimensions': bbox3d[k, 3:6]})
            output_data[k].update({'location': bbox3d[k, 0:3]})
            output_data[k].update({'rotation_y': bbox3d[k, 6]})
            output_data[k].update({'score': scores[k]})

        self.output_data = output_data

    def save_kitti_format_v2(self, bbox3d, scores, classes, img_shape):
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

        # output_data = dict()
        output_kittis = kittiMsgs()
        for k in range(bbox3d.shape[0]):
            # output_data[k] = dict()
            if cls_id_to_type(classes[k].item()) == 'Background':
                continue
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            output_kitti = kittiMsg()
            output_kitti.type = cls_id_to_type(classes[k].item())
            output_kitti.alpha = alpha
            output_kitti.bbox = img_boxes[k, 0:4]
            output_kitti.dimensions = bbox3d[k, 3:6]
            output_kitti.location = bbox3d[k, 0:3]
            output_kitti.rotation_y = bbox3d[k, 6]
            output_kitti.score = scores[k]

            output_kittis.kittis.append(output_kitti)

        self.pred_pub.publish(output_kittis)
        # self.output_data = output_data

    def save_kitti_format_v3(self, bbox3d, scores, classes, img_shape, sample_id):
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

        # output_data = dict()
        output_kittis = kittiMsgs()

        kitti_output_file = os.path.join("/ws/data2/epnet/carla_result_1118", '%06d.txt' % sample_id)
        with open(kitti_output_file, 'w') as f:
            for k in range(bbox3d.shape[0]):
                # output_data[k] = dict()
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
                       bbox3d[k, 6], scores[k]), file=f)

            output_kitti = kittiMsg()
            output_kitti.type = cls_id_to_type(classes[k].item())
            output_kitti.alpha = alpha
            output_kitti.bbox = img_boxes[k, 0:4]
            output_kitti.dimensions = bbox3d[k, 3:6]
            output_kitti.location = bbox3d[k, 0:3]
            output_kitti.rotation_y = bbox3d[k, 6]
            output_kitti.score = scores[k]

            output_kittis.kittis.append(output_kitti)

        self.pred_pub.publish(output_kittis)

    def callback_image(self, img):
        self.camera_seq += 1

        if self.input_data is not None and self.data_flag == 1:
            self.data_flag = 2
            output_start = rospy.Time.now().to_sec()
            self.output_num += 1
            cv_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)

            # may be corruption ?

            im = torch.as_tensor(cv_image).cuda()[:, :, :3].type(torch.float32)

            im = im[:, :, [2, 1, 0]]
            im = im / 255.0
            im = im - self.mean
            im = im / self.std

            imback = im

            img_shape = (im.shape[0], im.shape[1], 3)

            pts_lidar = self.input_data['pts_input']

            pts_rect, pts_intensity = pts_lidar[:, 0:3], pts_lidar[:, 3]  # pts_rect = N x 3

            # 아래 코드는 lidar_to_rect
            ones = torch.ones((pts_rect.shape[0], 1)).cuda()
            pts_lidar_hom = torch.cat([pts_rect, ones], dim=1)
            pts_rect = torch.matmul(pts_lidar_hom, torch.matmul(self.calib['Tr_velo2cam'].permute(1, 0),
                                                                self.calib['R0'].permute(1, 0)))
            # 여기까지 해야 PTS_RECT 랑 pts_intensity 가 생김

            # 이제 여기서 rect_to_img
            pts_rect_hom = torch.cat([pts_rect, ones], dim=1)
            pts_2d_hom = torch.matmul(pts_rect_hom, self.calib['P2'].permute(1, 0))  # N x 4 * 4 * 3

            pts_img = (pts_2d_hom[:, 0:2].permute(1, 0) / pts_rect_hom[:, 2]).permute(1, 0)  # 이거 중요
            pts_rect_depth = pts_2d_hom[:, 2] - self.calib['P2'].permute(1, 0)[3, 2]  # 이거 중요

            # 아래는 valid_flag
            val_flag_1 = (pts_img[:, 0] >= 0).bool() * (pts_img[:, 0] < img_shape[1]).bool()
            val_flag_2 = (pts_img[:, 1] >= 0).bool() * (pts_img[:, 1] < img_shape[0]).bool()

            val_flag_merge = val_flag_1 * val_flag_2
            pts_valid_flag = val_flag_merge * (pts_rect_depth >= 0).bool()

            x_range, y_range, z_range = torch.tensor([[-40, 40], [-1, 3], [0, 70.4]]).cuda()
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag

            pts_rect = pts_rect[pts_valid_flag][:, 0:3]

            pts_intensity = pts_intensity[pts_valid_flag]
            pts_origin_xy = pts_img[pts_valid_flag]

            if self.npoints < pts_rect.shape[0]:
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = (pts_near_flag == 0).nonzero()[:, 0]

                near_idxs = (pts_near_flag == 1).nonzero()[:, 0]
                perm = torch.randperm(self.npoints - far_idxs_choice.shape[0])
                near_idxs_choice = near_idxs[perm]

                choice = torch.cat((near_idxs_choice, far_idxs_choice), dim=0) \
                    if far_idxs_choice.shape[0] > 0 else near_idxs_choice
                choice = torch.randperm(choice.shape[0])
                # choice = np.arange(0, pts_rect.shape[0], dtype=np.int32)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_origin_xy = pts_origin_xy[choice, :]

            self.input_data['img'] = imback  # get_image_rgb_with_normal 일 때만
            self.input_data['pts_origin_xy'] = ret_pts_origin_xy
            self.input_data['pts_input'] = pts_rect[choice, :]

            # print("CAM SEQ:", self.camera_seq)
            # print("OUT SEQ:", self.output_num)

            self.eval()
            self.data_flag = 0

            current_time = rospy.Time.now().to_sec() - output_start
            print(self.output_num, "[current] Ros + model inference total processed time: ", current_time)
            self.total_time_mean = self.total_time_mean * ((self.output_num - 1) / self.output_num) + (1 / self.output_num) * current_time
            print("[Mean] Ros + model inference total processed time: ", self.total_time_mean)

    def callback_lidar(self, lidar):
        self.lidar_seq += 1

        if self.data_flag == 0:
            lidar_numpy = np.frombuffer(lidar.data, dtype=np.float32)
            lidar_tensor = torch.as_tensor(lidar_numpy).cuda().reshape(-1, 4)
            self.input_data = {'pts_input': lidar_tensor}
            self.data_flag = 1

        elif self.data_flag == 1:
            lidar_numpy = np.frombuffer(lidar.data, dtype=np.float32)
            lidar_tensor = torch.as_tensor(lidar_numpy).cuda().reshape(-1, 4)
            self.input_data = {'pts_input': lidar_tensor}

        elif self.data_flag == 2:  # image processed ...
            self.data_flag = 2


    def eval(self):
        np.random.seed(666)
        batch_size = 1

        # model inference
        with torch.no_grad():
            input_data = self.input_data.copy()
            if (input_data is not None) and (self.model is not None):
                print(f">>> model inference")
                start_model_inference = rospy.Time.now().to_sec()
                self.model.eval()

                pts_input = input_data['pts_input'].unsqueeze(dim=0)
                pts_origin_xy = input_data['pts_origin_xy'].unsqueeze(dim=0)
                # img = input_data['img'].permute(2, 0, 1).unsqueeze(dim=0)
                img = input_data['img'].permute(2, 0, 1)


                rand_num = torch.randint(0,5, (1, ))
                # img_blurred = gaussian_blur(img, [3,3], None)
                img_blurred = random_aug(img, rand_num)

                # img_bright = adjust_brightness(img, 0.6)

                # orig_img = torch_to_opencv(img)
                # cvimg_blurred = torch_to_opencv(img_blurred)
                # cvimg_bright = torch_to_opencv(img_bright)
                #
                # cv2.imshow('orig', orig_img)
                # cv2.imshow('blurred', cvimg_blurred)
                # # cv2.imshow('bright', cvimg_bright)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                #
                img_blurred = img_blurred.unsqueeze(dim=0)

                # input_data = dict(pts_input=pts_input, pts_origin_xy=pts_origin_xy, img=img)
                input_data = dict(pts_input=pts_input, pts_origin_xy=pts_origin_xy, img=img_blurred)
                print(f"EVAL: {pts_input.shape}, {pts_origin_xy.shape}, {img.shape}")

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
                pred_boxes3d = self.decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                                       loc_scope=cfg.RCNN.LOC_SCOPE,
                                                       loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                                       num_head_bin=cfg.RCNN.NUM_HEAD_BIN).view(batch_size, -1, 7)

                # scoring
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

                    _, _, height, width = input_data['img'].shape
                    image_shape = height, width, 3

                    self.save_kitti_format_v2(pred_boxes3d_selected, scores_selected, pred_classes_selected, image_shape)
                    # self.save_kitti_format(pred_boxes3d_selected, scores_selected, pred_classes_selected, image_shape)
                    final = rospy.Time.now().to_sec() - start_model_inference
                    print("[Current] model inference time: ", final)
                    self.model_time_mean = self.model_time_mean * ((self.output_num - 1) / self.output_num) + (1 / self.output_num) * final
                    print("[Mean] Model inference time: ", self.model_time_mean)

            else:
                return

    def run(self):
        rospy.spin()
        # self.rate.sleep()


if __name__ == "__main__":

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

        epnetcarla = EPNetCarla()
        epnetcarla.model = model


        print(f"START RUN!!")

        epnetcarla.run()