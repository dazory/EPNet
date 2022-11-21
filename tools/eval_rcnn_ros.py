import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
import tools.train_utils.train_utils as train_utils
from lib.utils.bbox_transform import decode_bbox_target
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import tqdm

from tools.utils.wandb_logger import WandbLogger
from pointnet2_lib.tools.kitti_utils import cls_id_to_type

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description = "arg parser")
parser.add_argument('--cfg_file', type = str, default = 'cfgs/default.yml', help = 'specify the config for evaluation')
parser.add_argument('--cur_ckpt', type = str, default = '', help = 'specify the config for evaluation')
parser.add_argument("--eval_mode", type = str, default = 'rpn', required = True, help = "specify the evaluation mode")

parser.add_argument('--eval_all', action = 'store_true', default = False, help = 'whether to evaluate all checkpoints')
parser.add_argument('--eval_ros', action = 'store_true', default = False, help = 'evaluation to ROS')
parser.add_argument('--test', action = 'store_true', default = False, help = 'evaluate without ground truth')
parser.add_argument("--ckpt", type = str, default = None, help = "specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type = str, default = None,
                    help = "specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size for evaluation')
parser.add_argument('--workers', type = int, default = 4, help = 'number of workers for dataloader')
parser.add_argument("--extra_tag", type = str, default = 'default', help = "extra tag for multiple evaluation")
parser.add_argument('--output_dir', type = str, default = None, help = 'specify an output directory if needed')
parser.add_argument("--ckpt_dir", type = str, default = None,
                    help = "specify a ckpt directory to be evaluated if needed")

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

parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--debug', action='store_true', help='debug mode')

parser.add_argument('--augmix', action='store_true', help='using augmix')
parser.add_argument('--dataset', default='kitti', required = True, type=str)

args = parser.parse_args()

from tools.utils.eval_rcnn import (create_logger, save_kitti_format, save_rpn_features,
                                   eval_one_epoch_rpn, eval_one_epoch_rcnn, eval_one_epoch_joint,
                                   eval_one_epoch, load_part_ckpt, load_ckpt_based_on_args,
                                   eval_single_ckpt, get_no_evaluated_ckpt, repeat_eval_ckpt,
                                   create_dataloader)

if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    name = args.output_dir.split('/log/')[-1].replace('/', '_')
    init_kwargs = dict(project='EPNet', entity='kaist-url-ai28', name=name)
    wandb_logger = WandbLogger(init_kwargs=init_kwargs,
                               train_epoch_interval=1, train_iter_interval=100,
                               val_epoch_interval=1, val_iter_interval=10,
                               use_wandb=args.wandb)
    wandb_logger.before_run()

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('/ws/data/', 'output', 'rpn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rpn', cfg.TAG, 'ckpt')

    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')

    elif args.eval_mode == 'rcnn_online':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG) # no_use
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt') # may be no use

    elif args.eval_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
        assert args.rcnn_eval_roi_dir is not None and args.rcnn_eval_feature_dir is not None
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok = True)

    with torch.no_grad():
        if args.eval_all:
            assert os.path.exists(ckpt_dir), '%s' % ckpt_dir
            repeat_eval_ckpt(root_result_dir, ckpt_dir, wandb_logger, args)
        elif args.eval_ros:
            # repeat_eval_ckpt(root_result_dir, ckpt_dir, wandb_logger, args)
            eval_single_ckpt(root_result_dir, args)
        else:

            eval_single_ckpt(root_result_dir, args)
