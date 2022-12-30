from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDatasetMultiCam, MultiTrajFolderDataset
from Datasets.transformation import ses2poses_quat, ses2pos_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.evaluate_rpe import calc_motion_error
from TartanVO import TartanVO

from pgo import run_pgo
from pvgo import run_pvgo
from imu_integrator import run_imu_preintegrator

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import axis_angle_to_quaternion

import pypose as pp
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import argparse
from os import mkdir
from os.path import isdir
from timer import Timer


def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--data-root', default='',
                        help='data root dir (default: "")')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--flow-model-name', default='',
                        help='name of pretrained flow model (default: "")')
    parser.add_argument('--pose-model-name', default='',
                        help='name of pretrained pose model (default: "")')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained vo model. if provided, this will override the other seperated models (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--train-step', type=int, default=1000000,
                    help='number of interactions in total (default: 1000000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--print-interval', type=int, default=1,
                        help='the interval for printing the loss (default: 1)')
    parser.add_argument('--snapshot-interval', type=int, default=1000,
                        help='the interval for snapshot results (default: 1000)')
    parser.add_argument('--train-name', default='',
                        help='name of the training (default: "")')
    parser.add_argument('--result-dir', default='',
                        help='root directory of results (default: "")')
    parser.add_argument('--device', default='cuda',
                        help='device (default: "cuda")')
    parser.add_argument('--mode', default='train-all', choices=['test', 'train-all'],
                        help='running mode: test, train-all (default: train-all)')
    parser.add_argument('--vo-optimizer', default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='VO optimizer: adam, rmsprop, sgd (default: adam)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    timer = Timer()
    args = get_args()

    if args.device.startswith('cuda:'):
        torch.cuda.set_device(args.device)
        
    transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
                            DownscaleFlow(), 
                            Normalize(), 
                            ToTensor(),
                            SqueezeBatchDim(),
                        ])

    trainDataset = MultiTrajFolderDataset(DatasetType=TrajFolderDatasetMultiCam,
                                            root=args.data_root, transform=transform)

    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    dataiter = iter(trainDataloader)

    trainroot = args.result_dir
    print('Train root:', trainroot)
    print(args)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                            device=args.device, use_stereo=2, correct_scale=False)
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)

    criterion = torch.nn.L1Loss()

    if args.mode == 'test':
        args.train_step = 1
    for train_step_cnt in range(args.train_step):
        # print('Start {} step {} ...'.format(args.mode, train_step_cnt))
        timer.tic('step')

        try:
            sample = next(dataiter)
        except StopIteration:
            dataiter = iter(trainDataloader)
            sample = next(dataiter)

        is_train = args.mode.startswith('train')
        res = tartanvo.run_batch(sample, is_train)
        motion = res['pose']

        gt_motion = sample['motion'].to(args.device)
        loss = criterion(motion, gt_motion)
        loss.backward()
        posenetOptimizer.step()

        timer.toc('step')

        if train_step_cnt % args.print_interval == 0:
            with torch.no_grad():
                loss_trans = criterion(motion[:3], gt_motion[:3])
                loss_rot = criterion(motion[3:], gt_motion[3:])
            print('step:{}, loss:{}, trans:{}, rot:{}, time:{}'.format(train_step_cnt, loss.item(), loss_trans.item(), loss_rot.item(), timer.last('step')))

        if train_step_cnt % args.snapshot_interval == 0:
            pass
