from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, RandomResizeCrop, RandomHSV, save_images
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

import pypose as pp
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import argparse
from os import mkdir
from os.path import isdir
from timer import Timer

from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--lr-decay-rate', type=float, default=0.4,
                        help='learning rate decay rate (default: 0.4)')
    parser.add_argument('--lr-decay-point', type=float, default=[], nargs='+',
                        help='learning rate decay point (default: [])')
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
    parser.add_argument('--debug-flag', default='0',
                        help='Debug flag: (default: 0) \
                                [0] rot/trans error \
                                [1] flow loss \
                                [2] pose output \
                                [3] flow output \
                                [4] images')
    parser.add_argument('--random-intrinsic', type=float, default=0.0,
                        help='similar with random-crop but cover contineous intrinsic values (default: 0.0)')
    parser.add_argument('--hsv-rand', type=float, default=0.0,
                        help='augment rand-hsv by adding different hsv to a set of images (default: 0.0)')
    parser.add_argument('--use-stereo', type=float, default=2.2, 
                        help='stereo mode (default: 2.2) \
                                [1] stereo disp \
                                [2.1] multicam single feat endocer \
                                [2.2] multicam sep feat encoder \
                                [3] multicam standalone scale')
    parser.add_argument('--fix_model_parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')

    args = parser.parse_args()

    args.lr_decay_point = (np.array(args.lr_decay_point) * args.train_step).astype(int)
    
    return args


if __name__ == '__main__':
    timer = Timer()
    args = get_args()

    if args.device.startswith('cuda:'):
        torch.cuda.set_device(args.device)

    trainroot = args.result_dir
    print('Train root:', trainroot)
    print(args)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))

    tb_dir = 'tensorboard/' + trainroot
    if not isdir(tb_dir):
        mkdir(tb_dir)
    writer = SummaryWriter(tb_dir)
        
    # transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
    #                         DownscaleFlow(), 
    #                         Normalize(), 
    #                         ToTensor(),
    #                         SqueezeBatchDim()
    #                     ])

    if args.random_intrinsic>0:
        transformlist = [ RandomResizeCrop( size=(args.image_height, args.image_width), 
                                            max_scale=args.random_intrinsic/320.0, 
                                            keep_center=False, fix_ratio=False) ]
    else:
        transformlist = [ CropCenter( size=(args.image_height, args.image_width), 
                                      fix_ratio=False, scale_w=1.0, scale_disp=False)]
    transformlist.append(DownscaleFlow())
    transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
    transformlist.extend([Normalize(), ToTensor(), SqueezeBatchDim()])
    transform = Compose(transformlist)

    trainDataset = MultiTrajFolderDataset(DatasetType=TrajFolderDatasetMultiCam,
                                            root=args.data_root, transform=transform)
    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    # debug dataset
    # data_dir = args.data_root + '/abandonedfactory/Easy/P000'
    # trainDataset = TrajFolderDatasetMultiCam(data_dir+'/image_left', posefile=data_dir+'/pose_left.txt', transform = transform, 
    #                                             sample_step = 1, start_frame=0, end_frame=50, use_fix_intervel_links=True)
    # trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)

    dataiter = iter(trainDataloader)

    # all_frames = trainDataset.list_all_frames()
    # np.savetxt(trainroot+'/all_frames.txt', all_frames, fmt="%s")
    # quit()

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                            device=args.device, use_stereo=args.use_stereo, correct_scale=False, fix_parts=args.fix_model_parts)
    lr = args.lr
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)

    L1Loss = torch.nn.L1Loss()

    for train_step_cnt in range(1, args.train_step+1):
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

        if args.use_stereo==3:
            rot = motion[..., 3:]
            gt_rot = gt_motion[..., 3:]
            rot_loss = L1Loss(rot, gt_rot)

            trans = motion[..., :3]
            gt_trans = gt_motion[..., :3]
            scale = torch.linalg.norm(trans, dim=1)
            gt_scale = torch.linalg.norm(gt_trans, dim=1)
            trans_norm = trans / scale.view(-1, 1)
            gt_trans_norm = gt_trans / gt_scale.view(-1, 1)
            trans_loss = L1Loss(trans_norm, gt_trans_norm)

            scale = res['scale']
            scale_loss = L1Loss(scale, gt_scale)

            loss = rot_loss + trans_loss + scale_loss
        else:
            loss = L1Loss(motion, gt_motion)

        loss.backward()
        posenetOptimizer.step()

        if train_step_cnt in args.lr_decay_point:
            lr *= args.lr_decay_rate
            for param_group in posenetOptimizer.param_groups: 
                param_group['lr'] = lr
            print('[!] lr decay to {} at step {}!'.format(lr, train_step_cnt))

        timer.toc('step')

        if train_step_cnt <= 10 or train_step_cnt % args.print_interval == 0:
            with torch.no_grad():
                tot_loss = loss.item()
                if args.use_stereo==3:
                    rot_errs, trans_errs = calc_motion_error(gt_motion.cpu().numpy(), motion.cpu().numpy(), allow_rescale=True)
                    scale_errs = torch.abs(scale - gt_scale).cpu().numpy() 
                else:
                    rot_errs, trans_errs = calc_motion_error(gt_motion.cpu().numpy(), motion.cpu().numpy(), allow_rescale=False)
                    scale = torch.linalg.norm(motion[..., :3], dim=1)
                    gt_scale = torch.linalg.norm(gt_motion[..., :3], dim=1)
                    scale_errs = torch.abs(scale - gt_scale).cpu().numpy() 
                trans_err = np.mean(trans_errs)
                rot_err = np.mean(rot_errs)

                scale_err = np.mean(scale_errs)
            writer.add_scalar('loss', tot_loss, train_step_cnt)
            writer.add_scalar('trans_err', trans_err, train_step_cnt)
            writer.add_scalar('rot_err', rot_err, train_step_cnt)
            writer.add_scalar('scale_err', scale_err, train_step_cnt)

            writer.add_scalar('time', timer.last('step'), train_step_cnt)
            print('step:{}, loss:{}, trans_err:{}, rot_err:{}, scale_err:{}, time:{}'.format(
                train_step_cnt, tot_loss, trans_err, rot_err, scale_err, timer.last('step')))
            
            if args.debug_flag != '':
                if not isdir(trainroot+'/debug'):
                    mkdir(trainroot+'/debug')
                debugdir = trainroot+'/debug/'+str(train_step_cnt)
                mkdir(debugdir)

            if '0' in args.debug_flag:
                info = np.array([train_step_cnt, tot_loss, trans_err, rot_err, scale_err, timer.last('step')])
                np.savetxt(debugdir+'/info.txt', info)

            if '1' in args.debug_flag:
                pass
            
            if '2' in args.debug_flag:
                np.savetxt(debugdir+'/motion.txt', motion.detach().cpu().numpy())
                np.savetxt(debugdir+'/gt_motion.txt', gt_motion.detach().cpu().numpy())

            verbose_debug = True
            if '3' in args.debug_flag:
                save_images(debugdir, res['flowAB']*20, suffix='_flowAB')
                save_images(debugdir, res['flowAC']*20, suffix='_flowAC')

            if '4' in args.debug_flag:
                save_images(debugdir, sample['img0'], suffix='_A')
                save_images(debugdir, sample['img0_r'], suffix='_B')
                save_images(debugdir, sample['img1'], suffix='_C')
                
        else:
            writer.add_scalar('loss', loss.item(), train_step_cnt)
            print('step:{}, loss:{}'.format(train_step_cnt, loss.item()))

        if train_step_cnt % args.snapshot_interval == 0:
            if not isdir(trainroot+'/models'):
                mkdir(trainroot+'/models')
            torch.save(tartanvo.vonet.flowPoseNet.state_dict(), '{}/models/multicamvo_posenet_{}.pkl'.format(trainroot, train_step_cnt))
