from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
from Datasets.transformation import tartan2kitti_pypose, motion2pose_pypose, cvtSE3_pypose
from Datasets.TrajFolderDataset import TrajFolderDatasetPVGO
from evaluator.evaluate_rpe import calc_motion_error
from TartanVO import TartanVO

from pvgo import run_pvgo
from imu_integrator import run_imu_preintegrator

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pypose as pp
import numpy as np
import cv2

import argparse
from os import mkdir
from os.path import isdir
from timer import Timer

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

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
    parser.add_argument('--stereo-model-name', default='',
                        help='name of pretrained stereo model (default: "")')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained vo model. if provided, this will override the other seperated models (default: "")')
    parser.add_argument('--data-root', default='',
                        help='data root dir (default: "")')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='start frame (default: 0)')
    parser.add_argument('--end-frame', type=int, default=-1,
                        help='end frame (default: -1)')
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
    parser.add_argument('--use-imu', action='store_true', default=True,
                        help='use imu (default: "True")')
    parser.add_argument('--use-pvgo', action='store_true', default=True,
                        help='use pose-velocity graph optimization (default: "True")')
    parser.add_argument('--loss-weight', default='(1,1,1,1)',
                        help='weight of the loss terms (default: \'(1,1,1,1)\')')
    parser.add_argument('--mode', default='train-all', choices=['test', 'train-all'],
                        help='running mode: test, train-all (default: train-all)')
    parser.add_argument('--vo-optimizer', default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='VO optimizer: adam, rmsprop, sgd (default: adam)')
    parser.add_argument('--use-loop-closure', action='store_true', default=False,
                        help='use loop closure or not (default: False)')
    parser.add_argument('--use-stop-constraint', action='store_true', default=False,
                        help='use stop constraint or not (default: False)')
    parser.add_argument('--use-stereo', type=int, default=0,
                        help='use stereo (default: 0)')
    parser.add_argument('--device', default='cuda',
                        help='device (default: "cuda")')
    parser.add_argument('--data-type', default='tartanair', choices=['tartanair', 'kitti', 'euroc'],
                        help='data type: tartanair, kitti, euroc (default: "tartanair")')
    parser.add_argument('--fix-model-parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')

    args = parser.parse_args()
    args.loss_weight = eval(args.loss_weight)   # string to tuple

    return args


if __name__ == '__main__':
    timer = Timer()
    args = get_args()

    device_id = 0
    if args.device.startswith('cuda:'):
        torch.cuda.set_device(args.device)
        device_id = int(args.device[5:])
        
    if args.use_stereo == 1:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
                                DownscaleFlow(), 
                                Normalize(mean=mean, std=std, keep_old=True), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])
    else:
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
                                DownscaleFlow(), 
                                Normalize(), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])

    dataset = TrajFolderDatasetPVGO(datadir=args.data_root, datatype=args.data_type, transform=transform,
                                    start_frame=args.start_frame, end_frame=args.end_frame, batch_size=args.batch_size)
    vo_batch_size = args.batch_size*2 - 1
    dataloader = DataLoader(dataset, batch_size=vo_batch_size, shuffle=False)
    dataiter = iter(dataloader)

    trainroot = args.result_dir
    print('Train root:', trainroot)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/link.txt', np.array(dataset.links), fmt='%d')
    np.savetxt(trainroot+'/stop_frame.txt', np.array(dataset.stop_frames), fmt='%d')

    if args.use_imu and not dataset.has_imu:
        print("No IMU data! Turn use_imu to False.")
        args.use_imu = False

    if args.use_imu:
        timer.tic('imu')

        imu_motion_mode = False

        imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
            dataset.accels, dataset.gyros, dataset.imu_dts, 
            init=dataset.imu_init, gravity=dataset.gravity, 
            device=args.device, motion_mode=imu_motion_mode)

        imu_poses = np.concatenate((imu_trans, imu_rots), axis=1)
        np.savetxt(trainroot+'/imu_pose.txt', imu_poses)
        np.savetxt(trainroot+'/imu_vel.txt', imu_vels)
        # dataset.load_imu_motion(imu_poses)

        imu_motion_mode = True

        imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
            dataset.accels, dataset.gyros, dataset.imu_dts, 
            init=dataset.imu_init, gravity=dataset.gravity, 
            device=args.device, motion_mode=imu_motion_mode)

        imu_motions = np.concatenate((imu_trans, imu_rots), axis=1)
        np.savetxt(trainroot+'/imu_motion.txt', imu_motions)
        np.savetxt(trainroot+'/imu_dvel.txt', imu_vels)
        
        timer.toc('imu')
        print('imu preintegration time:', timer.tot('imu'))

        np.savetxt(trainroot+'/imu_accel.txt', dataset.accels.reshape(-1, 3))
        np.savetxt(trainroot+'/imu_gyro.txt', dataset.gyros.reshape(-1, 3))
        np.savetxt(trainroot+'/imu_gt_vel.txt', dataset.vels.reshape(-1, 3))
        np.savetxt(trainroot+'/imu_dt.txt', dataset.imu_dts.reshape(-1, 1))
        # exit(0)

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                        device_id=device_id, use_stereo=args.use_stereo, correct_scale=(args.use_stereo==0), 
                        fix_parts=args.fix_model_parts, use_DDP=False)
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)

    current_idx = 0
    init_state = dataset.imu_init
    
    for train_step_cnt in range(args.train_step):
        timer.tic('step')
        
        try:
            sample = next(dataiter)
        except StopIteration:
            break
            
        print('Start {} step {} ...'.format(args.mode, train_step_cnt))

        use_joint_flow = True
        if use_joint_flow:
            links = sample['link'] - current_idx
            img0 = sample['img0']
            img1 = sample['img1']
            flow = []
            for i in range(links.shape[0]):
                if links[i, 0] + 1 == links[i, 1]:
                    f = tartanvo.pred_flow(img0[i], img1[i])
                    flow.append(f)
                else:
                    flow_to_join = [flow[k] for k in range(links[i, 0], links[i, 1])]
                    f = tartanvo.join_flow(flow_to_join)
                    flow.append(f)
            flow = torch.stack(flow)
            sample['flow'] = flow


        timer.tic('vo')
            
        is_train = args.mode.startswith('train')
        res = tartanvo.run_batch(sample, is_train)
        motions = res['pose']
        flows = res['flow']

        timer.toc('vo')

        timer.tic('cvt')

        # print(dataset.rgb2imu_pose.matrix())
        # print(dataset.rgb2imu_pose.Inv().matrix())

        motions = tartan2kitti_pypose(motions)
        T_ic = dataset.rgb2imu_pose.to(args.device)
        motions = T_ic @ motions @ T_ic.Inv()
        poses = motion2pose_pypose(motions[:args.batch_size])

        motions_np = motions.detach().cpu().numpy()
        poses_np = poses.detach().cpu().numpy()

        # motions_np = motions.detach().cpu().numpy()
        # poses_np = ses2poses_quat(motions_np[:args.batch_size])

        # quats = pp.so3(motions[:, 3:]).Exp().tensor()
        # motions = torch.cat((motions[:, :3], quats), dim=1)
        # motions_np = motions.detach().cpu().numpy()

        timer.toc('cvt')
        
        timer.tic('pgo')

        if args.use_imu and args.use_pvgo:
            current_links = sample['link'].numpy() - current_idx
            # TODO: use rgb2imu index
            current_imu_rots = imu_rots[current_idx:current_idx+args.batch_size]
            current_imu_trans = imu_trans[current_idx:current_idx+args.batch_size]
            current_imu_vels = imu_vels[current_idx:current_idx+args.batch_size]
            current_dts = dataset.rgb_dts[current_idx:current_idx+args.batch_size]
            loss, pgo_poses, pgo_vels, pgo_motions = run_pvgo(poses_np, motions, current_links, 
                current_imu_rots, current_imu_trans, current_imu_vels, init_state, current_dts, 
                device=args.device, loss_weight=args.loss_weight, stop_frames=dataset.stop_frames)

        timer.toc('pgo')
        
        # timer.tic('opt')

        # if args.mode.startswith('train'):
        #     posenetOptimizer.zero_grad()
        #     if args.only_backpropagate_loop_edge:
        #         N = dataset.num_img - 1
        #         loss[N:].backward(torch.ones_like(loss[N:]))
        #     else:
        #         loss.backward(torch.ones_like(loss))
        #     posenetOptimizer.step()

        # timer.toc('opt')

        timer.toc('step')

        timer.tic('print')

        if train_step_cnt % args.print_interval == 0:
            motions_gt = sample['motion']
            if args.data_type == 'tartanair':
                motions_gt = tartan2kitti_pypose(motions_gt)
            else:
                motions_gt = cvtSE3_pypose(motions_gt)
            poses_gt = motion2pose_pypose(motions_gt[:args.batch_size])

            motions_gt = motions_gt.numpy()
            poses_gt = poses_gt.numpy()

            R_errs, t_errs, R_norms, t_norms = calc_motion_error(motions_gt, motions_np, allow_rescale=False)
            print('Pred: R:%.5f t:%.5f' % (np.mean(R_errs), np.mean(t_errs)))
            
            R_errs, t_errs, R_norms, t_norms = calc_motion_error(motions_gt, pgo_motions, allow_rescale=False)
            print('PVGO: R:%.5f t:%.5f' % (np.mean(R_errs), np.mean(t_errs)))

            print('Norm: R:%.5f t:%.5f' % (np.mean(R_norms), np.mean(t_norms)))

        timer.toc('print')

        timer.tic('snapshot')

        if train_step_cnt % args.snapshot_interval == 0:
            traindir = trainroot+'/'+str(train_step_cnt)
            if not isdir(traindir):
                mkdir(traindir)
            
            np.savetxt(traindir+'/pose.txt', poses_np)
            np.savetxt(traindir+'/motion.txt', motions_np)

            np.savetxt(traindir+'/pgo_pose.txt', pgo_poses)
            np.savetxt(traindir+'/pgo_motion.txt', pgo_motions)
            np.savetxt(traindir+'/pgo_vel.txt', pgo_vels)

            np.savetxt(traindir+'/gt_pose.txt', poses_gt)
            np.savetxt(traindir+'/gt_motion.txt', motions_gt)

        timer.toc('snapshot')

        print('[time] step: {}, print: {}, snapshot: {}'.format(timer.last('step'), timer.last('print'), timer.last('snapshot')))

        current_idx += args.batch_size
        init_state = {'rot':pgo_poses[-1][3:], 'pos':pgo_poses[-1][:3], 'vel':pgo_vels[-1]}
