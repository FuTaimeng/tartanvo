from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, plottraj
from Datasets.TrajFolderDataset import TrajFolderDataset, TrajFolderDatasetPVGO
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

import argparse
from os import mkdir
from os.path import isdir
from timer import Timer

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--data-root', default='',
                        help='data root dir (default: "")')
    parser.add_argument('--data-type', default='tartanair', choices=['tartanair', 'euroc', 'kitti'],
                        help='data type: tartanair, euroc, kitti (default: tartanair)')
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
                        help='use imu (default: True)')                    
    parser.add_argument('--use-pvgo', action='store_true', default=True,
                        help='use pose-velocity graph optimization (default: True)')
    parser.add_argument('--loss-weight', default='(1,1,1,1)',
                        help='weight of the loss terms (default: (1,1,1,1))')
    parser.add_argument('--mode', default='train', choices=['test', 'train'],
                        help='running mode: test, train (default: train)')
    parser.add_argument('--vo-optimizer', default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='VO optimizer: adam, rmsprop, sgd (default: adam)')
    parser.add_argument('--use-loop-closure', action='store_true', default=False,
                        help='use loop closure or not (default: False)')
    parser.add_argument('--use-stop-constraint', action='store_true', default=False,
                        help='use stop constraint or not (default: False)')
    parser.add_argument('--use-stereo', type=float, default=0,
                        help='use stereo (default: 0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='device: cuda cpu (default: cuda)')
    parser.add_argument('--only-backpropagate-loop-edge', action='store_true', default=False,
                        help='only backpropagate loop edge (default: False)')   

    args = parser.parse_args()
    args.loss_weight = eval(args.loss_weight)   # string to tuple

    return args


if __name__ == '__main__':
    timer = Timer()
    args = get_args()
        
    if args.use_stereo==1:
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

    dataset = TrajFolderDataset(datadir=args.data_root, datatype=args.data_type,
                                transform=transform, start_frame=args.start_frame, end_frame=args.end_frame)

    trainroot = args.result_dir
    print('Train root:', trainroot)
    print(args)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)

    if args.use_imu:
        if dataset.has_imu:
            timer.tic('imu')

            imu_motion_mode = False

            imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
                dataset.accels, dataset.gyros, dataset.imu_dts, dataset.rgb2imu_sync,
                init=dataset.imu_init, gravity=dataset.gravity, device=args.device, motion_mode=imu_motion_mode)

            imu_poses = np.concatenate((imu_trans, imu_rots), axis=1)
            np.savetxt(trainroot+'/imu_pose.txt', imu_poses)
            np.savetxt(trainroot+'/imu_vel.txt', imu_vels)
            dataset.imu_pose2motion(imu_poses)
            fig = plottraj('imu_traj', [imu_poses, dataset.poses], ['cyan', 'g'])
            fig.savefig(trainroot+'/imu_traj.png')

            imu_motion_mode = True

            imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
                dataset.accels, dataset.gyros, dataset.imu_dts, dataset.rgb2imu_sync,
                init=dataset.imu_init, gravity=dataset.gravity, device=args.device, motion_mode=imu_motion_mode)
            # imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
            #     dataset.accels.reshape(-1,3), dataset.gyros.reshape(-1,3), dataset.imu_dts.reshape(-1), 
            #     init=dataset.imu_init, gravity=dataset.gravity, 
            #     device=args.device, motion_mode=imu_motion_mode)

            imu_motions = np.concatenate((imu_trans, imu_rots), axis=1)
            np.savetxt(trainroot+'/imu_motion.txt', imu_motions)
            np.savetxt(trainroot+'/imu_dvel.txt', imu_vels)
            
            timer.toc('imu')
            print('imu preintegration time:', timer.tot('imu'))

            np.savetxt(trainroot+'/imu_accel.txt', dataset.accels)
            np.savetxt(trainroot+'/imu_gyro.txt', dataset.gyros)
            np.savetxt(trainroot+'/imu_gt_vel.txt', dataset.vels)
            np.savetxt(trainroot+'/imu_dt.txt', dataset.imu_dts)

        else:
            print("No IMU data! Turn use_imu to False.")
            args.use_imu = False

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, 
                        pose_model_name=args.pose_model_name, stereo_model_name=args.stereo_model_name,
                        device=args.device, use_stereo=args.use_stereo, correct_scale=(args.use_stereo==0), fix_parts=['flow'])
    lr = args.lr
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)

    frame_cnt = 0
    current_pose = pp.SE3(np.concatenate([dataset.imu_init['pos'], dataset.imu_init['rot']])).to(args.device)

    for train_step_cnt in range(1, args.train_step+1):
        
        st = frame_cnt
        end = frame_cnt + args.batch_size
        if end > dataset.num_img:
            break
        frame_cnt = end - 1
        
        print('Start {} step {}, frame {}-{}'.format(args.mode, train_step_cnt, st, end))

        timer.tic('step')

        timer.tic('load')
            
        subdataset = TrajFolderDatasetPVGO(datadir=None, datatype=None, transform=transform, loader=dataset.loader, 
                                            start_frame=st, end_frame=end)
        dataloader = DataLoader(subdataset, batch_size=len(subdataset.links), shuffle=False, num_workers=args.worker_num)
        sample = next(iter(dataloader))

        sub_imu_rots = imu_rots[st:end-1]
        sub_imu_trans = imu_trans[st:end-1]
        sub_imu_vels = imu_vels[st:end-1]

        timer.toc('load')

        timer.tic('vo')

        is_train = args.mode.startswith('train')
        res = tartanvo.run_batch(sample, is_train)
        motions = res['pose']

        rgb2imu_pose = dataset.rgb2imu_pose.to(args.device)
        motions = rgb2imu_pose @ pp.se3(motions).Exp() @ rgb2imu_pose.Inv()

        timer.toc('vo')

        N = subdataset.num_img - 1
        poses = pp.randn_SE3(N+1).to(args.device)
        poses[0] = current_pose
        for i in range(0, N):
            current_pose = current_pose @ motions[i]
            poses[i+1] = current_pose
        
        timer.tic('pgo')

        if args.use_imu and args.use_pvgo:
            loss, pgo_poses, pgo_vels = run_pvgo(poses, motions, subdataset.links, 
                sub_imu_rots, sub_imu_trans, sub_imu_vels, subdataset.rgb_dts, 
                device=args.device, loss_weight=args.loss_weight, stop_frames=subdataset.stop_frames)
        else:
            loss, pgo_poses = run_pgo(poses, motions, subdataset.links, device=args.device)
            
        timer.toc('pgo')
        
        timer.tic('opt')

        if args.mode.startswith('train'):
            posenetOptimizer.zero_grad()
            if args.only_backpropagate_loop_edge:
                loss[N:].backward(torch.ones_like(loss[N:]))
            else:
                loss.backward(torch.ones_like(loss))
            posenetOptimizer.step()

        timer.toc('opt')

        timer.toc('step')

        timer.tic('print')

        if train_step_cnt % args.print_interval == 0:
            motions_gt = sample['motion'].numpy()
            R_errs, t_errs = calc_motion_error(motions_gt, motions.detach().cpu().numpy(), allow_rescale=False)
            print("#%d - loss:%.6f - lr:%.6f - [time] step:%.2f, load:%.2f, vo:%.2f, pgo:%.2f, opt:%.2f" % 
                    (train_step_cnt, torch.mean(loss), lr, timer.last('step'), timer.last('load'), timer.last('vo'), timer.last('pgo'), timer.last('opt')))
            print("\tadj.\tloop\ttot.")
            print("R\t%.4f\t%.4f\t%.4f" % (np.mean(R_errs[:N]), np.mean(R_errs[N:]), np.mean(R_errs)))
            print("t\t%.4f\t%.4f\t%.4f" % (np.mean(t_errs[:N]), np.mean(t_errs[N:]), np.mean(t_errs)))

        timer.toc('print')

        timer.tic('snapshot')

        if train_step_cnt % args.snapshot_interval == 0:
            traindir = trainroot+'/'+str(train_step_cnt)
            if not isdir(traindir):
                mkdir(traindir)
            
            np.savetxt(traindir+'/pose.txt', poses.detach().cpu().numpy())
            np.savetxt(traindir+'/motion.txt', motions.detach().cpu().numpy())
            np.savetxt(traindir+'/pgo_pose.txt', pgo_poses)
            if args.use_imu and args.use_pvgo:
                np.savetxt(traindir+'/pgo_vel.txt', pgo_vels)

        timer.toc('snapshot')
