from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, plottraj
from Datasets.TrajFolderDataset import TrajFolderDatasetPVGO
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
                        help='use imu (default: "True")')                    
    parser.add_argument('--use-pvgo', action='store_true', default=True,
                        help='use pose-velocity graph optimization (default: "True")')
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
    parser.add_argument('--use-stereo', type=int, default=0,
                        help='use stereo (default: 0)')

    args = parser.parse_args()
    args.loss_weight = eval(args.loss_weight)   # string to tuple

    return args


if __name__ == '__main__':
    timer = Timer()
    args = get_args()
        
    if args.use_stereo==1:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=False), 
                                DownscaleFlow(), 
                                Normalize(mean=mean, std=std, keep_old=True), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])
    else:
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=False), 
                                DownscaleFlow(), 
                                Normalize(), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])

    dataset = TrajFolderDatasetPVGO(datadir=args.data_root, datatype=args.data_type,
                                    transform=transform, start_frame=args.start_frame, end_frame=args.end_frame,
                                    use_loop_closure=False, use_stop_constraint=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_num)

    trainroot = args.result_dir
    print('Train root:', trainroot)
    print(args)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/link.txt', np.array(dataset.links), fmt='%d')
    try:
        np.savetxt(trainroot+'/stop_frame.txt', np.array(dataset.stop_frames), fmt='%d')
    except:
        pass

    if args.use_imu:
        if dataset.has_imu:
            timer.tic('imu')

            imu_motion_mode = False

            imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
                dataset.accels, dataset.gyros, dataset.imu_dts, dataset.rgb2imu_sync,
                init=dataset.imu_init, gravity=dataset.gravity, device='cuda', motion_mode=imu_motion_mode)

            imu_poses = np.concatenate((imu_trans, imu_rots), axis=1)
            np.savetxt(trainroot+'/imu_pose.txt', imu_poses)
            np.savetxt(trainroot+'/imu_vel.txt', imu_vels)
            dataset.imu_pose2motion(imu_poses)
            fig = plottraj('imu_traj', [imu_poses, dataset.poses], ['cyan', 'g'])
            fig.savefig(trainroot+'/imu_traj.png')

            imu_motion_mode = True

            imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
                dataset.accels, dataset.gyros, dataset.imu_dts, dataset.rgb2imu_sync,
                init=dataset.imu_init, gravity=dataset.gravity, device='cuda', motion_mode=imu_motion_mode)
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

    quit()

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name, stereo_model_name=args.stereo_model_name,
                            device=args.device, use_imu=args.use_imu, use_stereo=args.use_stereo, correct_scale=(not args.use_stereo))
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)

    if args.mode == 'test':
        args.train_step = 1
    for train_step_cnt in range(args.train_step):
        print('Start {} step {} ...'.format(args.mode, train_step_cnt))
        timer.tic('step')

        timer.tic('vo')

        motionlist = []
        flowlist = []
        dataiter = iter(dataloader)
        tot_batch = len(dataset)
        batch_cnt = 0
        while True:
            try:
                sample = next(dataiter)
            except StopIteration:
                break
            
            batch_cnt += 1
            if args.mode.startswith('test'):
                print('Batch {}/{} ...'.format(batch_cnt, tot_batch), end='\r')

            is_train = args.mode.startswith('train')
            res = tartanvo.run_batch(sample, is_train)
            motion = res['pose']
            flow = res['flow']
            motionlist.extend(motion)
            flowlist.extend(flow)

        timer.toc('vo')

        timer.tic('cvt')

        motions = torch.stack(motionlist)
        motions_np = motions.detach().cpu().numpy()
        poses_np = ses2poses_quat(motions_np[:dataset.num_img-1])

        quats = axis_angle_to_quaternion(motions[:, 3:])
        motions = torch.cat((motions[:, :3], quats[:, 1:], quats[:, :1]), dim=1)
        motions_np = motions.detach().cpu().numpy()

        timer.toc('cvt')
        
        timer.tic('pgo')

        if args.use_imu and args.use_pvgo:
            loss, pgo_poses, pgo_vels = run_pvgo(poses_np, motions, dataset.links, 
                imu_rots, imu_trans, imu_vels, dataset.imu_init, 1.0/args.frame_fps, 
                device=args.device, loss_weight=args.loss_weight, stop_frames=dataset.stop_frames)
        else:
            loss, pgo_poses = run_pgo(poses_np, motions, dataset.links, device=args.device)
            
        timer.toc('pgo')
        
        timer.tic('opt')

        if args.mode.startswith('train'):
            posenetOptimizer.zero_grad()
            if args.only_backpropagate_loop_edge:
                N = dataset.num_img - 1
                loss[N:].backward(torch.ones_like(loss[N:]))
            else:
                loss.backward(torch.ones_like(loss))
            posenetOptimizer.step()

        timer.toc('opt')

        timer.toc('step')

        timer.tic('print')

        if train_step_cnt % args.print_interval == 0:
            motions_gt = ses2pos_quat(dataset.motions)
            R_errs, t_errs = calc_motion_error(motions_gt, motions_np, allow_rescale=False)
            print("%s #%d - loss:%.6f - lr:%.6f - [avgtime] vo:%.2f, cvt:%.2f, pgo:%.2f, opt:%.2f" % (trainroot.split('/')[-1], 
                train_step_cnt, torch.mean(loss), args.lr, timer.avg('vo'), timer.avg('cvt'), timer.avg('pgo'), timer.avg('opt')))
            timer.clear(['vo', 'cvt', 'pgo', 'opt'])
            N = dataset.num_img - 1
            print("\tadj.\tloop\ttot.")
            print("R\t%.4f\t%.4f\t%.4f" % (np.mean(R_errs[:N]), np.mean(R_errs[N:]), np.mean(R_errs)))
            print("t\t%.4f\t%.4f\t%.4f" % (np.mean(t_errs[:N]), np.mean(t_errs[N:]), np.mean(t_errs)))

        timer.toc('print')

        timer.tic('snapshot')

        if train_step_cnt % args.snapshot_interval == 0:
            traindir = trainroot+'/'+str(train_step_cnt)
            if not isdir(traindir):
                mkdir(traindir)
            
            np.savetxt(traindir+'/pose.txt', poses_np)
            np.savetxt(traindir+'/motion.txt', motions_np)
            np.savetxt(traindir+'/pgo_pose.txt', pgo_poses)
            if args.use_imu and args.use_pvgo:
                np.savetxt(traindir+'/pgo_vel.txt', pgo_vels)

            if args.save_flow:
                flowdir = traindir+'/flow'
                if not isdir(flowdir):
                    mkdir(flowdir)

                for flowcnt, flow in enumerate(flowlist):
                    for k in range(flow.shape[0]):
                        flowk = flow[k].transpose(1,2,0)
                        np.save(flowdir+'/'+str(flowcnt).zfill(6)+'.npy', flowk)
                        flow_vis = visflow(flowk)
                        cv2.imwrite(flowdir+'/'+str(flowcnt).zfill(6)+'.png', flow_vis)

        timer.toc('snapshot')

        print('[time] step: {}, print: {}, snapshot: {}'.format(timer.avg('step'), timer.avg('print'), timer.avg('snapshot')))
        timer.clear(['step', 'print', 'snapshot'])
