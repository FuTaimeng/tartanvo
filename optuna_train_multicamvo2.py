from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, RandomResizeCrop, RandomHSV, save_images
from Datasets.TrajFolderDataset import TrajFolderDatasetMultiCam, MultiTrajFolderDataset, TrajFolderDatasetPVGO, LoopDataSampler
from Datasets.transformation import ses2poses_quat, ses2pos_quat
from evaluator.evaluate_rpe import calc_motion_error

from TartanVO import TartanVO

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import pypose as pp
from scipy.spatial.transform import Rotation

import re
import sys
import random
import argparse
from os import mkdir, makedirs
from os.path import isdir, isfile

import time
from timer import Timer
from datetime import datetime

import optuna
from optuna.trial import TrialState

import wandb
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
                        help='name of pretrained vo model (default: "")')
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
    parser.add_argument('--test-interval', type=int, default=100,
                        help='the interval for test results (default: 100)')
    parser.add_argument('--val-interval', type=int, default=20000,
                        help='the interval for validate results (default: 100)')
    parser.add_argument('--train-name', default='',
                        help='name of the training (default: "")')
    parser.add_argument('--result-dir', default='',
                        help='root directory of results (default: "")')
    parser.add_argument('--euroc-path', default='',
                        help='path to the EuRoC dataset (default: "")')
    parser.add_argument('--kitti-path', default='',
                        help='path to the KITTI dataset (default: "")')
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
    parser.add_argument('--use-stereo', type=float, default=0, 
                        help='stereo mode (default: 0) \
                                [0] monocular \
                                [1] stereo disp \
                                [2.1] multicam single feat endocer \
                                [2.2] multicam sep feat encoder')
    parser.add_argument('--fix_model_parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')
    parser.add_argument('--out-to-cml', action='store_true', default=False,
                        help='output to command line')
    parser.add_argument('--trial-num', type=int, default=10,
                        help='number of trials for optuna.')
    parser.add_argument('--enable-pruning', action='store_true', default=False,
                        help='Enable pruning for optuna.')
    parser.add_argument('--load-study', action='store_true', default=False,
                        help='load optuna study from a file')
    parser.add_argument('--study-name', default='',
                        help='the name of load study.')
    parser.add_argument('--not-write-log', action='store_true', default=False,
                        help='write log file')
    parser.add_argument('--enable-decay', action='store_true', default=False,
                        help='write log file')
    parser.add_argument('--tuning-val', default=[], nargs='+',
                        help='tuning variables for optuna (default: [])')
    parser.add_argument('--start-iter', type=int, default=1,
                        help='start iteration')
    parser.add_argument('--lr-lb', type=float, default=1e-7,
                        help='lower bound of learning rate')
    parser.add_argument('--lr-ub', type=float, default=1e-6,
                        help='upper bound of learning rate')
    parser.add_argument('--world-size', type=int, default=1,
                        help='number of processes')
    parser.add_argument('--tcp-port', type=int, default=65530,
                        help='tcp port for multi-processes')
    parser.add_argument('--only-one-traj', action='store_true', default=False,
                        help='only use one traj for fast loading when debug')
    parser.add_argument('--only-first-batch', action='store_true', default=False,
                        help='only use the first batch to test overfit')
    parser.add_argument('--stereo-data-type', default=['s', 'd'], nargs='+',
                        help='(default: [\'s\', \'d\'])')
    args = parser.parse_args()

    args.lr_decay_point = (np.array(args.lr_decay_point) * args.train_step).astype(int)
    
    return args


def create_dataset(args, DatasetType, mode='train'):
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
    if args.use_stereo==1:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transformlist.append(Normalize(mean=mean, std=std, keep_old=True))
    else:
        transformlist.append(Normalize())
    transformlist.extend([ToTensor(), SqueezeBatchDim()])
    transform = Compose(transformlist)

    dataset = MultiTrajFolderDataset(DatasetType=DatasetType, datatype_root={'tartanair':args.data_root}, 
                                    transform=transform, mode=mode, debug=args.only_one_traj)
    return dataset


def objective(trial, study_name, args, local_rank, datasets):
    timer = Timer()

    if 's' in args.stereo_data_type:
        trainsampler_sext = datasets['train_s']
        testsampler_sext = datasets['test_s']
    if 'd' in args.stereo_data_type:
        trainsampler_dext = datasets['train_d']
        testsampler_dext = datasets['test_d']
    
    print(" \nOptuna tuning val:", args.tuning_val, end='\n\n')

    if "lr" in args.tuning_val:
        lr = trial.suggest_float("lr", args.lr_lb, args.lr_ub, log=True)
    else:
        lr = args.lr

    if "extrinsic_encoder_layers" in args.tuning_val:
        extrinsic_encoder_layers = trial.suggest_int("extrinsic_encoder_layers", 1, 2)
    else:
        extrinsic_encoder_layers = 2

    if "trans_head_layers" in args.tuning_val:
        trans_head_layers = trial.suggest_int("trans_head_layers", 3, 6)
    else:
        trans_head_layers = 3
        
    if "optimizer" in args.tuning_val:
        optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    else:
        optimizer = args.vo_optimizer
        
    batch_size = args.batch_size
    
    file_name = study_name  + "_B"+str(batch_size) + "_lr"+"{:.3e}".format(lr) + "_O"+optimizer
    if args.use_stereo==2.1 or args.use_stereo==2.2:
        file_name += "_nel"+str(extrinsic_encoder_layers) + "_ntl"+str(trans_head_layers)

    print('==========================================')
    print('Local rank:', local_rank)
    print('Trial name:', file_name)
    print('Trial start at [{}]'.format(time.ctime()))
    print('')
    print('lr=', lr)
    print('batch_size=', batch_size)
    print('optimizer=', optimizer)
    print('extrinsic_encoder_layers=', extrinsic_encoder_layers)
    print('trans_head_layers=', trans_head_layers)
    print('==========================================')
    
    if args.enable_decay:
        LrDecrease = [int(args.train_step/2), int(args.train_step*3/4), int(args.train_step*7/8)]
        print('\nEnable lr decay')
        print('\tlr decay point:', LrDecrease)
        print('\tlr decay rate: ', args.lr_decay_rate)
        print('')
    else:
        print('\nDisable lr decay\n')

    if not args.not_write_log and local_rank==0:
        tb_dir = './tensorboard/' + study_name + '/' + file_name
        if not isdir(tb_dir):
            makedirs(tb_dir)
        writer = SummaryWriter(tb_dir)

        wandb.init(
            # set the wandb project where this run will be logged
            project=study_name,
            name=file_name,
            # # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "batch_size": batch_size,
                'optimizer': optimizer,
                'extrinsic_encoder_layers': extrinsic_encoder_layers,
                'trans_head_layers': trans_head_layers,
            }
        )
    else:
        writer = None

    trainroot = args.result_dir + '/' + study_name

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                        device_id=local_rank, use_stereo=args.use_stereo, correct_scale=False, fix_parts=args.fix_model_parts,
                        extrinsic_encoder_layers=extrinsic_encoder_layers, trans_head_layers=trans_head_layers, normalize_extrinsic=True)

    if optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)

    criterion = torch.nn.L1Loss()

    return_value_list = []

    for train_step_cnt in range(args.start_iter, args.train_step+1):
        # print('Start {} step {} ...'.format(args.mode, train_step_cnt))
        timer.tic('step')

        timer.tic('load')

        if 's' in args.stereo_data_type and 'd' in args.stereo_data_type:
            train_on_sext = (train_step_cnt % 2 == 0)
        else:
            train_on_sext = ('s' in args.stereo_data_type)

        if train_on_sext:
            if args.only_first_batch:
                sample = trainsampler_sext.first()
            else:
                sample = trainsampler_sext.next()
        else:
            if args.only_first_batch:
                sample = trainsampler_dext.first()
            else:
                sample = trainsampler_dext.next()

        timer.toc('load')

        timer.tic('infer')

        is_train = args.mode.startswith('train')
        res = tartanvo.run_batch(sample, is_train)
        motion = res['pose']

        timer.toc('infer')

        timer.tic('bp')

        gt_motion = sample['motion'].to(args.device)
        loss = criterion(motion, gt_motion)
        posenetOptimizer.zero_grad()
        loss.backward()
        posenetOptimizer.step()

        timer.toc('bp')

        if args.enable_decay and train_step_cnt in LrDecrease:
            lr *= args.lr_decay_rate
            for param_group in posenetOptimizer.param_groups: 
                param_group['lr'] = lr
            print('[!] lr decay to {} at step {}!'.format(lr, train_step_cnt))

        timer.toc('step')

        if local_rank != 0:
            continue
        # below section (print, test, snapshot) only runs on process 0

        if train_step_cnt <= 10 or train_step_cnt % args.print_interval == 0:
            with torch.no_grad():
                tot_loss = loss.item()
                trans_loss = criterion(motion[..., :3], gt_motion[..., :3]).item()
                rot_loss = criterion(motion[..., 3:], gt_motion[..., 3:]).item()
                rot_errs, trans_errs, rot_norms, trans_norms = \
                    calc_motion_error(gt_motion.cpu().numpy(), motion.cpu().numpy(), allow_rescale=False)
                trans_err = np.mean(trans_errs)
                rot_err = np.mean(rot_errs)
                trans_err_percent = np.mean(trans_errs / trans_norms)
                rot_err_percent = np.mean(rot_errs / rot_norms)

            if not args.not_write_log:
                writer.add_scalar('loss/train_loss', tot_loss, train_step_cnt)
                
                writer.add_scalar('loss/train_trans_loss', trans_loss, train_step_cnt)
                writer.add_scalar('loss/train_rot_loss', rot_loss, train_step_cnt)

                writer.add_scalar('error/train_trans_err', trans_err, train_step_cnt)
                writer.add_scalar('error/train_rot_err', rot_err, train_step_cnt)

                writer.add_scalar('error/train_trans_err_percent', trans_err_percent, train_step_cnt)
                writer.add_scalar('error/train_rot_err_percent', rot_err_percent, train_step_cnt)
                
                writer.add_scalar('time/time', timer.last('step'), train_step_cnt)

                wandb.log({ 
                        "training loss": loss.item(), 
                        "training trans loss": trans_loss, 
                        "training rot loss": rot_loss, 
                        "training trans err": trans_err, 
                        "training rot err": rot_err,
                        "training trans err percent": trans_err_percent, 
                        "training rot err percent": rot_err_percent
                    }, 
                    step=train_step_cnt
                )

            formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('[{}] TRAIN: step:{:07d}, loss:{:.4f}, trans_loss:{:.4f}, rot_loss:{:.4f}, trans_err:{:.4f}, rot_err:{:.4f},  lr:{:.10f}   time: total:{:.4f} ld:{:.4f} ife:{:.4f} bp:{:.4f}'.format(
                formatted_date, train_step_cnt, tot_loss, trans_loss, rot_loss, trans_err, rot_err, lr, timer.last('step'), timer.last('load'), timer.last('infer'), timer.last('bp')))
                
        else:
            if not args.not_write_log:
                writer.add_scalar('loss/loss', loss.item(), train_step_cnt)
                wandb.log({"training loss": loss.item() }, step= train_step_cnt)

        if train_step_cnt % args.test_interval == 0:
            timer.tic('test')

            if 's' in args.stereo_data_type and 'd' in args.stereo_data_type:
                test_on_sext = (train_step_cnt // args.test_interval % 2 == 0)
            else:
                test_on_sext = ('s' in args.stereo_data_type)

            motion_list = []
            gt_motion_list = []
            for i in range(10):
                if test_on_sext:
                    sample = testsampler_sext.next()
                else:
                    sample = testsampler_dext.next()

                res = tartanvo.run_batch(sample, is_train=False)
                motion = res['pose']

                gt_motion = sample['motion'].to(args.device)

                motion_list.append(motion)
                gt_motion_list.append(gt_motion)

            timer.toc('test')

            motion = torch.cat(motion_list, dim=0)
            gt_motion = torch.cat(gt_motion_list, dim=0)

            with torch.no_grad():
                test_tot_loss = criterion(motion, gt_motion).item()
                test_trans_loss = criterion(motion[..., :3], gt_motion[..., :3]).item()
                test_rot_loss = criterion(motion[..., 3:], gt_motion[..., 3:]).item()
                rot_errs, trans_errs, rot_norms, trans_norms = \
                    calc_motion_error(gt_motion.cpu().numpy(), motion.cpu().numpy(), allow_rescale=False)
                test_trans_err = np.mean(trans_errs)
                test_rot_err = np.mean(rot_errs)
                test_trans_err_percent = np.mean(trans_errs / trans_norms)
                test_rot_err_percent = np.mean(rot_errs / rot_norms)

            if not args.not_write_log:
                writer.add_scalar('loss/test_loss', test_tot_loss, train_step_cnt)
                
                writer.add_scalar('loss/test_trans_loss', test_trans_loss, train_step_cnt)
                writer.add_scalar('loss/test_rot_loss', test_rot_loss, train_step_cnt)

                writer.add_scalar('error/test_trans_err', test_trans_err, train_step_cnt)
                writer.add_scalar('error/test_rot_err', test_rot_err, train_step_cnt)
                if test_on_sext:
                    writer.add_scalar('error/test_trans_err_sext', test_trans_err, train_step_cnt)
                else:
                    writer.add_scalar('error/test_trans_err_dext', test_trans_err, train_step_cnt)

                writer.add_scalar('error/test_trans_err_percent', test_trans_err_percent, train_step_cnt)
                writer.add_scalar('error/test_rot_err_percent', test_rot_err_percent, train_step_cnt)

                writer.add_scalar('time/test_time', timer.last('test'), train_step_cnt)

                wandb.log({
                        "testing loss": test_tot_loss, 
                        "testing trans loss": test_trans_loss, 
                        "testing rot loss": test_rot_loss, 
                        "testing trans err": test_trans_err, 
                        "testing rot err": test_rot_err,
                        "testing trans err percent": test_trans_err_percent, 
                        "testing rot err percent": test_rot_err_percent
                    }, 
                    step = train_step_cnt
                )
                if test_on_sext:
                    wandb.log({"testing trans err static": test_trans_err}, step=train_step_cnt)
                else:
                    wandb.log({"testing trans err dynamic": test_trans_err}, step=train_step_cnt)

            if test_on_sext:
                if args.use_stereo==1:
                    optuna_metric = test_tot_loss
                else:
                    optuna_metric = test_trans_err

                return_value_list.append(optuna_metric)

                if args.enable_pruning:
                    trial.report(optuna_metric, train_step_cnt)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('[{}] TEST:  step:{:07d}, loss:{:.4f}, trans_loss:{:.4f}, rot_loss:{:.4f}, trans_err:{:.4f}, rot_err:{:.4f},                    time: total:{:.4f}'.format(
                formatted_date, train_step_cnt, test_tot_loss, test_trans_loss, test_rot_loss, test_trans_err, test_rot_err,                            timer.last('test')))


        if train_step_cnt % args.val_interval == 0:
            score = tartanvo.validate_model_result(args=args, train_step_cnt=train_step_cnt, writer=writer,verbose= False )
            
            if not args.not_write_log:
                wandb.log({"validation score": score}, step=train_step_cnt)

        if train_step_cnt % args.snapshot_interval == 0:
            if not isdir(trainroot+'/models/'+file_name):
                makedirs(trainroot+'/models/'+file_name)
            
            save_model_name = '{}/models/{}/{}_st{}.pkl'.format(trainroot, file_name, file_name, train_step_cnt)
            print(' \nSave model to:', save_model_name, end='\n\n')
            torch.save(tartanvo.vonet.flowPoseNet.state_dict(), save_model_name)
    
    if not args.not_write_log and local_rank==0:
        wandb.finish()
    
    # calcuatle average value of return_value_list with numpy
    return_value = np.array(return_value_list[-10:]).mean()
    return return_value


def process(local_rank, args):
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d'%args.tcp_port,
                                         rank=local_rank, world_size=args.world_size)

    print('local_rank=', local_rank)
    ngpu = torch.cuda.device_count()
    print('ngpu=', ngpu)
    infos = [torch.cuda.get_device_properties(i) for i in range(ngpu)]
    print('infos=', infos)

    torch.cuda.set_device(local_rank)
    device_name = torch.cuda.get_device_name(local_rank)

    study_name = args.train_name
    trainroot = args.result_dir + '/' + study_name

    if args.world_size > 1:
        sys.stdout = open(trainroot + "/log_P"+str(local_rank)+".txt", "w")
    
    print('==========================================')
    print('Local rank:', local_rank)
    print('Training name:', study_name)
    print('Traning start at [{}]'.format(time.ctime()))
    print('')
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('Device name:', device_name)
    print('==========================================')

    datasets = {}
    if 's' in args.stereo_data_type:
        traindataset_sext = create_dataset(args, DatasetType=(TrajFolderDatasetPVGO), mode='train')
        trainsampler_sext = LoopDataSampler(traindataset_sext, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.worker_num, distributed=True)
        datasets['train_s'] = trainsampler_sext

        testdataset_sext = create_dataset(args, DatasetType=(TrajFolderDatasetPVGO), mode='test')
        testsampler_sext  = LoopDataSampler(testdataset_sext, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.worker_num, distributed=True)
        datasets['test_s'] = testsampler_sext

    if 'd' in args.stereo_data_type:
        traindataset_dext = create_dataset(args, DatasetType=(TrajFolderDatasetMultiCam), mode='train')
        trainsampler_dext = LoopDataSampler(traindataset_dext, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.worker_num, distributed=True)
        datasets['train_d'] = trainsampler_dext

        testdataset_dext = create_dataset(args, DatasetType=(TrajFolderDatasetMultiCam), mode='test')
        testsampler_dext  = LoopDataSampler(testdataset_dext, batch_size=args.batch_size, shuffle=True, 
                                            num_workers=args.worker_num, distributed=True)
        datasets['test_d'] = testsampler_dext
    
    if args.world_size == 1 and len(args.tuning_val) > 0:
        storage_name = "sqlite:///./database/{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage_name, 
                                    load_if_exists=args.load_study, sampler=optuna.samplers.RandomSampler())

        study.optimize(lambda trial: objective(trial, study_name, args, local_rank, datasets), n_trials=args.trial_num)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print('==========================================')
        print("Study statistics: ")
        print("\tNumber of finished trials:", len(study.trials))
        print("\tNumber of pruned trials:", len(pruned_trials))
        print("\tNumber of complete trials:", len(complete_trials))
        print('')
        print('\tBest trial:')
        trial = study.best_trial
        print("\tValue:", trial.value)
        print("\tParams:")
        for key, value in trial.params.items():
            print("\t\t{}: {}".format(key, value))
        print('==========================================')
    
    else:
        start_time = time.time()

        objective(None, study_name, args, local_rank, datasets)

        end_time = time.time()
        print('\nTotal time consume:', end_time-start_time)

    sys.stdout.close()


if __name__ == "__main__":
    args = get_args()

    study_name = args.train_name
    trainroot = args.result_dir + '/' + study_name
    if not isdir(trainroot):
        makedirs(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))

    if args.world_size > 1:
        mp.spawn(process, nprocs=args.world_size, args=(args,))
    else:
        process(0, args)