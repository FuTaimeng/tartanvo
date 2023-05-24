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
from os import mkdir,makedirs
from os.path import isdir
from timer import Timer

from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState

import time
from datetime import datetime
import re
import sys

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
    parser.add_argument('--use-stereo', type=float, default=0, 
                        help='stereo mode (default: 0) \
                                [0] monocular \
                                [1] stereo disp \
                                [2.1] multicam single feat endocer \
                                [2.2] multicam sep feat encoder')
    parser.add_argument('--fix_model_parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')

    parser.add_argument('--out-to-cml',action='store_true', default=False,
                        help='Save output to a File')

    parser.add_argument('--trail-num', type=int, default=10,
                    help='The number of trails for optuna.')
    # rnn_hidden_size
    parser.add_argument('--rnn-hidden-size', type=int, default=1024,
                    help='The number of hidden state size for RNN.')
    # rnn_dropout_between
    parser.add_argument('--rnn-dropout-between', type=float, default=0.0,
                    help='The dropout rate between RNN layers.')
    # rnn_dropout_out
    parser.add_argument('--rnn-dropout-out', type=float, default=0.5,
                    help='The dropout rate for the output of RNN.')


    args = parser.parse_args()

    args.lr_decay_point = (np.array(args.lr_decay_point) * args.train_step).astype(int)
    
    return args





def objective(trial, study_name):

    timer = Timer()
    args = get_args()

    if args.device.startswith('cuda:'):
        torch.cuda.set_device(args.device)

    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    batch_size = args.batch_size

    # study_name = args.train_name # Unique identifier of the study.
    # storage_name = "sqlite:///{}.db".format(study_name)
    # study = optuna.create_study(study_name= study_name, direction="minimize", storage=storage_name)
    # study.optimize(lambda trial: objective(trial, study_name),  n_trials=args.trail_num)

    lr_rate =  "{:.3e}".format(lr).replace(".","_")
    batchsz_num = str(args.batch_size)

    file_name = study_name + "_B"+batchsz_num + "_lr"+ lr_rate

    print(' \n\n\nExp Name: ')
    print(file_name)
    print('lr:{} batch size : {}'.format(lr, batch_size))
    print()
    print(args)

    # trainroot = args.result_dir
    trainroot = args.result_dir + '/' +study_name

    print('\nTrain root:', trainroot)

    if not isdir(trainroot):
        mkdir(trainroot)
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))

    tb_dir = './tensorboard/' + study_name +'/' + file_name
    print('\nTensorboard dir:', tb_dir)
    if not isdir(tb_dir):
        makedirs(tb_dir)
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
    #                                             sample_step = 1, start_frame=0, end_frame=50, use_fixed_intervel_links=True)
    # trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)

    dataiter = iter(trainDataloader)

    # all_frames = trainDataset.list_all_frames()
    # np.savetxt(trainroot+'/all_frames.txt', all_frames, fmt="%s")
    # quit()

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                            device=args.device, use_stereo=args.use_stereo, correct_scale=False, fix_parts=args.fix_model_parts,args=args)
    # lr = args.lr
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=lr)

    criterion = torch.nn.L1Loss()

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
        loss = criterion(motion, gt_motion)
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
                trans_loss = criterion(motion[..., :3], gt_motion[..., :3]).item()
                rot_loss = criterion(motion[..., 3:], gt_motion[..., 3:]).item()
                rot_errs, trans_errs = calc_motion_error(gt_motion.cpu().numpy(), motion.cpu().numpy(), allow_rescale=False)
                trans_err = np.mean(trans_errs)
                rot_err = np.mean(rot_errs)

            writer.add_scalar('loss/loss', tot_loss, train_step_cnt)
            
            writer.add_scalar('loss/trans_loss', trans_loss, train_step_cnt)
            writer.add_scalar('loss/rot_loss', rot_loss, train_step_cnt)

            writer.add_scalar('error/trans_err', trans_err, train_step_cnt)
            writer.add_scalar('error/rot_err', rot_err, train_step_cnt)
            
            writer.add_scalar('time/time', timer.last('step'), train_step_cnt)

            trial.report(tot_loss, train_step_cnt)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()


            print('step:{}, loss:{}, trans_err:{}, rot_err:{},  time:{}'.format(
                train_step_cnt, tot_loss, trans_err, rot_err, timer.last('step')))
            
            if args.debug_flag != '':
                if not isdir(trainroot+'/debug'):
                    mkdir(trainroot+'/debug')
                debugdir = trainroot+'/debug/'+str(train_step_cnt)

                debugdir = args.result_dir + '/' +study_name + '/' + file_name + '/debug/' + str(train_step_cnt)
                if not isdir(debugdir):
                    makedirs(debugdir)

            if '0' in args.debug_flag:
                info = np.array([train_step_cnt, tot_loss, trans_loss, rot_loss, trans_err, rot_err, timer.last('step')])
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
            writer.add_scalar('loss/loss', loss.item(), train_step_cnt)
            print('step:{}, loss:{}'.format(train_step_cnt, loss.item()))

        if train_step_cnt % args.snapshot_interval == 0:
            if not isdir(trainroot+'/models'):
                mkdir(trainroot+'/models')
            torch.save(tartanvo.vonet.flowPoseNet.state_dict(), '{}/models/multicamvo_posenet_{}.pkl'.format(trainroot, train_step_cnt))

    return loss.item()


if __name__ == "__main__":
    
    args = get_args()
    print('debug_flag: ', args.debug_flag)

    torch.cuda.set_device(1)
    
    start_time = time.time()
    
    args.exp_prefix = args.train_name

    device_name = torch.cuda.get_device_name(0)
    device_name_num = str(re.findall(r'\d+', device_name)[-1]) +'_'
    current_time = datetime.now().strftime('%b_%d_%Y_%H_%M_%S')
    file_name = args.exp_prefix+"_dev"+  device_name_num+current_time
    print(' \nStudy Name: ')
    print(file_name)

    # open both files
    with open('optuna_strain_multicamvo.sh','r') as firstfile, open("./record/"+file_name+".txt", "w") as secondfile:
        # read content from first file
        for line in firstfile:
                # write content to second file
                secondfile.write(line)
    # save to file or output to command line
    if not args.out_to_cml:
        print('\n=====================')
        print('Saving to txt Files')
        print('=====================\n')
        print('\n============')
        print("Record Start")
        print('============\n')
        stdoutOrigin=sys.stdout  
        sys.stdout = open("./record/"+file_name+".txt", "a")
        print(' \n\n\nStudy Name: ')
        print(file_name)
    else:
        print('\n========================')
        print(' PRINT TO COMMAND LINE')
        print('========================\n')     
    
    now_date = time.ctime()
    print('==========================================')
    print('Traning Start at [{}]'.format(now_date) )
    print(' \ntorch.cuda.is_available() ')
    print(torch.cuda.is_available())
    print(' \nDevice Name: ')
    print(device_name)
    print('==========================================')
    
    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    study_name = file_name # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)


    study = optuna.create_study(study_name= study_name, direction="minimize", storage=storage_name)

    study.optimize(lambda trial: objective(trial, file_name),  n_trials=args.trail_num)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_trial = study.best_trial

    print('\n\n========================')
    print('      FINAL RESULT')
    print('========================\n')  

    for trial in study.trials:
        print("\nTrial number: {}".format(trial.number))
        print("Value: {}".format(trial.value))
        print("Params: {}".format(trial.params))
        print("Trial {} finished with value: {} and parameters: {}".format(trial.number, trial.value, trial.params))
        print(" Best is trial {} with value: {}.".format(best_trial.number, best_trial.value))
        print("\n")

    print("Done.")
    
    end_time = time.time()
    time_difference = int(end_time - start_time)
    end_date = time.ctime()

    print('Finished at [{}]'.format(end_date) )
    print('Total use [{}]s'.format(time_difference) )

    if not args.out_to_cml:
        sys.stdout.close()
        sys.stdout=stdoutOrigin