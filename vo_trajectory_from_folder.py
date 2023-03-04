from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat, ses2pos_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator

# from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir
import pickle

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
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: "")')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--sample-step', type=int, default=1,
                        help='frame sample step (default: 1)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='start frame (default: 0)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='end frame (default: None)')

    args = parser.parse_args()

    return args


def save_args(filename, *args):
    with open(filename, 'wb') as f:
        pickle.dump(args, f)
        
def load_args(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir, posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,
                                        sample_step=args.sample_step, start_frame=args.start_frame, end_frame=args.end_frame)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    testname = datastr + '_' + args.model_name.split('.')[0]
    np.savetxt('results/'+testname+'_gt.txt', testDataset.poses)
    np.savetxt('results/'+testname+'_link.txt', np.array(testDataset.links), fmt='%d')

    if not isdir('results'):
        mkdir('results')
    if args.save_flow:
        flowdir = 'results/'+testname+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0

    motionlist = []
    tot_batch = len(testDataloader)
    batch_cnt = 0
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break

        batch_cnt += 1
        print('Batch {}/{} ...'.format(batch_cnt, tot_batch), end='\r')

        motions, flow = testvo.test_batch(sample)
        motionlist.extend(motions)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1

    poselist = ses2poses_quat(np.array(motionlist[:testDataset.num_img-1]))
    motionlist = ses2pos_quat(np.array(motionlist))

    np.savetxt('results/'+testname+'.txt', poselist)
    np.savetxt('results/'+testname+'_motion.txt', motionlist)

    # # calculate ATE, RPE, KITTI-RPE
    # if args.pose_file.endswith('.txt'):
    #     evaluator = TartanAirEvaluator()
    #     results = evaluator.evaluate_one_trajectory(testDataset.poses, poselist, scale=True, kittitype=(datastr=='kitti'))
    #     if datastr=='euroc':
    #         print("==> ATE: %.4f" %(results['ate_score']))
    #     else:
    #         print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

    #     # save results and visualization
    #     plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
    #     np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    #     np.savetxt('results/'+testname+'_gt.txt',results['gt_aligned'])
    
    # else:
        # np.savetxt('results/'+testname+'.txt',poselist)

    from evaluator.evaluate_rpe import calc_motion_error
    motionlist_gt = ses2pos_quat(np.array(testDataset.motions))
    R_errs, t_errs = calc_motion_error(motionlist_gt, motionlist)
    motion_err = np.concatenate((R_errs.reshape(-1,1), t_errs.reshape(-1,1)), axis=1)
    np.savetxt('results/'+testname+'_motionerr.txt', motion_err)

    from Datasets.loopDetector import generate_g2o
    # motionlist[testDataset.num_img-1:] = motionlist_gt[testDataset.num_img-1:]
    generate_g2o('results/'+testname+'.g2o', poselist, motionlist, testDataset.links)
