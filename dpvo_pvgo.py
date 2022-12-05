import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream

from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset

from torch.utils.data import DataLoader


SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, args):

    # slam = None
    # queue = Queue(maxsize=8)

    # if os.path.isdir(imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
    #     reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    # reader.start()

    # load trajectory data from a folder
    datastr = 'tartanair'
    # if args.kitti:
    #     datastr = 'kitti'
    # elif args.euroc:
    #     datastr = 'euroc'
    # else:
    #     datastr = 'tartanair'

    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    trainDataset = TrajFolderDataset(args.test_dir, posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,
                                        sample_step=args.sample_step, start_frame=args.start_frame, end_frame=args.end_frame,
                                        imudir=args.imu_dir if args.use_imu else '', img_fps=args.frame_fps, imu_mul=10,
                                        use_loop_closure=args.use_loop_closure, use_stop_constraint=args.use_stop_constraint)
    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)

    testDataiter = iter(trainDataloader)
    tot_batch = len(trainDataloader)
    batch_cnt = 0
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break
        
        batch_cnt += 1
        if args.mode.startswith('test'):
            print('Batch {}/{} ...'.format(batch_cnt, tot_batch), end='\r')

        # TODO from here

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, args.network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    # reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--resultdir')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    poses, tstamps = run(cfg, args)

    np.savetxt(args.resultdir+'/pose.txt', poses)
    np.savetxt(args.resultdir+'/tstamp.txt', tstamps)
