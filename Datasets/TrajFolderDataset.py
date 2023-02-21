import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from os import listdir, path
from os.path import isdir, isfile

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer
from .loopDetector import gt_pose_loop_detector, bow_orb_loop_detector, adj_loop_detector
from .stopDetector import gt_vel_stop_detector
from .loopDetector import multicam_frame_selector


class TartanAirTrajFolderLoader:
    def __init__(self, datadir, sample_step=1, start_frame=0, end_frame=-1):
        ############################## load images ######################################################################
        try:
            imgfolder = datadir + '/image_left'
            files = listdir(imgfolder)
            self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles.sort()
        except:
            self.rgbfiles = None

        ############################## load stereo right images ######################################################################
        try:
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles_right.sort()
        except:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        self.focalx, self.focaly, self.centerx, self.centery = 320.0, 320.0, 320.0, 240.0
        self.baseline = 0.25

        ############################## load gt poses ######################################################################
        try:
            posefile = datadir + '/pose_left.txt'
            self.poses = np.loadtxt(posefile).astype(np.float32)
        except:
            self.poses = None

        ############################## load imu data ######################################################################
        try:
            imudir = datadir + '/imu'
            # acceleration in the body frame
            accels = np.load(imudir + '/accel_left.npy')
            # angular rate in the body frame
            gyros = np.load(imudir + '/gyro_left.npy')
            # velocity in the body frame
            vels = np.load(imudir + '/vel_body.npy')
            # velocity in the world frame
            vels_world = np.load(imudir + '/vel_left.npy')
            # # accel w/o gravity in body frame
            # accels_nograv = np.load(path.join(imudir, "accel_nograv_body.npy")).astype(np.float32)
            
            # self.accel_bias = -1.0 * np.array([0.01317811, 0.00902851, -0.00521479])
            accel_bias = -1.0 * np.array([-0.02437125, -0.00459115, -0.00392401])
            gyro_bias = -1.0 * np.array([0, 0, 0])
            self.accels = accels - accel_bias
            self.gyros = gyros - gyro_bias

            # imu_fps = 100
            self.imu_dts = np.ones(len(accels), dtype=np.float32) * 0.01
            
            # img_fps = 10
            self.rgb2imu_sync = np.arange(0, len(self.rgbfiles)) * 10

            self.rgb2imu_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

            if self.poses is not None:
                init_pos = self.poses[0, :3]
                init_rot = self.poses[0, 3:]
                init_vel = self.vels_world[0, :]
            else:
                init_pos = np.zeros(3, dtype=np.float32)
                init_rot = np.array([0, 0, 0, 1], dtype=np.float32)
                init_vel = np.zeros(3, dtype=np.float32)
            self.imu_init = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}

            self.gravity = -9.8

            self.has_imu = True
        except:
            self.has_imu = False


class TrajFolderDataset(Dataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1):
        if datatype == 'tartanair':
            loader = TartanAirTrajFolderLoader(datadir)
        
        self.rgbfiles = loader.rgbfiles
        self.num_img = len(self.rgbfiles)

        self.rgbfiles_right = loader.rgbfiles_right

        self.focalx, self.focaly, self.centerx, self.centery = loader.focalx, loader.focaly, loader.centerx, loader.centery
        self.baseline = loader.baseline

        self.poses = loader.poses

        if loader.has_imu:
            self.accles = loader.accels
            self.gyros = loader.gyros
            self.imu_dts = loader.imu_dts
            self.rgb2imu_sync = loader.rgb2imu_sync
            self.rgb2imu_pose = loader.rgb2imu_pose
            self.imu_init = loader.imu_init
            self.gravity = loader.gravity
            self.imu_motion = None
            self.has_imu = True
        else:
            self.has_imu = False

        self.transform = transform

        self.links = None
        self.num_link = 0

    def load_imu_motion(self, imu_poses):
        SEs = pos_quats2SEs(imu_poses)
        matrix = pose2motion(SEs, links=self.links)
        self.imu_motions = SEs2ses(matrix).astype(np.float32)

    def calc_motion_with_link(self, links):
        if self.poses is None:
            return None

        SEs = pos_quats2SEs(self.poses)
        matrix = pose2motion(SEs, links=links)
        motions = SEs2ses(matrix).astype(np.float32)
        return motions

    def __len__(self):
        return self.num_link

    def __getitem__(self, idx):
        res = {}

        img0 = cv2.imread(self.rgbfiles[self.links[idx][0]], cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(self.rgbfiles[self.links[idx][1]], cv2.IMREAD_UNCHANGED)
        res['img0'] = [img0]
        res['img1'] = [img1]

        if self.rgbfiles_right is not None:
            img0_r = cv2.imread(self.rgbfiles_right[self.links[idx][0]], cv2.IMREAD_UNCHANGED)
            img1_r = cv2.imread(self.rgbfiles_right[self.links[idx][1]], cv2.IMREAD_UNCHANGED)
            res['img0_r'] = [img0_r]
            res['img1_r'] = [img1_r]
            res['blxfx'] = np.array([self.focalx * self.baseline], dtype=np.float32) # used for convert disp to depth

        h, w, _ = img0.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = [intrinsicLayer]

        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]

        if self.has_imu and self.imu_motions is not None:
            res['imu_motion'] = self.imu_motions[idx]
        
        return res


class TrajFolderDatasetPVGO(TrajFolderDataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1,
                    use_loop_closure=False, use_stop_constraint=False):

        super(TrajFolderDatasetPVGO, self).__init__(datadir, datatype, transform, start_frame, end_frame)

        ############################## generate links ######################################################################
        self.links = []
        # # [loop closure] gt pose
        if use_loop_closure and self.poses is not None:
            loop_min_interval = 100
            trans_th = np.average([np.linalg.norm(self.poses[i+1, :3] - self.poses[i, :3]) for i in range(len(self.poses)-1)]) * 5
            self.links.extend(gt_pose_loop_detector(self.poses, loop_min_interval, trans_th, 5))
        # # [loop closure] bag of word (to do)
        # self.links = bow_orb_loop_detector(self.rgbfiles, loop_min_interval)
        # [loop closure] adjancent
        loop_range = 2
        loop_interval = 1
        self.links.extend(adj_loop_detector(self.num_img, loop_range, loop_interval))
        
        self.num_link = len(self.links)

        ############################## calc motions ######################################################################
        self.motions = self.calc_motion_with_link(self.links)

        ############################## pick stop frames ######################################################################
        self.stop_frames = []
        if use_stop_constraint and self.vels_world is not None:
            self.stop_frames = gt_vel_stop_detector(self.vels_world[::imu_mul], 0.02)


class TrajFolderDatasetMultiCam(TrajFolderDataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1):

        super(TrajFolderDatasetMultiCam, self).__init__(datadir, datatype, transform, start_frame, end_frame)

        ############################## generate links ######################################################################
        angle_range = (0, 5)
        trans_range = (0.1, 0.5)
        self.links = multicam_frame_selector(self.poses, trans_range, angle_range)

        self.num_link = len(self.links)

        ############################## calc extrinsics & motions ######################################################################
        self.motions = self.calc_motion_with_link(self.links[:, (0,2)])
        self.extrinsics = self.calc_motion_with_link(self.links[:, (0,1)])

    def __getitem__(self, idx):
        res = {}

        imgA = cv2.imread(self.rgbfiles[self.links[idx][0]], cv2.IMREAD_UNCHANGED)
        imgB = cv2.imread(self.rgbfiles[self.links[idx][1]], cv2.IMREAD_UNCHANGED)
        imgC = cv2.imread(self.rgbfiles[self.links[idx][2]], cv2.IMREAD_UNCHANGED)
        res['img0'] = [imgA]
        res['img1'] = [imgC]
        res['img0_r'] = [imgB]

        h, w, _ = imgA.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = [intrinsicLayer]

        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]

        if self.has_imu and self.imu_motions is not None:
            res['imu_motion'] = self.imu_motions[idx]

        if self.extrinsics is not None:
            res['extrinsic'] = self.extrinsics[idx]
        
        return res


class MultiTrajFolderDataset(Dataset):
    def __init__(self, DatasetType, dataroot, transform=None):
        self.dataroot = dataroot
        self.mode = 'train'

        folder_list = []
        folder_list.extend(self.list_tartanair_folders())

        self.datasets = []
        self.accmulatedDataSize = [0]
        for folder, datatype in folder_list:
            print('Loading dataset at {} ...'.format(folder))
            dataset = DatasetType(datadir=folder, datatype=datatype, transform=transform)
            self.datasets.append(dataset)
            self.accmulatedDataSize.append(self.accmulatedDataSize[-1] + len(dataset))
        
        print('Find {} datasets. Have {} frames in total.'.format(len(self.datasets), self.accmulatedDataSize[-1]))

    def list_tartanair_folders(self):
        res = []

        scenedirs = '''
            abandonedfactory        gascola        office         seasonsforest_winter
            abandonedfactory_night  hospital       office2        soulcity
            amusement               japanesealley  oldtown        westerndesert
            carwelding              neighborhood   seasidetown
            endofworld              ocean          seasonsforest
        '''
        scenedirs = scenedirs.replace('\n', '').split()
        if self.mode == 'train':
            scenedirs = scenedirs[0:10]

        # scenedirs = ['abandonedfactory', 'endofworld', 'hospital', 'office', 'ocean', 'seasidetown']
        # scenedirs = listdir(dataroot)

        for scene in scenedirs:
            for level in ['Easy']:
                trajdirs = listdir('{}/{}/{}'.format(self.dataroot, scene, level))
                # trajdirs = ['P000']
                for traj in trajdirs:
                    if not (len(traj)==4 and traj.startswith('P0')):
                        continue
                    folder = '{}/{}/{}/{}'.format(self.dataroot, scene, level, traj)
                    res.append([folder, 'tartanair'])
        
        return res
    
    def __len__(self):
        return self.accmulatedDataSize[-1]

    def __getitem__(self, idx):
        ds_idx = 0
        while idx >= self.accmulatedDataSize[ds_idx]:
            ds_idx += 1
        ds_idx -= 1

        return self.datasets[ds_idx][idx - self.accmulatedDataSize[ds_idx]]

    def list_all_frames(self):
        all_frames = []
        for ds in self.datasets:
            rgb = np.array([fname.replace(self.dataroot, '') for fname in ds.rgbfiles])
            all_frames.extend(rgb[ds.links].tolist())
        return all_frames