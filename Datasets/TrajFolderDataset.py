import pypose as pp
import numpy as np
import pandas
import yaml
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
        imgfolder = datadir + '/image_left'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_right'):
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        self.intrinsic = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0,   0, 0, 0, 1])

        ############################## load gt poses ######################################################################
        posefile = datadir + '/pose_left.txt'
        self.poses = np.loadtxt(posefile).astype(np.float32)

        ############################## load imu data ######################################################################
        if isdir(datadir + '/imu'):
            imudir = datadir + '/imu'
            # acceleration in the body frame
            accels = np.load(imudir + '/accel_left.npy')
            # angular rate in the body frame
            gyros = np.load(imudir + '/gyro_left.npy')
            # velocity in the world frame
            vels = np.load(imudir + '/vel_left.npy')
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

            self.rgb2imu_pose = pp.SE3([0, 0, 0,   0, 0, 0, 1])

            if self.poses is not None:
                init_pos = self.poses[0, :3]
                init_rot = self.poses[0, 3:]
                init_vel = self.vels[0, :]
            else:
                init_pos = np.zeros(3, dtype=np.float32)
                init_rot = np.array([0, 0, 0, 1], dtype=np.float32)
                init_vel = np.zeros(3, dtype=np.float32)
            self.imu_init = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}

            self.gravity = -9.81

            self.has_imu = True
            self.vels = vels

        else:
            self.has_imu = False
            self.vals = None


class EuRoCTrajFolderLoader:
    def __init__(self, datadir, sample_step=1, start_frame=0, end_frame=-1):
        all_timestamps = []

        ############################## load images ######################################################################
        df = pandas.read_csv(datadir + '/cam0/data.csv')
        timestamps_left = df.values[:, 0].astype(int)
        all_timestamps.append(timestamps_left)
        self.rgbfiles = datadir + '/cam0/data/' + df.values[:, 1]

        ############################## load stereo right images ######################################################################
        if isfile(datadir + '/cam1/data.csv'):
            df = pandas.read_csv(datadir + '/cam1/data.csv')
            timestamps_right = df.values[:, 0].astype(int)
            all_timestamps.append(timestamps_right)
            self.rgbfiles_right = datadir + '/cam1/data/' + df.values[:, 1]
        else:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        with open(datadir + '/cam0/sensor.yaml') as f:
            res = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.intrinsic = np.array(res['intrinsics'])
            T_BL = np.array(res['T_BS']['data']).reshape(4, 4)
        
        if self.rgbfiles_right is not None:
            with open(datadir + '/cam1/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.intrinsic_right = np.array(res['intrinsics'])
                T_BR = np.array(res['T_BS']['data']).reshape(4, 4)
        else:
            self.intrinsic_right = None

        if self.rgbfiles_right is not None:
            T_LR = np.matmul(np.linalg.inv(T_BL), T_BR)
            self.right2left_pose = pp.from_matrix(torch.tensor(T_LR), ltype=pp.SE3_type)
        else:
            self.right2left_pose = None

        ############################## load gt poses ######################################################################
        df = pandas.read_csv(datadir + '/state_groundtruth_estimate0/data.csv')
        timestamps_pose = df.values[:, 0].astype(int)
        all_timestamps.append(timestamps_pose)
        self.poses = (df.values[:, 1:8])[:, (0,1,2, 4,5,6,3)]
        self.vels = df.values[:, 8:11]
        accel_bias = np.mean(df.values[:, 14:17], axis=0)
        gyro_bias = np.mean(df.values[:, 11:14], axis=0)

        ############################## align timestamps ######################################################################
        timestamps = set(all_timestamps[0])
        for i in range(1, len(all_timestamps)):
            timestamps = timestamps.intersection(set(all_timestamps[i]))
        self.rgbfiles = self.rgbfiles[[i for i, t in enumerate(timestamps_left) if t in timestamps]]
        if self.rgbfiles_right is not None:
            self.rgbfiles_right = self.rgbfiles_right[[i for i, t in enumerate(timestamps_left) if t in timestamps]]
        self.poses = self.poses[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        self.vels = self.vels[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        timestamps = np.array(list(timestamps))
        timestamps.sort()

        ############################## load imu data ######################################################################
        if isfile(datadir + '/imu0/data.csv'):
            df = pandas.read_csv(datadir + '/imu0/data.csv')
            timestamps_imu = df.values[:, 0].astype(int)
            accels = df.values[:, 4:7]
            gyros = df.values[:, 1:4]
            
            self.accels = accels - accel_bias
            self.gyros = gyros - gyro_bias

            self.imu_dts = np.diff(timestamps_imu) * 1e-9
            
            self.rgb2imu_sync = np.searchsorted(timestamps_imu, timestamps)

            with open(datadir + '/imu0/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                T_BI = np.array(res['T_BS']['data']).reshape(4, 4)
                T_IL = np.matmul(np.linalg.inv(T_BI), T_BL)
                self.rgb2imu_pose = pp.from_matrix(torch.tensor(T_IL), ltype=pp.SE3_type)

            if self.poses is not None:
                init_pos = self.poses[0, :3]
                init_rot = self.poses[0, 3:]
                init_vel = self.vels[0, :]
            else:
                init_pos = np.zeros(3, dtype=np.float32)
                init_rot = np.array([0, 0, 0, 1], dtype=np.float32)
                init_vel = np.zeros(3, dtype=np.float32)
            self.imu_init = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}

            self.gravity = 9.81

            self.has_imu = True

        else:
            self.has_imu = False


class TrajFolderDataset(Dataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1):
        if datatype == 'tartanair':
            loader = TartanAirTrajFolderLoader(datadir)
        elif datatype == 'euroc':
            loader = EuRoCTrajFolderLoader(datadir)

        if end_frame <= 0:
            end_frame += len(loader.rgbfiles)
        
        self.rgbfiles = loader.rgbfiles[start_frame:end_frame]
        self.num_img = len(self.rgbfiles)

        self.rgbfiles_right = loader.rgbfiles_right
        if self.rgbfiles_right is not None:
            self.rgbfiles_right = self.rgbfiles_right[start_frame:end_frame]

        self.intrinsic = loader.intrinsic
        self.intrinsic_right = loader.intrinsic_right
        self.right2left_pose = loader.right2left_pose

        self.poses = loader.poses[start_frame:end_frame]
        self.vels = loader.vels[start_frame:end_frame]
        if self.vels is not None:
            self.vels = self.vels[start_frame:end_frame]

        if loader.has_imu:
            self.rgb2imu_sync = loader.rgb2imu_sync[start_frame:end_frame]
            start_imu = self.rgb2imu_sync[0]
            end_imu = self.rgb2imu_sync[-1]
            self.rgb2imu_sync -= start_imu

            self.accels = loader.accels[start_imu:end_imu]
            self.gyros = loader.gyros[start_imu:end_imu]
            self.imu_dts = loader.imu_dts[start_imu:end_imu]
            
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

    def imu_pose2motion(self, imu_poses):
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
    def __init__(self, DatasetType, dataroot, transform=None, mode='train'):
        self.dataroot = dataroot
        self.mode = mode

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