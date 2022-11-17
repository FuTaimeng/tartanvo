import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer
from .loopDetector import gt_pose_loop_detector, bow_orb_loop_detector, adj_loop_detector


class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder, posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0,
                    sample_step = 10, start_frame=0, end_frame=-1,
                    imudir = None, img_fps = 10.0, imu_mul = 10):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        total_num_img = len(self.rgbfiles)
        end_frame += total_num_img+1 if end_frame<0 else 1
        self.rgbfiles = self.rgbfiles[start_frame:end_frame:sample_step]
        self.num_img = len(self.rgbfiles)
        self.images = []
        for rgbname in self.rgbfiles:
            img = cv2.imread(rgbname)
            self.images.append(img)
        print('Load {} of {} image files (st:{}, end:{}, step:{}) in {}'.format(
            self.num_img, total_num_img, start_frame, end_frame, sample_step, imgfolder))
        
        if posefile is not None and posefile != "":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1] == 7) # position + quaternion
            self.poses = poselist[start_frame:end_frame:sample_step]

            # # [loop closure] gt pose
            # loop_min_interval = 100
            # loop_min_interval = int(loop_min_interval / sample_step)
            # trans_th = np.average([np.linalg.norm(poselist[i+1, :3] - poselist[i, :3]) for i in range(len(poselist)-1)]) * 5
            # self.links = gt_pose_loop_detector(self.poses, loop_min_interval, trans_th, 5)

            # # [loop closure] bag of word (to do)
            # self.links = bow_orb_loop_detector(self.rgbfiles, loop_min_interval)

            # [loop closure] adjancent
            loop_range = 2
            loop_interval = 1
            self.links = adj_loop_detector(self.num_img, loop_range, loop_interval)

            SEs = pos_quats2SEs(self.poses)
            matrix = pose2motion(SEs, links=self.links)
            self.motions = SEs2ses(matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.links))

        else:
            self.links = [(i, i+1) for i in range(self.num_img-1)]
            self.poses = None
            self.motions = None

        if imudir is not None and imudir != "":
            # acceleration in the body frame
            accels = np.load(path.join(imudir, "accel_left.npy"))
            # angular rate in the body frame
            gyros = np.load(path.join(imudir, "gyro_left.npy"))
            # velocity in the body frame
            vels = np.load(path.join(imudir,"vel_body.npy"))
            # velocity in the world frame
            vels_world = np.load(path.join(imudir,"vel_left.npy"))
            # # accel w/o gravity in body frame
            # accels_nograv = np.load(path.join(imudir, "accel_nograv_body.npy")).astype(np.float32)
            
            # self.accel_bias = -1.0 * np.array([0.01317811, 0.00902851, -0.00521479])
            self.accel_bias = -1.0 * np.array([-0.02437125, -0.00459115, -0.00392401])
            self.gyro_bias = -1.0 * np.array([0, 0, 0])
            self.accels_raw = accels[start_frame*imu_mul : (end_frame-1)*imu_mul, :].reshape(end_frame-start_frame-1, -1, 3)
            self.accels = self.accels_raw - self.accel_bias
            self.gyros_raw = gyros[start_frame*imu_mul : (end_frame-1)*imu_mul, :].reshape(end_frame-start_frame-1, -1, 3)
            self.gyros = self.gyros_raw - self.gyro_bias
            
            self.vels = vels[start_frame*imu_mul : (end_frame-1)*imu_mul+1, :]
            self.vels_world = vels_world[start_frame*imu_mul : (end_frame-1)*imu_mul+1, :]
            
            assert(self.accels.shape == self.gyros.shape)
            assert(self.vels.shape == self.vels_world.shape)
            print('Load {} of {} IMU frames in {}'.format(self.accels.shape[:2], len(accels), imudir))

            dt = 1.0/img_fps * sample_step / imu_mul
            self.imu_dts = np.full(self.accels.shape[:2], dt, dtype=np.float32)
            if self.poses is not None:
                init_pos = self.poses[0, :3]
                init_rot = self.poses[0, 3:]
                init_vel = self.vels_world[0, :]
            else:
                init_pos = np.zeros(3, dtype=np.float32)
                init_rot = np.array([0, 0, 0, 1], dtype=np.float32)
                init_vel = np.zeros(3, dtype=np.float32)
            self.imu_init = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}
            self.gravity = -9.81007

            # call load_imu_motion after imu preintegration to fill self.imu_motions
            self.imu_motions = None

        else:
            self.accels = None
            self.gyros = None
            self.imu_motions = None

        self.num_link = len(self.links)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery


    def load_imu_motion(self, imu_poses):
        SEs = pos_quats2SEs(imu_poses)
        matrix = pose2motion(SEs, links=self.links)
        self.imu_motions = SEs2ses(matrix).astype(np.float32)


    def __len__(self):
        return self.num_link

    def __getitem__(self, idx):
        img1 = self.images[self.links[idx][0]].copy()
        img2 = self.images[self.links[idx][1]].copy()

        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]

        if self.imu_motions is not None:
            res['imu_motion'] = self.imu_motions[idx]
        
        return res

