from __future__ import print_function

import cv2
import numpy as np
from os.path import join

from .DatasetBase import DatasetBase
from .utils import flow16to32, depth_rgba_float32
from scipy.spatial.transform import Rotation

# from DatasetBase import DatasetBase
# from utils import flow16to32, depth_rgba_float32

class TartanAirDatasetBase(DatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,depth0,depth1,flow,motion,imu", \
        modalitylens = [1,1,1,1,1,1,1,1,10], \
        imgdir="", flowdir="", depthdir="", imudir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., depth_norm = 1., \
        pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
        imu_freq = 10, \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0, \
        baseline = 0.25, \
        frame_skip = 0, frame_stride = 1):

        super(TartanAirDatasetBase, self).__init__(framelistfile=framelistfile, \
                        datatypes=datatypes, \
                        modalitylens = modalitylens, \
                        imgdir=imgdir, flowdir=flowdir, depthdir=depthdir, imudir=imudir, \
                        motionFn=motionFn, norm_trans=norm_trans, \
                        transform=transform, \
                        flow_norm=flow_norm, has_mask=has_mask, flowinv=flowinv, \
                        disp_norm=disp_norm, depth_norm=depth_norm, \
                        pose_norm=pose_norm, \
                        imu_freq = imu_freq, \
                        intrinsic=intrinsic, focalx=focalx, focaly=focaly, centerx=centerx, centery=centery, \
                        baseline=baseline, frame_skip = frame_skip, frame_stride = frame_stride)
        if 'motion' in self.datatypelist:
            if motionFn is not None:
                self.loadMotionFromFile(motionFn, pose_norm, norm_trans)
            else:
                self.loadMotionFromPoseFile(self.trajlist, self.startframelist, self.trajlenlist, pose_norm, norm_trans)
        else:
            self.motions = None

        # self.loadMotionFromFile(motionFn, pose_norm, norm_trans)
        # self.loadMotionFromPoseFile(self.trajlist, self.startframelist, self.trajlenlist, pose_norm, norm_trans)
        if 'imu' in self.datatypelist:
            self.loadIMUFromFile(self.trajlist, self.startframelist, self.trajlenlist)

    def loadMotionFromFile(self, motionFn, pose_norm, norm_trans):
        print('Loading motion from file...')
        self.motions = np.load(motionFn).astype(np.float32)
        # Done: in this version, motion could be one less than the image frames
        assert len(self.motions)==self.acc_motionlen[-1], 'Motion file length mismatch {}, should be: {}'.format(len(self.motions), self.acc_motionlen[-1])
        self.motions = self.motions / np.array(pose_norm, dtype=np.float32)
        if norm_trans: # normalize the translation for monocular VO 
            trans = self.motions[:,:3]
            trans_norm = np.linalg.norm(trans, axis=1)
            self.motions[:,:3] = self.motions[:,:3]/trans_norm.reshape(-1,1)
        print('Loaded {} motion frames from file'.format(len(self.motions)))

    def loadMotionFromPoseFile(self, trajlist, startframelist, trajlenlist, pose_norm, norm_trans):
        print('Loading Motion data from posefiles...')
        from evaluator.transformation import pose_quats2motion_ses
        self.motionlist = []

        lastposedir = ''
        for k, trajdir in enumerate(trajlist): 
            posedir = self.flowroot + '/' + trajdir
            if lastposedir != posedir:
                poses = np.loadtxt(join(posedir,"pose_left.txt")).astype(np.float32) # framenum
                cammotions = pose_quats2motion_ses(poses, skip=self.frame_skip) # framenum - 1 - skip
            startind = startframelist[k] 
            motionnum = trajlenlist[k] - self.frame_skip -1
            # this NO LONGER assume the frame number in each traj is reduced according to the frame_skip value
            endind = startframelist[k] + motionnum  #Done: assume trajlenlist[k] is the total length of the trajectory, so that we can use same datafile for all frame_skip values
            assert endind<=len(cammotions), 'Motion calculating error: total motion number {}, startind {}, endind {}'.format(len(cammotions),startind,endind)
            assert motionnum == self.motionlenlist[k], 'Motion length error: motionnum {}, motionlenlist {}'.format(motionnum, self.motionlenlist[k])
            self.motionlist.append(cammotions[startind:endind, :])
            lastposedir = posedir
            if k%100==0:
                print('    Processed {} trajectories...'.format(k))

        self.motions = np.concatenate(self.motionlist, axis=0)    
        print('Loaded {} motion frames from file'.format(len(self.motions)))
        
        # backup the motionfile since it's quite time consuming to compute
        bkmotionfilename = 'motions_traj{}_skip{}_frame{}.npy'.format(len(trajlist),self.frame_skip, len(self.framelist))
        np.save(bkmotionfilename, self.motions)
        print('Motion file is backuped as {}'.format(bkmotionfilename))

        self.motions = self.motions / pose_norm
        if norm_trans: # normalize the translation for monocular VO 
            trans = self.motions[:,:3]
            trans_norm = np.linalg.norm(trans, axis=1)
            self.motions[:,:3] = self.motions[:,:3]/trans_norm.reshape(-1,1)

    def loadIMUFromFile(self, trajlist, startframelist, trajlenlist): 
        print('Loading IMU data from files...')
        self.accels, self.gyros, self.vels, self.oris = [], [], [], []
        self.oris6 = []
        self.accels_nograv = []
        # self.poss = []
        for k, trajdir in enumerate(trajlist): 
            imudir = self.imuroot + '/' + trajdir + '/imu' # debug
            # acceleration in the body frame
            accel = np.load(join(imudir,"accel_left.npy")).astype(np.float32)
            # angular rate in the body frame
            gyro = np.load(join(imudir,"gyro_left.npy")).astype(np.float32)
            # # velocity in the world frame
            # vel_world = np.load(join(imudir,"vel_left.npy")).astype(np.float32)
            # velocity in the body frame
            vel = np.load(join(imudir,"vel_body.npy")).astype(np.float32)
            # orientation in the world frame (in radius)
            angles_world = np.load(join(imudir,"angles_left.npy")).astype(np.float32) 
            # accel w/o gravity in body frame
            accel_nograv_body = np.load(join(imudir, "accel_nograv_body.npy")).astype(np.float32)

            angles_world_mat = Rotation.from_euler("xyz", angles_world, degrees=False).as_matrix() 
            angles_6dof = np.stack(np.split(angles_world_mat[:, 0:3, 0:2],2, axis= 2), axis=1).reshape(-1, 6).astype(np.float32)
            # # translation in the world frame
            # pose_world = np.load(join(imudir,"xyz_left.npy")).astype(np.float32)
            # mat = R.from_euler("xyz", gyro/100., degrees=False).as_matrix()
            # accelGT = accel
            # gyroGT = gyro
            startind = startframelist[k] * self.imu_freq
            endind = (startframelist[k] + trajlenlist[k] - 1) * self.imu_freq
            imunum = endind - startind
            assert endind<=len(accel), 'IMU calculating error: total IMU number {}, startind {}, endind {}'.format(len(accel),startind,endind)
            assert imunum == self.imulenlist[k], 'IMU length error: imunum {}, imulenlist {}'.format(imunum, self.motionlenlist[k])

            self.accels.append(accel[startind: endind])
            self.gyros.append(gyro[startind: endind])
            self.vels.append(vel[startind: endind])
            self.oris.append(angles_world[startind: endind])
            self.oris6.append(angles_6dof[startind: endind])
            self.accels_nograv.append(accel_nograv_body[startind: endind])
            # self.poss.append(pose_world[startind: endind])

        self.accels = np.concatenate(self.accels, axis=0)
        self.gyros = np.concatenate(self.gyros, axis=0)
        self.vels = np.concatenate(self.vels, axis=0)
        self.oris = np.concatenate(self.oris, axis=0)
        self.accels_nograv = np.concatenate(self.accels_nograv, axis=0)
        # self.oris6 = np.concatenate(self.oris6, axis=0)
        # self.poss = np.concatenate(self.poss, axis=0)
        self.imudata = np.concatenate([self.accels, self.gyros, self.vels, self.oris, self.accels_nograv], axis=1) # N x (15)
        print('Loaded {} IMU frames from file'.format(len(self.accels)))
        # import ipdb;ipdb.set_trace()

    def getDataPath(self, trajstr, framestrlist, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        return NotImplementedError

    def load_image(self, fns):
        # print('images:', fns) # for debug
        imglist = []
        for fn in fns: 
            img = cv2.imread(self.imgroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            imglist.append(img)
            assert img is not None, "Error loading image {}".format(fn)
        return imglist

    def load_motion(self, idx):
        # print('motions:', idx)
        return self.motions[idx]

    def load_imu(self, startidx, len):
        # TODO: calculate the average of adjacent frames if skipping frames
        return self.imudata[startidx: startidx+(len*(self.frame_skip+1)): self.frame_skip+1]

# TODO: flowinv are not handled correctly now that we control the skipping using frame_skip
class TartanAirDataset(TartanAirDatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,depth0,depth1,flow,motion,imu", \
        modalitylens = [1,1,1,1,1,1,1,1,10], \
        imgdir="", flowdir="", depthdir="", imudir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., depth_norm = 1., \
        pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
        imu_freq = 10, \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0, \
        baseline = 0.25, \
        frame_skip = 0, frame_stride = 1, \
        random_blur = 0.0):

        super(TartanAirDataset, self).__init__(framelistfile, \
                                    datatypes=datatypes, \
                                    modalitylens = modalitylens, \
                                    imgdir=imgdir, flowdir=flowdir, depthdir=depthdir, imudir=imudir, \
                                    motionFn=motionFn, norm_trans=norm_trans, \
                                    transform=transform, \
                                    flow_norm=flow_norm, has_mask=has_mask, flowinv=flowinv, \
                                    disp_norm=disp_norm, depth_norm=depth_norm, \
                                    pose_norm=pose_norm, \
                                    imu_freq = imu_freq, \
                                    intrinsic=intrinsic, focalx=focalx, focaly=focaly, centerx=centerx, centery=centery, \
                                    baseline=baseline, frame_skip = frame_skip, frame_stride = frame_stride)
        self.random_blur = random_blur

    def getDataPath(self, trajstr, framestrlist, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        datapathlist = []

        for framestr in framestrlist: 
            if datatype == 'img0':
                if np.random.rand() < self.random_blur and framestr!='000000':
                    datapathlist.append(trajstr + '/image_left_blur_0.5/' + framestr + '_left.png')
                else:
                    datapathlist.append(trajstr + '/image_left/' + framestr + '_left.png')
            if datatype == 'img1':
                datapathlist.append(trajstr + '/image_right/' + framestr + '_right.png')
            if datatype == 'disp0' or datatype == 'depth0':
                datapathlist.append(trajstr + '/depth_left/' + framestr + '_left_depth.png')
            if datatype == 'disp1' or datatype == 'depth1':
                datapathlist.append(trajstr + '/depth_right/' + framestr + '_right_depth.png')

            if datatype == 'flow':
                flownum = self.frame_skip + 1
                flowfolder = 'flow'
                if flownum>1:
                    flowfolder = flowfolder + str(flownum) 
                # if self.flowinv: # TODO: How to handle flowinv better? 
                #     flownum = - flownum
                framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
                datapathlist.append(trajstr + '/' + flowfolder + '/' + framestr + '_' + framestr2 + '_flow.png')

        return datapathlist

    def load_flow(self, fns):
        """This function should return 2 objects, flow and mask. """
        flow32list, masklist = [], []
        for fn in fns: 
            flow16 = cv2.imread(self.flowroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            assert flow16 is not None, "Error loading flow {}".format(fn)
            flow32, mask = flow16to32(flow16)
            flow32list.append(flow32)
            masklist.append(mask)
        return flow32list, masklist

    def load_depth(self, fns):
        depthlist = []
        for fn in fns:
            depth_rgba = cv2.imread(self.depthroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            assert depth_rgba is not None, "Error loading depth {}".format(fn)
            depth = depth_rgba_float32(depth_rgba)
            depthlist.append(depth)

        return depthlist

    def load_disparity(self, fns):
        depthlist = self.load_depth(fns)
        displist = []
        for depth in depthlist:
            disp = self.fb/depth
            displist.append(disp)
        return displist


class TartanAirDatasetNoCompress(TartanAirDatasetBase):
    def __init__(self, \
        framelistfile, \
        datatypes = "img0,img1,disp0,disp1,depth0,depth1,flow,motion,imu", \
        modalitylens = [1,1,1,1,1,1,1,1,10], \
        imgdir="", flowdir="", depthdir="", imudir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., depth_norm = 1., \
        pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
        imu_freq = 10, \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0, \
        baseline = 0.25, \
        frame_skip = 0, frame_stride = 1, \
        random_blur = 0.0):

        super(TartanAirDatasetNoCompress, self).__init__(framelistfile, \
                                    datatypes=datatypes, \
                                    modalitylens = modalitylens, \
                                    imgdir=imgdir, flowdir=flowdir, depthdir=depthdir, imudir=imudir, \
                                    motionFn=motionFn, norm_trans=norm_trans, \
                                    transform=transform, \
                                    flow_norm=flow_norm, has_mask=has_mask, flowinv=flowinv, \
                                    disp_norm=disp_norm, depth_norm=depth_norm, \
                                    pose_norm=pose_norm, \
                                    imu_freq = imu_freq, \
                                    intrinsic=intrinsic, focalx=focalx, focaly=focaly, centerx=centerx, centery=centery, \
                                    baseline=baseline, frame_skip = frame_skip, frame_stride = frame_stride)
        self.random_blur = random_blur

    def getDataPath(self, trajstr, framestrlist, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        datapathlist = []

        for framestr in framestrlist: 

            if datatype == 'img0':
                if np.random.rand() < self.random_blur and framestr!='000000':
                    datapathlist.append(trajstr + '/image_left_blur_0.5/' + framestr + '_left.png')
                else:
                    datapathlist.append(trajstr + '/image_left/' + framestr + '_left.png')
            if datatype == 'img1':
                datapathlist.append(trajstr + '/image_right/' + framestr + '_right.png')
            if datatype == 'disp0' or datatype == 'depth0':
                datapathlist.append(trajstr + '/depth_left/' + framestr + '_left_depth.npy')
            if datatype == 'disp1' or datatype == 'depth1':
                datapathlist.append(trajstr + '/depth_right/' + framestr + '_right_depth.npy')

            if datatype == 'flow':
                flownum = self.frame_skip + 1
                flowfolder = 'flow'
                if flownum>1:
                    flowfolder = flowfolder + str(flownum) 
                framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
                datapathlist.append(trajstr + '/' + flowfolder + '/' + framestr + '_' + framestr2 + '_flow.npy')

        return datapathlist

    def load_flow(self, fns):
        """This function should return 2 objects, flow and mask. """
        # print(fns) # for debug
        flowlist, masklist = [], []
        for fn in fns:
            flow = np.load(self.flowroot + '/' + fn)
            flowlist.append(flow)
            if self.flagFlowMask:
                mask = np.load(self.flowroot + '/' + fn.replace('_flow.npy', '_mask.npy'))
                masklist.append(mask)
            else:
                mask = None
                masklist = None
        return flowlist, masklist

    def load_depth(self, fns):
        # print(fns) # for debug
        # import ipdb;ipdb.set_trace()
        depthlist = []
        for fn in fns:
            depth = np.load(self.depthroot + '/' + fn)
            assert depth is not None, "Error loading depth {}".format(fn)
            depthlist.append(depth)
        return depthlist

    def load_disparity(self, fns):
        displist = []
        depthlist = self.load_depth(fns)
        for depth in depthlist:
            disp = self.fb / depth
            displist.append(disp)
        return displist

if __name__ == '__main__':
    import time
    from evaluator.transformation import pose_quats2motion_ses, sixdof2SO3, SO2so, SO2sixdof, so2SO
    # from traj_integration_visualization import integration, motion_integrate
    from scipy.spatial.transform import Rotation
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from Datasets.utils import RandomCrop, ResizeData, CropCenter, ToTensor, Compose, RandomHSV, FlipFlow, RandomResizeCrop, DownscaleFlow

    # test the efficiency of the dataloader for sequential data
    np.set_printoptions(suppress=True)    # test add noise and mask to flow
    # rootdir = '/home/wenshan/tmp/data/tartan'
    rootdir = '/home/amigo/tmp/data/tartanair_pose_and_imu'
    framefile = './data/tartan_train.txt'
    # motionFn = './data/tartan_train_flow2.npy'
    # rootdir = '/home/amigo/tmp/data/tartan'
    # framefile = 'data/tartan_train_local.txt'
    motionFn = None#'data/tartan_train_local0.npy'
    typestr = "imu,motion"#,imu,motion
    modalitylens = [200,20] #,100,10
    frame_skip = 0
    frame_stride = 1
    batch = 32
    workernum = 0

    transform = Compose((RandomResizeCrop(size=(448,640)),DownscaleFlow(),ToTensor())) # RandomHSV(HSVscale=(10,80,80),random_random=1.0), , 
    dataset = TartanAirDatasetNoCompress(framelistfile=framefile,
    # dataset = TartanAirDataset(framelistfile=framefile,
                                    datatypes = typestr, \
                                    modalitylens = modalitylens, \
                                    imgdir=rootdir, flowdir=rootdir, depthdir=rootdir, imudir=rootdir, \
                                    motionFn=motionFn, norm_trans=False, \
                                    transform= None, \
                                    flow_norm = 0.05, has_mask=False, \
                                    disp_norm = 1., depth_norm = 1.0, \
                                    pose_norm = [1.,1.,1.,1.,1.,1.], \
                                    intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0,
                                    baseline = 0.25, \
                                    frame_skip = frame_skip, frame_stride = frame_stride, \
                                    random_blur=0.)
    # print(len(dataset))
    # for k in range(len(dataset)):
    #     sample = dataset[k]
    # import ipdb;ipdb.set_trace()
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workernum)
    dataiter = iter(dataloader)
    starttime = time.time()
    lasttime = time.time()
    for k in range(100):
        try:
            sample = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            sample = dataiter.next()
        print(k,time.time()-lasttime, sample.keys(),sample['imu'].shape)
        lasttime = time.time()

        # # data visualization
        # disps = []
        # for kk in range(10):
        #     img = sample['img0'][0,kk,:].numpy().transpose(1,2,0)
        #     img = (img).astype(np.uint8)
        #     disps.append(img)
        # # import ipdb;ipdb.set_trace()
        # disps = np.array(disps)
        # disps = disps.reshape(-1,disps.shape[-2],3)
        # disps = cv2.resize(disps,(0,0),fx=0.3,fy=0.3)
        # cv2.imshow('img',disps)
        # cv2.waitKey(0)
    print("total time", time.time()-starttime)
    # imudir = '/home/wenshan/tmp/data/tartanair_pose_and_imu/abandonedfactory/Data/P000'
    # accel = np.load(join(imudir,"accel_left.npy")).astype(np.float32)
    # gyro = np.load(join(imudir,"gyro_left.npy")).astype(np.float32)
    # vel_world = np.load(join(imudir,"vel_left.npy")).astype(np.float32)
    # angles_world = np.load(join(imudir,"angles_left.npy")).astype(np.float32) 
    # poss_world = np.load(join(imudir,"xyz_left.npy")).astype(np.float32) 

    # poses = np.loadtxt(join(imudir, "pose_left.txt")).astype(np.float32)
    # motions = pose_quats2motion_ses(poses, 0)

    # vel, vel_body, vel_body_acc, pose, angles, gyro_acc = integration(accel, gyro, vel_world[0], angles_world[0], gravity = [0,0,-9.8])

    # diff_trans = motions[200:,:3] - vel_body_acc[:-200]
    # diff_rots = motions[200:, 3:] - gyro_acc[:-200]

    # print(np.abs(diff_trans).max(), np.abs(diff_trans).mean())
    # print(np.abs(diff_rots).max(), np.abs(diff_rots).mean())

    # trans = motions[:, :3]
    # rots_matrix = Rotation.from_rotvec(motions[:,3:]).as_matrix()
    # rots_euler = Rotation.from_rotvec(motions[:,3:]).as_euler("XYZ", degrees=False)
    # rots = Rotation.from_rotvec(motions[:,3:]).as_rotvec()
    # import ipdb;ipdb.set_trace()

    # # all_pose = motion_integrate(trans, rots, rate=1.)
    # pose_init = np.eye(4,4)
    # pose_init[:3, :3] = Rotation.from_euler("XYZ", angles_world[0], degrees=False).as_matrix()
    # pose_init[:3, 3] = poss_world[0]
    # all_pose = motion_integrate(trans, rots, pose_init=np.matrix(pose_init), rots_matrix=rots_matrix, rate=1.)
    # aaa=np.arange(0,100,10).tolist()
    # import ipdb;ipdb.set_trace()

    # starttime = time.time()
    # # from utils import visflow, visdepth
    # for k in range(0, len(dataset)):
    #     sample = dataset[k]
    #     # print ('{}, {}, {}'.format(k, sample['motion'], sample['imu'].shape))
    #     accel = sample['imu'][:, :3]
    #     gyro = sample['imu'][:, 3:6]
    #     vel_world = sample['imu'][:, 6:9]
    #     angles_world = sample['imu'][:, 9:12]
    #     # poss_world = sample['imu'][:, 12:15]
    #     angles_6dof_world = sample['imu'][:, 12:18]
    #     motions = sample['motion']

    #     # evaluate the angles_6dof_world
    #     # angles_world_convert == angles_world
    #     angles_world_mat_convert = sixdof2SO3(angles_6dof_world)
    #     angles_world_convert = Rotation.from_matrix(angles_world_mat_convert).as_euler("xyz", degrees=False)
    #     import ipdb;ipdb.set_trace()

    #     vel, vel_body, vel_body_acc, pose, angles, gyro_acc = integration(accel, gyro, vel_world[0], angles_world[0], gravity = [0,0,-9.8],ips = 100/(frame_skip+1))

    #     trans = motions[:, :3]
    #     rots_matrix = Rotation.from_rotvec(motions[:,3:]).as_matrix()
    #     rots_euler = Rotation.from_rotvec(motions[:,3:]).as_euler("XYZ", degrees=False)
    #     rots = Rotation.from_rotvec(motions[:,3:]).as_rotvec()
    #     # all_pose = motion_integrate(trans, rots, rate=1.)
    #     pose_init = np.eye(4,4)
    #     pose_init[:3, :3] = Rotation.from_euler("XYZ", angles_world[0], degrees=False).as_matrix()
    #     pose_init[:3, 3] = poss_world[0]
    #     all_pose = motion_integrate(trans, rots, pose_init=np.matrix(pose_init), rots_matrix=rots_matrix, rate=1.)

    #     diff_trans = motions[:,:3] - vel_body_acc
    #     diff_rots = motions[:, 3:] - gyro_acc

    #     if np.abs(diff_trans).mean()>0.005 or np.abs(diff_rots).mean() > 0.001:
    #         print('diff trans: ',np.abs(diff_trans).max(), np.abs(diff_trans).mean())
    #         print('diff rots:', np.abs(diff_rots).max(), np.abs(diff_rots).mean())
    #         import ipdb;ipdb.set_trace()

    #     if k%100==0:
    #         print(k)
            
    #     plt.subplot(221)
    #     plt.plot(vel_body_acc, 'x-')
    #     plt.subplot(223)
    #     plt.plot(motions[:,:3], 'x-')
    #     plt.subplot(222)
    #     plt.plot(gyro_acc, 'x-')
    #     plt.subplot(224)
    #     plt.plot(motions[:, 3:], 'x-')
    #     plt.show()

    #     # aaa=np.arange(0,1000,10).tolist()
    #     # plt.subplot(311)
    #     # plt.plot(poss_world[aaa]-poss_world[0], 'x-')
    #     # plt.subplot(312)
    #     # plt.plot(all_pose[:, :3, 3]-poss_world[0], 'x-')
    #     # plt.subplot(313)
    #     # plt.plot(pose[aaa], 'x-')
    #     # plt.show()
    #     # import ipdb;ipdb.set_trace()

    #     # d0 = sample['img0'][0]
    #     # d1 = sample['img0'][1]
    #     # # d1 = visdepth(sample['disp0'])
    #     # d2 = visflow(sample['flow'][0]* 20) 
    #     # # d3 = visflow(sample['flow'][1]* 20) 
    #     # # print (sample['flow_unc'].max(),sample['flow_unc'].min(),sample['flow_unc'].mean())
    #     # # import ipdb;ipdb.set_trace()
    #     # d3 = visdepth(np.exp(sample['depth0'][0]),scale=1)
    #     # dd = np.concatenate((np.concatenate((d0, d1), axis=0), np.concatenate((d2, d3), axis=0)), axis=1)
    #     # # dd = cv2.resize(dd, (640, 480))
    #     # cv2.imshow('img',dd)
    #     # cv2.waitKey(0)
    #     # trans = motions[:, :3]
    #     # rots_matrix = Rotation.from_rotvec(motions[:,3:]).as_matrix()
    #     # rots_euler = Rotation.from_rotvec(motions[:,3:]).as_euler("XYZ", degrees=False)
    #     # rots = Rotation.from_rotvec(motions[:,3:]).as_rotvec()
    #     # import ipdb;ipdb.set_trace()

    #     # # all_pose = motion_integrate(trans, rots, rate=1.)
    #     # pose_init = np.eye(4,4)
    #     # pose_init[:3, :3] = Rotation.from_euler("XYZ", angles_world[0], degrees=False).as_matrix()
    #     # pose_init[:3, 3] = poss_world[0]
    #     # all_pose = motion_integrate(trans, rots, pose_init=np.matrix(pose_init), rots_matrix=rots_matrix, rate=1.)
    #     # import ipdb;ipdb.set_trace()
    # print (time.time() - starttime)    