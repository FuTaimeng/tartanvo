from __future__ import print_function

import numpy as np
import os
import io
import cv2

from torch.utils.data import Dataset
from .utils import make_intrinsics_layer

from .pfm import readPFM_Bytes
from .FLO import read_flo_bytes
# from utils import make_intrinsics_layer

# from pfm import readPFM_Bytes
# from FLO import read_flo_bytes

class DatasetBase(Dataset):
    '''
    Loader for multi-modal data
    -----
    framelistfile: 
    TRAJNAME FRAMENUM
    FRAMESTR0
    FRAMESTR1
    ...
    -----
    Requirements: 
    The actural data path consists three parts: DATAROOT+TRAJNAME+CONVERT(FRAMESTR)
    The frames under the same TRAJNAME should be consequtive. So when the next frame is requested, it chould return the next one in the same sequence. 
    The frames should exists on the harddrive. 
    Sequential data: 
    When a sequence of data is required, the code will automatically adjust the length of the dataset, to make sure the longest modality queried exists. 
    The IMU has a higher frequency than the other modalities. The frequency is imu_freq x other_freq. 
    '''
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

        super(DatasetBase, self).__init__()
        self.framelistfile = framelistfile
        self.imgroot = imgdir 
        self.flowroot = flowdir
        self.depthroot = depthdir
        self.imuroot = imudir

        self.flow_norm = flow_norm
        self.disp_norm = disp_norm
        self.depth_norm = depth_norm
        self.pose_norm = pose_norm
        self.transform = transform

        self.imu_freq = imu_freq
        self.flagFlowMask = has_mask
        self.flowinv = flowinv
        self.intrinsic = intrinsic
        self.focalx    = focalx
        self.focaly    = focaly
        self.centerx   = centerx
        self.centery   = centery
        self.baseline = baseline
        self.fb = focalx * baseline

        self.frame_skip = frame_skip # sample not consequtively, skip a few frames within a sequences
        self.frame_stride = frame_stride # sample less sequence, skip a few frames between two sequences 

        self.datatypelist = datatypes.split(',')
        self.modalitylenlist = modalitylens
        assert len(self.datatypelist)==len(modalitylens), "Error: datatype len {}, modalitylens len {}".format(len(self.datatypelist),len(modalitylens))
        self.trajlist, self.trajlenlist, self.framelist, self.startframelist, self.motionlenlist, self.imulenlist = self.parse_inputfile(framelistfile, frame_skip)
        self.sample_seq_len = self.calc_seq_len(self.datatypelist, modalitylens, imu_freq)
        self.seqnumlist = self.parse_length(self.trajlenlist, frame_skip, frame_stride, self.sample_seq_len)


        self.framenumFromFile = len(self.framelist)
        self.N = sum(self.seqnumlist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(self.trajlenlist).tolist()
        self.acc_seqlen = [0,] + np.cumsum(self.seqnumlist).tolist() # [0, num[0], num[0]+num[1], ..]
        self.acc_motionlen = [0,] + np.cumsum(self.motionlenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        self.acc_imulen = [0,] + np.cumsum(self.imulenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        print('Loaded {} sequences from {}...'.format(self.N, framelistfile))

    def parse_inputfile(self, inputfile, frame_skip):
        '''
        trajlist: [TRAJ0, TRAJ1, ...]
        trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
        framelist: [FRAMESTR0, FRAMESTR1, ...]
        startframelist: the index of starting frames, used for calculate IMU and pose frames
                        this is to deal with the sequence in TartanAir, which got cut into subsequences, but the posefile is not cut
                        this is not very general, might be better to deprecated in the future
        motionlenlist: length of motion frames in each trajectory
                       [MotionLen0, MotionLen1, ...]
                       this is used to calculate the motion frame index in __item__()                        
        imulenlist: length of imu frames in each trajectory
                       [IMULen0, IMULen1, ...]
                       this is used to calculate the IMU frame index in __item__()                        
        '''
        with open(inputfile,'r') as f:
            lines = f.readlines()
        trajlist, trajlenlist, framelist, startframelist, motionlenlist, imulenlist = [], [], [], [], [], []
        ind = 0
        while ind<len(lines):
            line = lines[ind].strip()
            traj, trajlen = line.split(' ')
            trajlen = int(trajlen)
            trajlist.append(traj)
            trajlenlist.append(trajlen)
            motionlenlist.append(trajlen - 1 - frame_skip)
            imulenlist.append((trajlen - 1)*self.imu_freq)
            ind += 1
            for k in range(trajlen):
                if ind>=len(lines):
                    print("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                    raise Exception("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                line = lines[ind].strip()
                framelist.append(line)
                ind += 1
                if k==0:
                    startframelist.append(int(line))

        print('Read {} trajectories, including {} frames'.format(len(trajlist), len(framelist)))
        return trajlist, trajlenlist, framelist, startframelist, motionlenlist, imulenlist

    def calc_seq_len(self, datatypelist, seqlens, imu_freq):
        '''
        decide what is the sequence length for cutting the data, considering the different length of different modalities
        For now, all the modalities are at the same frequency except for the IMU which is faster by a factor of 'imu_freq'
        seqlens: the length of seq for each modality
        '''
        maxseqlen = 0
        for ttt, seqlen in zip(datatypelist, seqlens):
            if ttt=='imu': # IMU has a higher freqency than other modalities
                seqlen = int((float(seqlen+imu_freq-1)/imu_freq))
            if ttt == 'flow' or ttt == 'motion': # if seqlen of flow is equal to or bigger than other modality, add one to the seqlen
                seqlen += 1 # flow and motion need one more frame to calculate the relative change
            if seqlen > maxseqlen:
                maxseqlen = seqlen
        return maxseqlen

    def parse_length(self, trajlenlist, skip, stride, sample_length): 
        '''
        trajlenlist: the length of each trajectory in the dataset
        skip: skip frames within sequence
        stride: skip frames between sequence
        sample_length: the sequence length 
        Return: 
        seqnumlist: the number of sequences in each trajectory
        the length of the whole dataset is the sum of the seqnumlist
        '''
        seqnumlist = []
        # sequence length with skip frame 
        # e.g. x..x..x (sample_length=3, skip=2, seqlen_w_skip=1+(2+1)*(3-1)=7)
        seqlen_w_skip = (skip + 1) * sample_length - skip
        # import ipdb;ipdb.set_trace()
        for trajlen in trajlenlist:
            # x..x..x---------
            # ----x..x..x-----
            # --------x..x..x-
            # ---------x..x..x <== last possible sequence
            #          ^-------> this starting frame number is (trajlen - seqlen_w_skip + 1)
            # stride = 4, skip = 2, sample_length = 3, seqlen_w_skip = 7, trajlen = 16
            # seqnum = (16 - 7)/4 + 1 = 3
            seqnum = int((trajlen - seqlen_w_skip)/ stride) + 1
            if trajlen<seqlen_w_skip:
                seqnum = 0
            seqnumlist.append(seqnum)
        return seqnumlist


    def getDataPath(self, trajstr, framestrlist, datatype):
        raise NotImplementedError

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        raise NotImplementedError

    def load_image(self, fn):
        raise NotImplementedError

    def load_disparity(self, fn):
        raise NotImplementedError

    def load_depth(self, fn):
        raise NotImplementedError

    def load_motion(self, idx):
        raise NotImplementedError

    def load_imu(self, startidx, len):
        raise NotImplementedError

    def idx2traj(self, idx):
        '''
        handle the stride and the skip
        return: 1. the index of trajectory 
                2. the indexes of all the frames in a sequence
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_seqlen[k+1]:
                break

        remainingframes = (idx-self.acc_seqlen[k]) * self.frame_stride
        frameind = self.acc_trajlen[k] + remainingframes
        motionframeind = self.acc_motionlen[k] + remainingframes
        imuframeind = self.acc_imulen[k] + remainingframes * self.imu_freq

        # put all the frames in the seq into a list
        frameindlist = []
        motionindlist = []
        for w in range(self.sample_seq_len):
            frameindlist.append(frameind)
            frameind += self.frame_skip + 1
            motionindlist.append(motionframeind)
            motionframeind += self.frame_skip + 1
        return self.trajlist[k], frameindlist, motionindlist, imuframeind

    def disp2depth(self, disp):
        return self.focalx * self.baseline / disp

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # parse the idx to trajstr
        trajstr, frameindlist, motionindlist, imuframeind = self.idx2traj(idx)
        framestrlist = [self.framelist[k] for k in frameindlist]

        # # for debugging
        # for k in range(len(framestrlist)-1):
        #     str1 = framestrlist[k]
        #     str2 = framestrlist[k+1]
        #     if int(str1)+ self.frame_skip + 1 != int(str2):
        #         print (trajstr)
        #         print (frameindlist)
        #         print(framestrlist)
        # if idx % 5000 == 0:
        #     print (idx, trajstr, frameindlist, framestrlist)

        sample = {}
        h, w = None, None
        for datatype, datalen in zip(self.datatypelist, self.modalitylenlist): 
            datafilelist = self.getDataPath(trajstr, framestrlist[:datalen], datatype)
            if datatype == 'img0' or datatype == 'img1':
                imglist = self.load_image(datafilelist)
                if imglist is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(idx, trajstr, framestrlist, datafilelist))
                sample[datatype] = imglist
                h, w, _ = imglist[0].shape
            elif datatype == 'disp0' or datatype == 'disp1':
                displist = self.load_disparity(datafilelist)
                for k in range(len(displist)): 
                    displist[k] = displist[k] * self.disp_norm
                sample[datatype] = displist
                h, w = displist[0].shape
            elif datatype == 'depth0' or datatype == 'depth1':
                depthlist = self.load_depth(datafilelist)
                for k in range(len(depthlist)):
                    depthlist[k] = depthlist[k] * self.depth_norm
                sample[datatype] = depthlist
                h, w = depthlist[0].shape
            elif datatype[:4] == 'flow':
                flowlist, masklist = self.load_flow(datafilelist)
                for k in range(len(flowlist)):
                    flowlist[k] = flowlist[k] * self.flow_norm
                sample['flow'] = flowlist # do not distinguish flow flow2 anymore
                if self.flagFlowMask:
                    sample['fmask'] = masklist
                h, w, _ = flowlist[0].shape
            elif datatype == 'motion':
                motionlist = self.load_motion(motionindlist[:datalen])
                sample[datatype] = motionlist
            elif datatype == 'imu': 
                imulist = self.load_imu(imuframeind, datalen)
                sample[datatype] = imulist
            else:
                print('Unknow Datatype {}'.format(datatype))

        if self.intrinsic:
            if h is None or w is None:
                Exception("Unknow Input H/W {}".format(self.datatypelist))
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            sample['intrinsic'] = [intrinsicLayer]

        sample['blxfx'] = np.array([self.focalx * self.baseline], dtype=np.float32) # used for convert disp to depth

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

def get_container_client(connectionStr, containerName):
    from azure.storage.blob import BlobServiceClient, ContainerClient

    serviceClient   = BlobServiceClient.from_connection_string(connectionStr)
    containerClient = serviceClient.get_container_client(containerName)

    return containerClient, serviceClient

def get_azure_container_client(envString, containerString):
    # Get the connection string from the environment variable.
    connectStr = os.getenv(envString)
    # print(connectStr)

    # print("Get the container client. ")
    cClient, _ = get_container_client( connectStr, containerString )

    return cClient

class AzureDataLoaderBase():
    def __init__(self, ):
        accStrEnv = 'AZURE_STORAGE_CONNECTION_STRING'
        containerStr = 'tartanairdataset'
        self.azCClient = get_azure_container_client(accStrEnv, containerStr)

        self.maxTrial = 10

    def set_max_trial(self, n):
        assert (n > 0), "n must be positive. {} encountered. ".format(n)

        self.maxTrial = n

    def download_npy(self, fn):
        '''
        return a numpy array given the file path in the Azure container.
        '''
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            f = np.load(e)
        except Exception as ex:
            print('npy: Exception: {}'.format(ex))
            return None

        return f

    def download_image(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            decoded = cv2.imdecode( \
                np.asarray( bytearray( e.read() ), dtype=np.uint8 ), \
                cv2.IMREAD_UNCHANGED )
        except Exception as ex:
            print('image: Exception: {}'.format(ex))
            return None

        return decoded

    def download_pfm(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            p = readPFM_Bytes(e)
        except Exception as ex:
            print('pfm: Exception: {}'.format(ex))
            return None

        return p

    def download_flo(self, fn):
        try:
            b = self.azCClient.get_blob_client(blob=fn)
            d = b.download_blob()
            e = io.BytesIO(d.content_as_bytes())
            p = read_flo_bytes(e)
        except Exception as ex:
            print('flo: Exception: {}'.format(ex))
            return None

        return p

if __name__ == '__main__':
    framelistfile = '../data/tartan_train.txt'
    dataset = DatasetBase(framelistfile, \
        datatypes = "img0,imu", \
        modalitylens = [10,100], \
        imgdir="", flowdir="", depthdir="", imudir="", \
        motionFn=None, norm_trans=False, \
        transform=None, \
        flow_norm = 1., has_mask=False, flowinv = False, \
        disp_norm = 1., depth_norm = 1., \
        pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013], \
        imu_freq = 10, \
        intrinsic=False, focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0, \
        baseline = 0.25, \
        frame_skip = 0, frame_stride=5)
    print(len(dataset))
    for k in range(len(dataset)):
        dataset[k]
    # dataset[0]
    # dataset[1]
    # dataset[100]
    # dataset[1000]
    # dataset[1397]
    # dataset[1688]
    # dataset[10000]
    # dataset[len(dataset)-1]