from torch.utils.data import DataLoader
from Datasets.utils import RandomCrop, RandomResizeCrop, RandomHSV, ToTensor, Normalize, Compose, FlipFlow, ResizeData, CropCenter, FlipStereo, RandomRotate #, CombineLR, Combine12
import numpy as np
from Datasets.data_roots import *
from os.path import isfile

class MultiDatasetsBase(object):

    def __init__(self, datafiles, datatypes, databalence, 
                       args, batch, workernum, 
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       shuffle=True):

        self.datafiles = datafiles.split(',')
        self.datatypes = datatypes.split(',')
        databalence = databalence.split(',')
        databalence = [int(tdb) for tdb in databalence]
        assert len(self.datafiles) == len(self.datatypes)
        self.numDataset = len(self.datafiles)
        self.loss_mask = [False] * self.numDataset
        self.batch = batch
        self.workernum = workernum
        self.shuffle = shuffle

        self.dataloders = []
        self.dataiters = []
        self.lossmasks = []
        self.datalens = []

        self.init_datasets(args, mean, std)

        # calculate the percentage
        if len(databalence)>1:
            assert len(databalence) == len(self.datalens)
            self.datalens = np.array(self.datalens) * np.array(databalence)
        self.accDataLens = np.cumsum(self.datalens).astype(np.float)/np.sum(self.datalens)    

    def init_datasets(self, args, mean, std):
        raise NotImplementedError

    def load_sample(self, fullbatch=True):
        # Randomly pick the dataset in the list
        randnum = np.random.rand()
        datasetInd = 0 
        while randnum > self.accDataLens[datasetInd]: # 
            datasetInd += 1

        try:
            # load sample from the dataloader
            try:
                sample = self.dataiters[datasetInd].next()
                if sample[list(sample.keys())[0]].shape[0] < self.batch and (fullbatch is True): # the imcomplete batch is thrown away
                    self.dataiters[datasetInd] = iter(self.dataloders[datasetInd])
                    sample = self.dataiters[datasetInd].next()
            except StopIteration:
                self.dataiters[datasetInd] = iter(self.dataloders[datasetInd])
                sample = self.dataiters[datasetInd].next()

            return sample, self.lossmasks[datasetInd]

        # when some datasets broken
        except Exception as e:
            print('Failed to load sample:', str(e))
            return self.load_sample(fullbatch)

def parse_datatype_vo(datatype, compressed, platform):
    # vo
    lossmask = False
    nextind = 1

    if datatype.startswith('tartan'):
        if platform=='azure':
            from Datasets.TartanAirDatasetAzure import TartanAirDatasetAzureNoCompress as DataSetType
        else:
            if compressed: # load compressed data
                from Datasets.TartanAirDataset import TartanAirDataset as DataSetType
            else:
                from Datasets.TartanAirDataset import TartanAirDatasetNoCompress as DataSetType
        lossmask = True 
        if len(datatype)>6: # tartanX
            nextind = int(datatype[6:])

    elif datatype == 'euroc':
        from Datasets.eurocDataset import EurocDataset as DataSetType
    elif datatype == 'kitti':
        from Datasets.kittiDataset import KittiVODataset as DataSetType
    else:
        print ('unknow train datatype {}!!'.format(datatype))
        assert False

    return DataSetType, lossmask, nextind

# TODO: Azure dataset? 
class StereoMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean, std):

        normalize = Normalize(mean=mean,std=std)

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            imgh, imgw = args.image_height,args.image_width
            lossmask = False
            if datatype == 'sceneflow':
                from .stereoDataset import SceneflowStereoDataset as DataSetType
            elif datatype == 'kitti':
                from .kittiDataset import KittiStereoDataset as DataSetType
                lossmask = True
                imgh = min(256, imgh)
            elif datatype == 'euroc':
                from .eurocDataset import EurocDataset as DataSetType
                lossmask = False
            elif datatype == 'tartan':
                if args.compressed:
                    from .TartanAirDataset import TartanAirDataset as DataSetType
                else:
                    from .TartanAirDataset import TartanAirDatasetNoCompress as DataSetType
            else:
                print ('unknow train datatype {}!!'.format(datatype))
                assert False

            if not args.no_data_augment:
                if args.random_intrinsic>0 and (datatype != 'kitti'): # kitti has sparse label which can not be resized. 
                    transformlist = [ RandomResizeCrop(size=(imgh, imgw), max_scale=args.random_intrinsic/320.0, 
                                                        keep_center=args.random_crop_center, fix_ratio=args.fix_ratio, scale_disp=True) ]
                else:
                    transformlist = [ RandomCrop(size=(imgh, imgw))]
                if datatype == 'kitti': # prevent resizing kitti data
                    assert imgh < 375, "KITTI image height should be less than 375!"
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
                transformlist.append(FlipStereo())
                if args.random_rotate_rightimg>0:
                    transformlist.append(RandomRotate(maxangle=args.random_rotate_rightimg))
            else: 
                transformlist = [CropCenter(size=(imgh, imgw))]
            transformlist.extend([normalize, ToTensor()])
            # if args.combine_lr:
            #     transformlist.append(CombineLR())

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot = STEREO_DR[dataset_term][platform]

            if args.no_gt:
                datastr = 'img0,img1'
                modalitylens = [1,1]
            else:
                datastr = 'img0,img1,disp0'
                modalitylens = [1,1,1]

            dataset = DataSetType(framelistfile = args.working_dir + '/' + datafile, \
                                    datatypes = datastr, \
                                    modalitylens = modalitylens, \
                                    imgdir=dataroot[0], depthdir=dataroot[1], 
                                    transform=Compose(transformlist), random_blur=args.rand_blur)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}".format(datafile, datatype,datalen))

# TODO: test the seqloader
class FlowMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            image_height,image_width = args.image_height, args.image_width
            max_intrinsic = args.random_intrinsic
            lossmask = False
            nextind = 1
            if datatype == 'sintel':
                if args.platform=='azure':
                    from .flowDatasetAzure import SintelFlowDatasetAzure as DataSetType
                else:
                    from .flowDataset import SintelFlowDataset as DataSetType
                image_height = min(image_height, 384) # hard code
            elif datatype == 'chair':
                if args.platform=='azure':
                    from .flowDatasetAzure import ChairsFlowDatasetAzure as DataSetType
                else:
                    from .flowDataset import ChairsFlowDataset as DataSetType
                image_height = min(image_height, 384) # hard code
                image_width = min(image_width, 512)
            elif datatype == 'flying' or datatype == 'flyinginv':
                if args.platform=='azure':
                    from .flowDatasetAzure import FlyingFlowDatasetAzure as DataSetType
                else:
                    from .flowDataset import FlyingFlowDataset as DataSetType
                image_height = max(image_height, 512) # hard code - decrease the difficulty of flyingthings, could cause out-of-mem error
                image_width = max(image_width, 640)
                max_intrinsic = min(370, max_intrinsic)
            # elif datatype == 'kitti': # TODO
            #     DataSetType = KITTIFlowDataset
                # lossmask = True
            elif datatype.startswith('tartan'):
                if args.platform=='azure':
                    from .TartanAirDatasetAzure import TartanAirDatasetAzureNoCompress as DataSetType
                else:
                    if args.compressed: # load compressed data
                        from .TartanAirDataset import TartanAirDataset as DataSetType
                    else:
                        from .TartanAirDataset import TartanAirDatasetNoCompress as DataSetType
                lossmask = True # set to False for now, the dynamic objects sometime are not shown in depth image
                if datatype == 'tartan2':
                    nextind = 2
                if datatype == 'tartan4':
                    nextind = 4
                if datatype == 'tartan6':
                    nextind = 6
            else:
                print ('unknow train datatype {}!!'.format(datatype))
                assert False
            # import ipdb;ipdb.set_trace()
            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOW_DR[dataset_term][platform]

            if max_intrinsic>0:
                transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=max_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else:
                transformlist = [ RandomCrop(size=(image_height,image_width)) ] 

            if not args.no_data_augment:
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
                transformlist.append(FlipFlow())

            transformlist.extend([Normalize(mean=mean,std=std),ToTensor()])
            # if args.combine_lr:
            #     transformlist.append(Combine12())

            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                    datatypes = "img0,flow", 
                                    modalitylens = [2,1], \
                                    imgdir=dataroot_list[0], flowdir=dataroot_list[1],
                                    transform=Compose(transformlist), 
                                    flow_norm = args.flow_norm, has_mask=lossmask, 
                                    flowinv = (datatype[-3:] == 'inv'), frame_skip=nextind-1, random_blur=args.rand_blur)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,image_height,image_width))

class FlowPoseMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        for datafile, datatype in zip(self.datafiles, self.datatypes):

            DataSetType, lossmask, nextind = parse_datatype_vo(datatype, args.compressed, args.platform)

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot = FLOWVO_DR[dataset_term][platform]

            if args.random_intrinsic > 0:
                transformlist = [RandomResizeCrop(size=(args.image_height, args.image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else: # No augmentation
                transformlist = [CropCenter((args.image_height, args.image_width))]

            if args.downscale_flow:    
                from Datasets.utils import DownscaleFlow
                transformlist.append(DownscaleFlow())

            if args.uncertainty:
                from Datasets.utils import RandomUncertainty
                if args.test:
                    transformlist.append(RandomUncertainty(patchnum=0))
                else: 
                    transformlist.append(RandomUncertainty())
            transformlist.append(ToTensor())

            if args.no_gt: # when testing trajectory, no gt file is available
                datastr = 'flow'
                motionFn = None
                modalitylens = [1]
            else:
                datastr = "flow,motion"
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                modalitylens = [1,1]
                if not isfile(motionFn):
                    motionFn = None

            norm_trans = (args.norm_trans_loss or args.linear_norm_trans_loss) # normalize the translation GT
            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                        datatypes = datastr, \
                                        modalitylens = modalitylens, \
                                        flowdir = dataroot, 
                                        motionFn = motionFn, norm_trans=norm_trans, 
                                        transform=Compose(transformlist), 
                                        flow_norm = 0.05, # hard code
                                        intrinsic=args.intrinsic_layer,
                                        frame_skip=nextind-1)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,args.image_height,args.image_width))


class EndToEndMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            image_height,image_width = args.image_height, args.image_width

            DataSetType, lossmask, nextind = parse_datatype_vo(datatype, args.compressed, args.platform)

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOW_DR[dataset_term][platform]

            if args.random_intrinsic>0:
                transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else:
                transformlist = [CropCenter((image_height, image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]
                # transformlist = [ RandomCrop(size=(image_height,image_width)) ]

            if datatype=='kitti':
                transformlist = [ ResizeData(size=(image_height,1226)), RandomCrop(size=(image_height,image_width)) ] # hard code

            if args.downscale_flow:
                from Datasets.utils import DownscaleFlow
                transformlist.append(DownscaleFlow())

            if not args.no_data_augment:
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
            transformlist.extend([Normalize(mean=mean,std=std),ToTensor()])

            if args.no_gt: # when testing trajectory, no gt file is available
                datastr = "img0"
                modalitylens = [2]
                motionFn = None
            else:
                datastr = "img0,motion,flow"
                modalitylens = [2,1,1]
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                if datatype=='kitti':
                    motionFn = args.working_dir + '/' + datafile.replace('_flow.txt','_motion.npy') # hard code
                    datastr = "img0,motion"
                    modalitylens = [2,1]
                if not isfile(motionFn):
                    motionFn = None

            norm_trans = (args.norm_trans_loss or args.linear_norm_trans_loss) # normalize the translation GT
            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                datatypes = datastr, \
                                modalitylens = modalitylens, \
                                imgdir = dataroot_list[0], flowdir = dataroot_list[1], 
                                motionFn = motionFn, norm_trans=norm_trans, 
                                transform=Compose(transformlist), 
                                flow_norm = args.normalize_output, has_mask=lossmask,
                                intrinsic=args.intrinsic_layer,
                                frame_skip=nextind-1, random_blur=args.rand_blur)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,image_height,image_width))

class StereoFlowPoseMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            DataSetType, lossmask, nextind = parse_datatype_vo(datatype, args.compressed, args.platform)

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot = FLOWVO_DR[dataset_term][platform]

            if args.random_intrinsic > 0:
                transformlist = [RandomResizeCrop(size=(args.image_height, args.image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio, scale_disp=False) ]
            else: # No augmentation
                transformlist = [CropCenter((args.image_height, args.image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]

            if args.downscale_flow:    
                from Datasets.utils import DownscaleFlow
                transformlist.append(DownscaleFlow()) # TODO: is this correct to replace the ResizeData? 
                # transformlist.append(ResizeData(size=(int(args.image_height/4), int(args.image_width/4))))

            if args.uncertainty:
                from Datasets.utils import RandomUncertainty
                if args.test:
                    transformlist.append(RandomUncertainty(patchnum=0))
                else: 
                    transformlist.append(RandomUncertainty())

            if args.random_scale_disp_motion:
                from Datasets.utils import RandomScaleDispMotion
                transformlist.append(RandomScaleDispMotion())

            if args.random_static>0.0:
                from Datasets.utils import StaticMotion
                transformlist.append(StaticMotion(Rate=args.random_static))

            transformlist.append(ToTensor())

            if args.no_gt:
                datastr = "flow,depth0"
                modalitylens = [1,1]
                motionFn = None
            else:
                datastr = "flow,motion,depth0"
                modalitylens = [1,1,1]
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                if not isfile(motionFn):
                    motionFn = None

            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                        datatypes = datastr, 
                                        modalitylens = modalitylens, \
                                        flowdir = dataroot, depthdir = dataroot, 
                                        motionFn = motionFn, norm_trans=False, 
                                        transform=Compose(transformlist), 
                                        flow_norm = 0.05, # hard code
                                        depth_norm = 0.25, 
                                        intrinsic=args.intrinsic_layer,
                                        frame_skip=nextind-1)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,args.image_height,args.image_width))

class EndToEndStereoMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            image_height,image_width = args.image_height, args.image_width
            DataSetType, lossmask, nextind = parse_datatype_vo(datatype, args.compressed, args.platform)

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOW_DR[dataset_term][platform]

            if args.random_intrinsic>0:
                transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio, scale_disp=False) ]
            else:
                transformlist = [CropCenter((args.image_height, args.image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]

            if args.downscale_flow:
                from .utils import DownscaleFlow # without resize rgbs
                transformlist.append(DownscaleFlow())

            if not args.no_data_augment:
                transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
            transformlist.extend([Normalize(mean=mean,std=std,keep_old=True),ToTensor()])

            if args.no_gt: # when testing trajectory, no gt file is available
                datastr = "img0,img1"
                modalitylens = [2,1]
                motionFn = None
            else:
                datastr = "img0,img1,disp0,motion,flow"
                modalitylens = [2,1,1,1,1]
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                if datatype=='kitti':
                    motionFn = args.working_dir + '/' + datafile.replace('_flow.txt','_motion.npy') # hard code
                    datastr = "img0,img1,motion,"
                    modalitylens = [2,1,1]
                if not isfile(motionFn):
                    motionFn = None

            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                datatypes = datastr, \
                                modalitylens = modalitylens, \
                                imgdir = dataroot_list[0], flowdir = dataroot_list[1], depthdir = dataroot_list[0],
                                motionFn = motionFn, norm_trans=False, 
                                transform=Compose(transformlist), 
                                flow_norm = args.normalize_output, has_mask=lossmask,
                                intrinsic=args.intrinsic_layer,
                                frame_skip = nextind-1)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,image_height,image_width))


class IMUMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        from .utils import IMUNoise, IMUNormalization

        for datafile, datatype in zip(self.datafiles, self.datatypes):
            if datatype.startswith('tartan'):
                if args.compressed: # load compressed data
                    from .TartanAirDataset import TartanAirDataset as DataSetType
                else:
                    from .TartanAirDataset import TartanAirDatasetNoCompress as DataSetType
            else:
                print ('unknow train datatype {}!!'.format(datatype))
                assert False
            # import ipdb;ipdb.set_trace()
            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot_list = FLOWVO_DR[dataset_term][platform]

            if args.imu_noise > 0.0: 
                transformlist = [ IMUNoise(args.imu_noise)] 
            else:
                transformlist = []

            if not args.no_imu_norm: 
                transformlist.append(IMUNormalization())

            motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
            if not isfile(motionFn):
                motionFn = None
            dataset = DataSetType(args.working_dir + '/' + datafile, 
                                    datatypes = "imu,motion", 
                                    modalitylens = [args.imu_input_len, int(args.imu_input_len/10)], \
                                    imudir=dataroot_list,
                                    motionFn=motionFn, norm_trans=False, \
                                    transform=Compose(transformlist), 
                                    frame_skip = args.frame_skip, frame_stride = args.imu_stride)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(False)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}".format(datafile, datatype,datalen))


class FlowVIOMultiDatasets(MultiDatasetsBase):
    '''
    Load data from different sources
    '''
    def init_datasets(self, args, mean=None, std=None):

        from Datasets.utils import IMUNoise, IMUNormalization
        for datafile, datatype in zip(self.datafiles, self.datatypes):

            DataSetType, lossmask, nextind = parse_datatype_vo(datatype, args.compressed, args.platform)

            dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
            platform = args.platform
            dataroot = FLOWVO_DR[dataset_term][platform]

            if args.random_intrinsic > 0:
                transformlist = [RandomResizeCrop(size=(args.image_height, args.image_width), max_scale=args.random_intrinsic/320.0, 
                                                    keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
            else: # No augmentation
                transformlist = [CropCenter((args.image_height, args.image_width))]

            if args.downscale_flow:    
                from Datasets.utils import DownscaleFlow
                transformlist.append(DownscaleFlow())

            # if args.uncertainty:
            #     if args.test:
            #         transformlist.append(RandomUncertainty(patchnum=0))
            #     else: 
            #         transformlist.append(RandomUncertainty())
            transformlist.append(ToTensor())

            if args.imu_noise > 0.0: 
                transformlist.append(IMUNoise(args.imu_noise))
            if not args.no_imu_norm: 
                transformlist.append(IMUNormalization())

            flowlens = int(args.imu_input_len/10) # hard code
            if args.no_gt: # when testing trajectory, no gt file is available
                datastr = "flow,imu"
                motionFn = None 
                modalitylens = [flowlens,args.imu_input_len]
            else:
                datastr = "flow,imu,motion"
                motionFn = args.working_dir + '/' + datafile.replace('.txt','.npy')
                modalitylens = [flowlens,args.imu_input_len,flowlens]
                if not isfile(motionFn):
                    motionFn = None

            dataset = DataSetType(framelistfile=args.working_dir + '/' + datafile,
                                    datatypes = datastr, \
                                    modalitylens = modalitylens, \
                                    flowdir=dataroot, imudir=dataroot, \
                                    motionFn=motionFn, norm_trans=False, \
                                    transform= Compose(transformlist), \
                                    flow_norm = 0.05, has_mask=False, \
                                    imu_freq = 10, \
                                    intrinsic=args.intrinsic_layer,
                                    frame_skip = args.frame_skip, frame_stride = args.imu_stride)

            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            dataiter = iter(dataloader)
            datalen = len(dataset)
            self.dataloders.append(dataloader)
            self.dataiters.append(dataiter)
            self.lossmasks.append(lossmask)
            self.datalens.append(datalen)

            print("Load dataset {}, data type {}, data size {}, crop size ({}, {})".format(datafile, datatype,datalen,args.image_height,args.image_width))

if __name__ == '__main__':

    class ARGS(object):
        def __init__(self):
            self.image_height = 448
            self.image_width = 640
            self.no_data_augment = False
            # self.combine_lr = False
            self.platform = 'local'
            self.working_dir = '.'
            self.image_scale = 1 
            self.random_intrinsic = 1500
            self.random_crop_center = False
            self.fix_ratio = False
            self.normalize_output = 0.05
            self.intrinsic_kitti = False
            self.norm_trans_loss = False
            self.linear_norm_trans_loss = True
            self.downscale_flow = True
            self.intrinsic_layer = False
            self.azure = False
            self.no_gt = False

            self.imu_input_len = 100
            self.imu_noise = True
            self.imu_stride = 1
            self.frame_skip = 0
            self.compressed = False

    args = ARGS()

    # ===== Test MultiDatasets ======
    # trainDataloader = MultiDatasets(['data/sceneflow_stereo_local_test.txt'], ['sceneflow'], args, 1, 1)
    # sample = trainDataloader.load_sample()


    # # ===== Test FlowMultiDatasets =====
    # from utils import visflow, tensor2img
    # import cv2
    # trainDataloader = FlowMultiDatasets(['data/tartan_flow_rgbs.txt', 'data/sintel_16.txt'], ['tartan', 'sintel'], 
    #                                         args, 2, 1, databalence=[1,10])

    # for k in range(100):
    #     sample, _ = trainDataloader.load_sample()
    #     flownp = sample['flow'].numpy()

    #     flowvis = visflow(flownp[0].transpose(1,2,0) / args.normalize_output)
    #     img1 = tensor2img(sample['img1'][0],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #     img2 = tensor2img(sample['img2'][0],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    #     flowvis_2 = visflow(flownp[1].transpose(1,2,0) / args.normalize_output)
    #     img1_2 = tensor2img(sample['img1'][1],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #     img2_2 = tensor2img(sample['img2'][1],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    #     imgdisp1 = np.concatenate((img1, img2 ,flowvis), axis=0)
    #     imgdisp2 = np.concatenate((img1_2, img2_2 ,flowvis_2), axis=0)
    #     imgdisp = np.concatenate((imgdisp1 ,imgdisp2), axis=1)
    #     imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
    #     cv2.imshow('img',imgdisp)
    #     cv2.waitKey(0)

    # # ===== Test FlowPoseMultiDatasets =====
    # from utils import visflow, tensor2img
    # import cv2
    # trainDataloader = FlowPoseMultiDatasets(['data/tartan_flow_pose_local', 'data/tartan_flow_pose_local'], ['tartan', 'tartan'], 
    #                                         args, 2, 1, databalence=[1,2])

    # for k in range(100):
    #     sample, _ = trainDataloader.load_sample()
    #     flownp = sample['flow'].numpy()

    #     flowvis = visflow(flownp[0].transpose(1,2,0) / args.normalize_output)
    #     # img1 = tensor2img(sample['img1'][0],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #     # img2 = tensor2img(sample['img2'][0],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    #     flowvis_2 = visflow(flownp[1].transpose(1,2,0) / args.normalize_output)
    #     # img1_2 = tensor2img(sample['img1'][1],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #     # img2_2 = tensor2img(sample['img2'][1],mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    #     imgdisp = np.concatenate((flowvis, flowvis_2), axis=0)
    #     imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
    #     cv2.imshow('img',imgdisp)
    #     cv2.waitKey(0)

    # # ===== Test FlowPoseMultiDatasets =====
    # from utils import visflow, tensor2img
    # import cv2
    # trainDataloader = EndToEndMultiDatasets('data/tartan_flow_rgbs3,data/tartan_flow_rgbs3', 
    #                                         'tartan,tartan',  '1,2',
    #                                         args, 2, 1, mean=[0., 0., 0.],std=[1., 1., 1.])
    # # import ipdb;ipdb.set_trace()
    # for k in range(100):
    #     sample, has_mask = trainDataloader.load_sample()
    #     flownp = sample['flow'].numpy()
    #     if has_mask:
    #         masknp = sample['fmask'].numpy()

    #     flowvis = visflow(flownp[0].transpose(1,2,0) / args.normalize_output)
    #     img1 = tensor2img(sample['img1'][0],mean=[0., 0., 0.],std=[1., 1., 1.])
    #     img2 = tensor2img(sample['img2'][0],mean=[0., 0., 0.],std=[1., 1., 1.])

    #     flowvis_2 = visflow(flownp[1].transpose(1,2,0) / args.normalize_output)
    #     img1_2 = tensor2img(sample['img1'][1],mean=[0., 0., 0.],std=[1., 1., 1.])
    #     img2_2 = tensor2img(sample['img2'][1],mean=[0., 0., 0.],std=[1., 1., 1.])

    #     if has_mask:
    #         flowvis[masknp[0][0]>128] = 0
    #         flowvis_2[masknp[1][0]>128] = 0

    #     imgdisp1 = np.concatenate((flowvis, flowvis_2), axis=0)
    #     if args.downscale_flow:
    #         imgdisp1 = cv2.resize(imgdisp1, (0,0), fx=4, fy=4)
    #     imgdisp2 = np.concatenate((img1, img1_2), axis=0)
    #     imgdisp3 = np.concatenate((img2, img2_2), axis=0)
    #     # import ipdb;ipdb.set_trace()
    #     imgdisp = cv2.resize(np.concatenate((imgdisp1, imgdisp2, imgdisp3), axis=1),(0,0),fx=0.5,fy=0.5)
    #     cv2.imshow('img',imgdisp)
    #     cv2.waitKey(0)

    # # ===== Test IMUMultiDatasets =====
    # trainDataloader = IMUMultiDatasets('data/tartan_train_local.txt,data/tartan_train_local.txt', 
    #                                         'tartan,tartan',  '1,2', args, 2, 0)

    # # import ipdb;ipdb.set_trace()
    # for k in range(100):
    #     sample, _ = trainDataloader.load_sample()
    #     imudata = sample['imu']
    #     # noisedata = sample['imu_noise']

    #     accel = imudata[:, :, :3]
    #     gyro = imudata[:, :, 3:6]
    #     vel_world = imudata[:, :, 6:9]
    #     angles_world = imudata[:, :, 9:12]

    #     print(accel.shape, gyro.shape, vel_world.shape, angles_world.shape)

    #     # accel_noise = noisedata[:, :3]
    #     # gyro_noise = noisedata[:, 3:6]
    #     # vel_world_noise = noisedata[:, 6:9]
    #     # angles_world_noise = noisedata[:, 9:12]

    # ===== Test IMUMultiDatasets =====
    trainDataloader = FlowVIOMultiDatasets('data/tartan_train_local.txt,data/tartan_train_local.txt', 
                                            'tartan,tartan',  '1,2', args, 2, 0, shuffle=True)
    from utils import visflow
    import cv2
    import ipdb;ipdb.set_trace()
    for k in range(100):
        sample, _ = trainDataloader.load_sample()
        imudata = sample['imu']
        # noisedata = sample['imu_noise']

        accel = imudata[:, :, :3]
        gyro = imudata[:, :, 3:6]
        vel_world = imudata[:, :, 6:9]
        angles_world = imudata[:, :, 9:12]

        print(accel.shape, gyro.shape, vel_world.shape, angles_world.shape)

        flownp = sample['flow'].numpy()
        masknp = sample['fmask'].numpy()

        flowvis = visflow(flownp[0][0].transpose(1,2,0) / args.normalize_output)
        flowvis_2 = visflow(flownp[0][1].transpose(1,2,0) / args.normalize_output)

        flowvis[masknp[0][0][0]>5] = 0
        flowvis_2[masknp[0][1][0]>5] = 0

        imgdisp1 = np.concatenate((flowvis, flowvis_2), axis=0)
        if args.downscale_flow:
            imgdisp1 = cv2.resize(imgdisp1, (0,0), fx=4, fy=4)
        cv2.imshow('img',imgdisp1)
        cv2.waitKey(0)
