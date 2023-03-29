# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import time
import random
import numpy as np
import pypose as pp

from Network.VONet import VONet, MultiCamVONet
from Network.StereoVONet import StereoVONet

np.set_printoptions(precision=4, suppress=True, threshold=10000)


class TartanVO:
    def __init__(self, vo_model_name=None, pose_model_name=None, flow_model_name=None, stereo_model_name=None,
                    use_imu=False, use_stereo=0, device_id=0, correct_scale=True, fix_parts=(),
                    extrinsic_encoder_layers=2, trans_head_layers=3, normalize_extrinsic=False):
        
        # import ipdb;ipdb.set_trace()
        self.device_id = device_id
        
        if use_stereo==0:
            self.vonet = VONet()
        elif use_stereo==1:
            stereonorm = 0.02 # the norm factor for the stereonet
            self.vonet = StereoVONet(network=1, intrinsic=True, flowNormFactor=1.0, stereoNormFactor=stereonorm, poseDepthNormFactor=0.25, 
                                        down_scale=True, config=1, fixflow=True, fixstereo=True, autoDistTarget=0.)
        elif use_stereo==2.1 or use_stereo==2.2:
            self.vonet = MultiCamVONet(flowNormFactor=1.0, use_stereo=use_stereo, fix_parts=fix_parts,
                                        extrinsic_encoder_layers=extrinsic_encoder_layers, trans_head_layers=trans_head_layers)

        # load the whole model
        if vo_model_name is not None and vo_model_name != "":
            print('load vo network...')
            self.load_model(self.vonet, vo_model_name)
        # can override part of the model
        if flow_model_name is not None and flow_model_name != "":
            print('load pwc network...')
            # data = torch.load('models/' + flow_model_name)
            # self.vonet.flowNet.load_state_dict(data)
            self.load_model(self.vonet.flowNet, flow_model_name)
        if pose_model_name is not None and pose_model_name != "":
            print('load pose network...')
            self.load_model(self.vonet.flowPoseNet, pose_model_name)
        if use_stereo==1 and stereo_model_name is not None and stereo_model_name != "":
            print('load stereo network...')
            self.load_model(self.vonet.stereoNet, stereo_model_name)
            
        self.pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32).cuda() # the output scale factor
        self.flow_norm = 20 # scale factor for flow

        self.use_imu = use_imu
        self.use_stereo = use_stereo
        self.correct_scale = correct_scale
        self.normalize_extrinsic = normalize_extrinsic
        
        # self.vonet = self.vonet.cuda(device_id)
        # self.vonet = DistributedDataParallel(self.vonet, device_ids=[device_id])
        self.vonet.flowNet = self.vonet.flowNet.cuda(device_id)
        if use_stereo==1:
            self.vonet.stereoNet = self.vonet.stereoNet.cuda(device_id)
        self.vonet.flowPoseNet = DistributedDataParallel(self.vonet.flowPoseNet.cuda(device_id), device_ids=[device_id])


    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname, map_location='cuda:%d'%self.device_id)
        model_dict = model.state_dict()

        preTrainDictTemp = {}
        for k, v in preTrainDict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                preTrainDictTemp[k] = v

        # print('model_dict:')
        # for k in model_dict:
        #     print(k, model_dict[k].shape)
        # print('pretrain:')
        # for k in preTrainDict:
        #     print(k, preTrainDict[k].shape)

        if 0 == len(preTrainDictTemp):
            for k, v in preTrainDict.items():
                kk = k[7:]
                if kk in model_dict and v.size() == model_dict[kk].size():
                    preTrainDictTemp[kk] = v

        if 0 == len(preTrainDictTemp):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        for k in model_dict.keys():
            if k not in preTrainDictTemp:
                print("! [load_model] Key {} in model but not in {}!".format(k, modelname))
                if k.endswith('weight'):
                    print('\tinit with kaiming_normal_')
                    w = torch.rand_like(model_dict[k])
                    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                else:
                    print('\tinit to zeros')
                    w = torch.zeros_like(model_dict[k])
                preTrainDictTemp[k] = w

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)

        del preTrainDict
        del preTrainDictTemp

        return model


    def run_batch(self, sample, is_train=True):        
        # import ipdb;ipdb.set_trace()
        nb = False
        img0   = sample['img0'].cuda(non_blocking=nb)
        img1   = sample['img1'].cuda(non_blocking=nb)
        intrinsic = sample['intrinsic'].cuda(non_blocking=nb)

        if self.use_stereo==1:
            img0_norm = sample['img0_norm'].cuda(non_blocking=nb)
            img0_r_norm = sample['img0_r_norm'].cuda(non_blocking=nb)
            # blxfx = sample['blxfx'].view(1, 1, 1, 1).cuda(non_blocking=nb)
            blxfx = torch.tensor([0.25 * 320]).view(1, 1, 1, 1).cuda(non_blocking=nb)
        elif self.use_stereo==2.1 or self.use_stereo==2.2:
            extrinsic = sample['extrinsic'].cuda(non_blocking=nb)
            if self.normalize_extrinsic:
                extrinsic_scale = torch.linalg.norm(extrinsic[:, :3], dim=1).view(-1, 1)
                extrinsic[:, :3] /= extrinsic_scale
            img0_r = sample['img0_r'].cuda(non_blocking=nb)

        if is_train:
            self.vonet.train()
        else:
            self.vonet.eval()

        res = {}

        _ = torch.set_grad_enabled(is_train)

        if self.use_stereo==0:
            inputs = [torch.cat([img0, img1], axis=1), intrinsic]
            flow, pose = self.vonet(inputs)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            res['pose'] = pose
            res['flow'] = flow

        elif self.use_stereo==1:
            flow, disp, pose = self.vonet(img0, img1, img0_norm, img0_r_norm, intrinsic, 
                                            scale_w=1.0, scale_disp=1.0, blxfx=blxfx)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            res['pose'] = pose
            res['flow'] = flow
            res['disp'] = disp

        elif self.use_stereo==2.1 or self.use_stereo==2.2:
            flowAB, flowAC, pose = self.vonet(img0, img0_r, img1, intrinsic, extrinsic)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            if self.normalize_extrinsic:
                pose[:, :3] *= extrinsic_scale
            res['pose'] = pose
            res['flowAB'] = flowAB
            res['flowAC'] = flowAC
            
        if self.correct_scale:
            pose = self.handle_scale(sample, pose)
            res['pose'] = pose

        return res


    def handle_scale(self, sample, pose):
        motion_tar = None
        if self.use_imu and 'imu_motion' in sample:
            motion_tar = sample['imu_motion']
        elif 'motion' in sample:
            motion_tar = sample['motion']

        # calculate scale
        if motion_tar is not None:
            scale = torch.from_numpy(np.linalg.norm(motion_tar[:,:3], axis=1)).cuda()
            trans_est = pose[:,:3]
            trans_est = trans_est/torch.linalg.norm(trans_est,dim=1).view(-1,1)*scale.view(-1,1)
            pose = torch.cat((trans_est, pose[:,3:]), dim=1)
        else:
            print('    scale is not given, using 1 as the default scale value.')
        
        return pose


    # def validate_model_result(self, train_step_cnt=None,writer =None):
    #     kitti_ate, kitti_trans, kitti_rot = self.validate_model(count=train_step_cnt, writer=writer,verbose = False, datastr = 'kitti')
    #     euroc_ate = self.validate_model(count=train_step_cnt, writer = writer,verbose = False, datastr = 'euroc')

    #     print("  VAL %s #%d - KITTI-ATE/T/R/EuRoc-ATE: %.4f  %.4f  %.4f %.4f"  % (self.args.exp_prefix[:-1], 
    #     self.val_count, kitti_ate, kitti_trans, kitti_rot, euroc_ate))
    #     score = kitti_ate/ self.kitti_ate_bs/self.kitti_trans_bs + kitti_trans + kitti_rot/self.kitti_rot_bs   + euroc_ate/self.euroc_ate_bs
    #     print('score: ', score)


    # def validate_model(self,writer,count = None, verbose = False, datastr =None,source_dir = '/home/data2'):
    #     euroc_dataset = ['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02', 'V2_03']
    #     if datastr == None:
    #         print('Here is a bug!!!')

    #     args = load_args('args/args_'+datastr+'.pkl')[0]
    #     # read testdir adn posefile from kitti from tarjectory 1 to 10
    #     self.count = count

    #     result_dict = {}

    #     for i in range(11):
    #         # args.test_dir = '/data/azcopy/kitti/10/image_left'
    #         # args.pose_file = '/data/azcopy/kitti/10/pose_left.txt'
    #         if datastr == 'kitti':
    #             args.test_dir = source_dir + '/kitti/'+str(i).zfill(2)+'/image_left'
    #             args.pose_file = source_dir + '/kitti/'+str(i).zfill(2)+'/pose_left.txt'
                
    #             # Specify the path to the KITTI calib.txt file
    #             args.kitti_intrinsics_file = source_dir + '/kitti/'+str(i).zfill(2)+'/calib.txt'
    #             calib_file = source_dir + '/kitti/'+str(i).zfill(2)+'/calib.txt'
    #             focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    #             result_dict['kitti_ate'] = []
    #             result_dict['kitti_trans'] = []
    #             result_dict['kitti_rot'] = []

    #         elif datastr == 'euroc':
    #             args.test_dir = source_dir + '/euroc/'+euroc_dataset[i]+ '/cam0' + '/data2'
    #             args.pose_file = source_dir + '/euroc/'+euroc_dataset[i] + '/cam0' +'/pose_left.txt'
    #             focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
                
    #             result_dict['euroc_ate'] = []

    #         transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    #         testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
    #                                             focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,verbose = False)
            
    #         testDataset  = MultiTrajFolderDataset(DatasetType=TrajFolderDatasetMultiCam,
    #                                             root=args.data_root, transform=transform, mode = 'test')
            
    #         testDataloader  = DataLoader(testDataset,  batch_size=args.batch_size, shuffle=False,num_workers=args.worker_num)
            
    #         args.batch_size = 64
    #         args.worker_num = 4
    #         testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
    #                                             shuffle=False, num_workers=args.worker_num)
    #         testDataiter = iter(testDataloader)

    #         motionlist = []
    #         testname = datastr + '_' + args.model_name.split('.')[0]
    #         # length = len(testDataiter)

    #         motionlist_array = np.zeros((len(testDataset), 6))
    #         batch_size = args.batch_size
            
    #         for idx in tqdm(range(len(testDataiter))):    
    #             try:
    #                 sample = next(testDataiter)
    #             except StopIteration:
    #                 break
                
    #             # motions, flow = self.validate_test_batch(sample)
    #             res =  self.run_batch(sample)
    #             motions = res['pose']

    #             try:
    #                 motionlist_array[batch_size*idx:batch_size*idx+batch_size,:] = motions
    #             except:
    #                 motionlist_array[batch_size*idx:,:] = motions

    #         # poselist = ses2poses_quat(np.array(motionlist))
    #         poselist = ses2poses_quat( motionlist_array)
            
    #         # calculate ATE, RPE, KITTI-RPE
    #         # if args.pose_file.endswith('.txt'):
    #         evaluator = TartanAirEvaluator()
    #         results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
            
    #         if datastr=='kitti':
    #             result_dict['kitti_ate'].append(results['ate_score'])
    #             result_dict['kitti_trans'].append( results['kitti_score'][1]* 100  )
    #             result_dict['kitti_rot'].append( results['kitti_score'][0] * 100)
    #             print("==> KITTI: %d ATE: %.4f,\t KITTI-T/R: %.4f, %.4f" %(i, results['ate_score'], results['kitti_score'][1]* 100, results['kitti_score'][0]* 100 ))

    #         elif datastr=='euroc':
    #             result_dict['euroc_ate'].append(results['ate_score'])
    #             print("==> EuRoc: %s ATE: %.4f" %(euroc_dataset[i], results['ate_score']))
        
    #     # print average result
    #     if datastr=='euroc':
    #         ate_score = np.mean(result_dict['ate_score'])
    #         print("==> EuRoc: ATE: %.4f" %(ate_score))

    #         if not self.args.not_write_log:
    #             writer.add_scalar('Error/EuRoc_ATE', results['ate_score'], self.count)
    #             wandb.log({"EuRoc_ATE": results['ate_score']}, step=self.count)

    #         return ate_score
        
    #     elif datastr == 'kitti':
    #         ate_score = np.mean(result_dict['kitti_ate'])
    #         trans_score = np.mean(result_dict['kitti_trans'])
    #         rot_score = np.mean(result_dict['kitti_rot'])

    #         print("==> KITTI: ATE: %.4f" %(ate_score))
    #         print("==> KITTI: Trans: %.4f" %(trans_score))
    #         print("==> KITTI: Rot: %.4f" %(rot_score))

    #         if not self.args.not_write_log:
    #             writer.add_scalar('Error/KITTI_ATE', results['ate_score'], self.count)
    #             writer.add_scalar('Error/KITTI_trans', results['kitti_score'][1]* 100, self.count)
    #             writer.add_scalar('Error/KITTI_rot', results['kitti_score'][0]* 100, self.count)
    #             wandb.log({"KITTI_ATE": results['ate_score'], "KITTI_trans": results['kitti_score'][1]* 100, "KITTI_rot": results['kitti_score'][0]* 100 }, step=self.count)

    #         return ate_score, trans_score, rot_score


    # def validate_test_batch(self, sample):
    #     # self.test_count += 1
        
    #     # import ipdb;ipdb.set_trace()
    #     img0   = sample['img1'].cuda()
    #     img1   = sample['img2'].cuda()
    #     intrinsic = sample['intrinsic'].cuda()
    #     inputs = [img0, img1, intrinsic]

    #     self.vonet.eval()

    #     with torch.no_grad():
    #         starttime = time.time()

    #         imgs = torch.cat((inputs[0], inputs[1]), 1)
    #         intrinsic = inputs[2]
    #         # in tartanvo val 
    #         # flow, pose = self.vonet(inputs)
    #         # in tartanvo training
    #         flow_output, pose_output = self.vonet([imgs, intrinsic])
    #         # flow, pose = self.vonet([imgs, intrinsic])

    #         res = self.run_batch(sample)
    #         motion = res['pose']
            
    #         # print(pose)

    #         # Transfer SE3 to translation and rotation
            
    #         if pose.shape[-1] == 7:
    #             posenp,_,_ = SE32ws(pose)

    #         else:
    #             posenp = pose.data.cpu().numpy()

    #         inferencetime = time.time()-starttime
    #         # import ipdb;ipdb.set_trace()
            
    #         # Very very important
    #         posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
    #         flownp = flow.data.cpu().numpy()
    #         # flownp = flownp * self.flow_norm

    #     # calculate scale from GT posefile
    #     if 'motion' in sample:
    #         motions_gt = sample['motion']
    #         scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #         trans_est = posenp[:,:3]    
            
    #         '''
    #         trans_est_norm = np.linalg.norm(trans_est,axis=1).reshape(-1,1)
    #         eps = 1e-12 * np.ones(trans_est_norm.shape)
    #         '''
            
    #         # trans_est = trans_est/np.max(( trans_est_norm , eps)) * scale.reshape(-1,1)
    #         # trans_est = trans_est/np.max(( trans_est_norm , eps)) * scale.reshape(-1,1)

    #         posenp[:,:3] = trans_est 
    #         # print(posenp)
    #     else:
    #         print('    scale is not given, using 1 as the default scale value..')

    #     return posenp, flownp

