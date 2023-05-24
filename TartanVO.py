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

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

# Data processing imports
import time
import random
import numpy as np
import pypose as pp
from Datasets.utils import (
    DownscaleFlow, ToTensor, Compose, CropCenter, SqueezeBatchDim, 
    Normalize, plot_traj, visflow, dataset_intrinsics, load_kiiti_intrinsics
)
from Datasets.TrajFolderDataset import (
    MultiTrajFolderDataset as MultiTrajFolderDataset, 
    TrajFolderDatasetPVGO as TrajFolderDatasetPVGO
)
from evaluator.tartanair_evaluator import TartanAirEvaluator
from Datasets.transformation import ses2poses_quat

# Network imports
from Network.VONet import VONet, MultiCamVONet
from Network.StereoVONet import StereoVONet

# Visualization imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Other imports
import wandb
from evaluator.evaluate_rpe import calc_motion_error
from tqdm import tqdm
from datetime import datetime


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

        self.kitti_ate_bs = 8.77
        self.kitti_trans_bs = 9.1
        self.kitti_rot_bs = 2.89
        self.euroc_ate_bs = 0.378


    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname, map_location='cuda:%d'%self.device_id)
        model_dict = model.state_dict()

        preTrainDictTemp = {}
        for k, v in preTrainDict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                preTrainDictTemp[k] = v
                print(f'load{k}...')

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
        img0 = sample['img0'].cuda(non_blocking=nb)
        img1 = sample['img1'].cuda(non_blocking=nb)
        intrinsic = sample['intrinsic'].cuda(non_blocking=nb)

        if self.use_stereo==1:
            img0_norm = sample['img0_norm'].cuda(non_blocking=nb)
            img0_r_norm = sample['img0_r_norm'].cuda(non_blocking=nb)
            # blxfx = sample['blxfx'].view(1, 1, 1, 1).cuda(non_blocking=nb)
            blxfx = torch.tensor([0.25 * 320]).view(1, 1, 1, 1).cuda(non_blocking=nb)
            scale_w = sample['scale_w'].view(-1, 1, 1, 1).cuda(non_blocking=nb)
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
                                            scale_w=scale_w, scale_disp=1.0, blxfx=blxfx)
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

    def validate_model_result(self, args, train_step_cnt=None, writer=None,verbose=False):    
        print('\nvalidating on euroc dataset...')
        euroc_ate = self.validate_model(args,count=train_step_cnt, writer=writer, verbose=verbose, datastr='euroc')

        print('\nvalidating on kitti dataset...')
        kitti_ate, kitti_trans, kitti_rot = \
            self.validate_model(args,count=train_step_cnt, writer=writer, verbose=verbose, datastr='kitti')
    
      
        formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{formatted_date}]  VAL: {train_step_cnt:07d} - KITTI-ATE/T/R/EuRoc-ATE: {kitti_ate:.4f}  {kitti_trans:.4f}  {kitti_rot:.4f} {euroc_ate:.4f}")

        
        score = kitti_ate / self.kitti_ate_bs + \
            kitti_trans/self.kitti_trans_bs + kitti_rot/self.kitti_rot_bs + euroc_ate/self.euroc_ate_bs
        if not args.not_write_log:
            wandb.log({"ValScore": score }, step=self.count)
        
        if verbose:
            print('score: ', score)
            print()

    def validate_model(self, args, writer, count=None, verbose=False, datastr=None, 
                       source_dir='/home/data2',verbose_whole = True,  verbose_each = False):
        if datastr == None:
            print('Here is a bug!!!')
        # read testdir adn posefile from kitti from tarjectory 1 to 10

        self.count = count
        result_dict = {}

        # kitti do not have trajectory 3
        # kitti_dataset_list = [0,1,2,4,5,6,7,8,9,10]
        # euroc_dataset_list = [0,1,2,3,4,5,6,7,8,9,10]

        kitti_dataset_list = [7]
        euroc_dataset_list = [0]

        if datastr == 'kitti':
            dataset_list = kitti_dataset_list
        elif datastr == 'euroc':
            dataset_list = euroc_dataset_list
        
        
        with tqdm(dataset_list, disable= True) as pbar:
            for i in pbar:
                # this dataset does not exist
                # if verbose:

                print(f'\nvalidating {datastr} trajectory {i}...')
                # if datastr == 'kitti' and i in kitti_skip_list:
                #     continue
                # if datastr == 'euroc' and i in euroc_skip_list:
                #     continue

                if datastr == 'kitti':
                    result_dict = {'kitti_ate': [],'kitti_trans': [], 'kitti_rot': []}

                    # kitti_path = "/home/data2/kitti_raw"
                    datatype_root = {'kitti': args.kitti_path}

                elif datastr == 'euroc':

                    result_dict = {'euroc_ate': []}

                    # euroc_path = "/home/data2/euroc_raw"
                    datatype_root = {'euroc': args.euroc_path}

                elif datastr == 'tartanair':

                    result_dict = {'tartanair_ate': []}
                    tartanair_path = "/home/data2/TartanAir/TartanAir_comb"
                    datatype_root = {'tartanair': tartanair_path}

                image_height = 448
                image_width = 640
                transform = Compose([CropCenter(
                    (image_height, image_width)), DownscaleFlow(),  Normalize(), ToTensor(), SqueezeBatchDim()])

                testDataset = MultiTrajFolderDataset(DatasetType=TrajFolderDatasetPVGO, datatype_root=datatype_root,
                                                        transform=transform, mode='train', debug=False, validate = True, traj_idx=i ,verbose=False)

                batch_size = 64
                worker_num = 2
                testDataloader = DataLoader(testDataset, batch_size=batch_size,
                                            shuffle=False, num_workers=worker_num)
                testDataiter = iter(testDataloader)

                motionlist_array = np.zeros((len(testDataset), 6))
                gtmotionlist_array = np.zeros((len(testDataset), 6))

                # print()
                # print(f'{datastr} raw gt shape {motionlist_array.shape}')

                # for idx in range(len(testDataiter)):
                with tqdm( range(len(testDataiter)) , disable=False) as pbar:
                    for idx in pbar:
                        try:
                            sample = next(testDataiter)
                        except StopIteration:
                            break

                        res = self.run_batch(sample)
                        motions = res['pose']
                        gtmotions = sample['motion']

                        try:
                            motionlist_array[batch_size*idx:batch_size*idx +
                                            batch_size, :] = motions.detach().cpu().numpy()
                            gtmotionlist_array[batch_size*idx:batch_size*idx +
                                            batch_size, :] = gtmotions.detach().cpu().numpy()
                        except:
                            motionlist_array[batch_size*idx:, :] = motions
                            gtmotionlist_array[batch_size*idx:, :] = gtmotions

                motionlist_array_old = motionlist_array.copy()
                val_rot_errs, val_trans_errs, rot_norms, trans_norms, motionlist_array = calc_motion_error(
                    gtmotionlist_array, motionlist_array, allow_rescale=False)
                val_trans_errs = np.mean(val_trans_errs)
                val_rot_errs = np.mean(val_rot_errs)

                val_trans_err_percent = np.mean(val_trans_errs / trans_norms)
                val_rot_err_percent = np.mean(val_rot_errs / rot_norms)
                
                if verbose_each:
                    print(
                        f'val_rot_errs {val_rot_errs:.5f} val_trans_errs {val_trans_errs:.5f} val_trans_err_percent {val_trans_err_percent:.5f} val_rot_err_percent {val_rot_err_percent:.5f}')

                poselist = ses2poses_quat(motionlist_array)
                gtposelist = ses2poses_quat(gtmotionlist_array)

                # np.savetxt(
                #     f'./test_csv/{datastr}_{i}_gtposelist.csv', gtposelist, delimiter=',')
                # np.savetxt(
                #     f'./test_csv/{datastr}_{i}_poselist.csv', poselist, delimiter=',')
                # gt_traj_ori = np.loadtxt(args.pose_file)
                # np.savetxt(
                #     f'./test_csv/{datastr}_{i}_gt_traj_ori.csv', gt_traj_ori, delimiter=',')

                # calculate ATE, RPE, KITTI-RPE
                evaluator = TartanAirEvaluator()
                testname = datastr + '_' + str(i) + '_' + 'val'

                results = evaluator.evaluate_one_trajectory(
                    gtposelist, poselist, scale=False, kittitype=(datastr == 'kitti'), verbose= verbose)

                plot_traj(results['gt_aligned'], results['est_aligned'], vis=False,
                        savefigname='results/'+testname+'_aligned'+'.png', 
                        title=f"ATE {results['ate_score']:.4f}   Scale {results['scale']:.4f}" )
                
                # plot_traj(gtposelist, poselist, vis=False,
                #           savefigname='results/' +testname+'_origin_aligned.png', title='ATE %.4f' % (results['ate_score']))
                
                results = evaluator.evaluate_one_trajectory(
                    gtposelist, poselist, scale=True, kittitype=(datastr == 'kitti'), verbose= verbose)

                plot_traj(results['gt_aligned'], results['est_aligned'], vis=False,
                        savefigname='results/'+testname+'_aligned_scaled'+'.png', 
                        title=f"ATE {results['ate_score']:.4f}   Scale {results['scale']:.4f}" )


                if datastr == 'kitti':
                    result_dict['kitti_ate'].append(results['ate_score'])
                    result_dict['kitti_trans'].append(
                        results['kitti_score'][1] * 100)
                    result_dict['kitti_rot'].append(
                        results['kitti_score'][0] * 100)
                    print(f"==> KITTI: {i} ATE: {results['ate_score']:.4f},\t KITTI-T/R: {results['kitti_score'][1] * 100:.4f}, {results['kitti_score'][0] * 100:.4f}")
                    
                elif datastr == 'euroc':
                    result_dict['euroc_ate'].append(results['ate_score'])
                    print(f"==> EuRoc: {i} ATE: {results['ate_score']:.4f}")

        # print average result
        if datastr == 'euroc':
            ate_score = np.mean(result_dict['euroc_ate'])
            if verbose:
                print("==> EuRoc: ATE: %.4f" % (ate_score))

            if not args.not_write_log:
                writer.add_scalar('Error/EuRoc_ATE',results['ate_score'], self.count)
                wandb.log({"EuRoc_ATE": results['ate_score']}, step=self.count)

            return ate_score

        elif datastr == 'kitti':
            ate_score = np.mean(result_dict['kitti_ate'])
            trans_score = np.mean(result_dict['kitti_trans'])
            rot_score = np.mean(result_dict['kitti_rot'])
            if verbose:
                print("==> KITTI: ATE: %.4f" % (ate_score))
                print("==> KITTI: Trans: %.4f" % (trans_score))
                print("==> KITTI: Rot: %.4f" % (rot_score))

            if not args.not_write_log:
                writer.add_scalar('Error/KITTI_ATE', results['ate_score'], self.count)
                writer.add_scalar('Error/KITTI_trans', results['kitti_score'][1] * 100, self.count)
                writer.add_scalar('Error/KITTI_rot', results['kitti_score'][0] * 100, self.count)
                wandb.log({"KITTI_ATE": results['ate_score'], "KITTI_trans": results['kitti_score'][1] * 100, "KITTI_rot": results['kitti_score'][0] * 100}, step=self.count)

            return ate_score, trans_score, rot_score