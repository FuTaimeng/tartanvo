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
import numpy as np
import time

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from Network.VONet import VONet
from Network.StereoVONet import StereoVONet

class TartanVO(object):
    def __init__(self, vo_model_name=None, pose_model_name=None, flow_model_name=None, stereo_model_name=None,
                    use_imu=False, use_stereo=False, device='cuda', correct_scale=True):
        
        # import ipdb;ipdb.set_trace()
        if not use_stereo:
            self.vonet = VONet()
        else:
            stereonorm = 0.02 # the norm factor for the stereonet
            self.vonet = StereoVONet(network=1, intrinsic=True, flowNormFactor=1.0, stereoNormFactor=stereonorm, poseDepthNormFactor=0.25, 
                                        down_scale=True, config=1, fixflow=True, fixstereo=True, autoDistTarget=0.)

        # load the whole model
        if vo_model_name is not None and vo_model_name != "":
            print('load vo network...')
            self.load_model(self.vonet, 'models/' + vo_model_name)
        else:
            if flow_model_name is not None and flow_model_name != "":
                print('load pwc network...')
                # data = torch.load('models/' + flow_model_name)
                # self.vonet.flowNet.load_state_dict(data)
                self.load_model(self.vonet.flowNet, 'models/' + flow_model_name)
            if pose_model_name is not None and pose_model_name != "":
                print('load pose network...')
                self.load_model(self.vonet.flowPoseNet, 'models/' + pose_model_name)
            if use_stereo and stereo_model_name is not None and stereo_model_name != "":
                print('load stereo network...')
                self.load_model(self.vonet.stereoNet, 'models/' + stereo_model_name)
            
        self.pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32).to(device) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

        self.device = device
        self.use_imu = use_imu
        self.use_stereo = use_stereo
        self.correct_scale = correct_scale
        
        self.vonet.to(self.device)

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if 0 == len(preTrainDictTemp):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if 0 == len(preTrainDictTemp):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        return model

    def test_batch(self, sample):        
        # import ipdb;ipdb.set_trace()
        img0   = sample['img0'].to(self.device)
        img1   = sample['img1'].to(self.device)
        intrinsic = sample['intrinsic'].to(self.device)
        if self.use_stereo:
            img0_norm = sample['img0_norm'].to(self.device)
            img0_r_norm = sample['img0_r_norm'].to(self.device)
            blxfx = sample['blxfx'].view(1, 1, 1, 1).to(self.device)

        self.vonet.eval()

        with torch.no_grad():
            starttime = time.time()
            if not self.use_stereo:
                inputs = [torch.cat([img0, img1], axis=1), intrinsic]
                flow, pose = self.vonet(inputs)
            else:
                flow, disp, pose = self.vonet(img0, img1, img0_norm, img0_r_norm, intrinsic, 
                                                scale_w=1.0, scale_disp=1.0, blxfx=blxfx)
            inferencetime = time.time()-starttime
            # print("Pose inference using {}s".format(inferencetime))

            # import ipdb;ipdb.set_trace()
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            flow = flow * self.flow_norm

            if self.correct_scale:
                pose = self.handle_scale(sample, pose)

        return pose, flow

    def train_batch(self, sample):
        # import ipdb;ipdb.set_trace()
        img0 = sample['img0'].to(self.device)
        img1 = sample['img1'].to(self.device)
        intrinsic = sample['intrinsic'].to(self.device)
        inputs = [torch.cat([img0, img1], axis=1), intrinsic]

        self.vonet.train()

        starttime = time.time()
        flow, pose = self.vonet(inputs)
        inferencetime = time.time()-starttime
        # print("Pose inference using {}s".format(inferencetime))

        # import ipdb;ipdb.set_trace()
        pose = pose * self.pose_std # The output is normalized during training, now scale it back
        flow = flow * self.flow_norm

        pose = self.handle_scale(sample, pose)

        return pose, flow

    def handle_scale(self, sample, pose):
        motion_tar = None
        if self.use_imu and 'imu_motion' in sample:
            motion_tar = sample['imu_motion']
        elif 'motion' in sample:
            motion_tar = sample['motion']

        # calculate scale
        if motion_tar is not None:
            scale = torch.from_numpy(np.linalg.norm(motion_tar[:,:3], axis=1)).to(self.device)
            trans_est = pose[:,:3]
            trans_est = trans_est/torch.linalg.norm(trans_est,dim=1).view(-1,1)*scale.view(-1,1)
            pose = torch.cat((trans_est, pose[:,3:]), dim=1)
        else:
            print('    scale is not given, using 1 as the default scale value.')
        
        return pose

