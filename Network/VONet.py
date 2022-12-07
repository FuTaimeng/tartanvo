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
import torch.nn.functional as F
from .PWC import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet


class VONet(nn.Module):
    def __init__(self, fix_flow=False):
        super(VONet, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()

        if fix_flow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x[0] := left1 (B*3*H*W)
        # x[1] := left2
        # x[2] := intrisic (B*2*h*w)
        # h = H/4, w= W/4

        # import ipdb;ipdb.set_trace()
        flow = self.flowNet(x[(0, 1)]) # B*2*h*w
        y = torch.cat((flow, x[2]), dim=1)
        pose = self.flowPoseNet(y)

        return flow, pose


class StereoVONet(nn.Module):
    def __init__(self, fix_flow=False):
        super(StereoVONet, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet(inputnum=6)

        if fix_flow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x[0] := left1 (B*3*H*W)
        # x[1] := left2
        # x[2] := right1
        # x[3] := right2
        # x[4] := intrisic (B*2*h*w)
        # h = H/4, w= W/4

        # import ipdb;ipdb.set_trace()
        flow = self.flowNet(x[(0, 1)]) # B*2*h*w
        flow_lr = self.flowNet(x[(0, 2)])
        y = torch.cat((flow, flow_lr, x[4]), dim=1)
        pose = self.flowPoseNet(y)

        return flow, flow_lr, pose