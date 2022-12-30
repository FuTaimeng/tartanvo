import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
        nn.ReLU(inplace=True)
    )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class VOFlowRes(nn.Module):
    def __init__(self, intrinsic=True, down_scale=True, config=1, stereo=0):
        super(VOFlowRes, self).__init__()
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        if stereo==1:
            inputnum += 1
        elif stereo==2:
            inputnum += 2
        
        self.down_scale = down_scale
        self.config = config
        self.stereo = stereo

        if config==0:
            blocknums = [2,2,3,3,3,3,3]
            outputnums = [32,64,64,64,128,128,128]
        elif config==1:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif config==2:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif config==3:
            blocknums = [3,4,7,9,9,5,3]
            outputnums = [32,64,128,128,256,256,512]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1),
                                       conv(      32, 32, 3, 1, 1, 1),
                                       conv(      32, 32, 3, 1, 1, 1))

        self.inplanes = 32
        if not down_scale:
            self.layer0 = self._make_layer(BasicBlock, outputnums[0], blocknums[0], 2, 1, 1) # (160 x 112)
            self.layer0_2 = self._make_layer(BasicBlock, outputnums[1], blocknums[1], 2, 1, 1) # (80 x 56)

        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1) # 28 x 40
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1) # 14 x 20
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1) # 7 x 10
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1) # 4 x 5
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1) # 2 x 3

        if config==2:
            self.layer6 = conv(outputnums[6], outputnums[6]*2, kernel_size=(2, 3), stride=1, padding=0) # 1 x 1

        if config==2:
            fcnum = outputnums[6]*2
        elif config==3:
            fcnum = outputnums[6]
        else:
            fcnum = outputnums[6]*6

        if stereo==2:
            fcnum_trans = fcnum + 10*2*6
            fcnum_rot = fcnum
        else:
            fcnum_trans = fcnum
            fcnum_rot = fcnum

        fc1_trans = linear(fcnum_trans, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum_rot, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _encode_pose(self, x, L):
        c = (torch.pow(2, torch.arange(L)) * torch.pi).to(x.device)
        y = c.view(1, -1, 1) * x.unsqueeze(1)
        z = torch.cat([torch.sin(y), torch.cos(y)], dim=1).view(x.shape[0], -1)
        return z

    def forward(self, x, scale_disp=1.0, extrinsic=None):
        # import ipdb;ipdb.set_trace()
        if self.stereo==1:
            if self.autoDistTarget > 0:
                distTarget = 1.0/(self.autoDistTarget * 0.25) # normalize the target by 0.25 -- hard code
                depth_mean = torch.mean(x[:,2,:,:], (1,2))
                scale_disp = distTarget / depth_mean
                x[:,2,:,:] = x[:,2,:,:] * scale_disp.view(scale_disp.shape+(1,1)) # tensor: (n, 1, 1)
            else:
                x[:,2,:,:] = x[:,2,:,:] * scale_disp

        x = self.firstconv(x)
        if not self.down_scale:
            x  = self.layer0(x)
            x  = self.layer0_2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.config==2:
            x = self.layer6(x)
        if self.config==3:
            x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        
        x = x.view(x.shape[0], -1)
        if self.stereo==2:
            ep = self._encode_pose(extrinsic, L=10)
            x_ep = torch.cat([x, ep], dim=1)
            x_trans = self.voflow_trans(x_ep)
            x_rot = self.voflow_rot(x)
        else:
            x_trans = self.voflow_trans(x)
            x_rot = self.voflow_rot(x)

        if self.stereo==1:
            if self.autoDistTarget > 0:
                x_trans = x_trans * scale_disp.view(scale_disp.shape+(1,))
            else:
                x_trans = x_trans * scale_disp

        if self.out_feature:
            return torch.cat((x_trans, x_rot), dim=1), x

        return torch.cat((x_trans, x_rot), dim=1)
