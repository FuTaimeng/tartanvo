
import torch 
import torch.nn as nn
import torch.nn.functional as F


class VONet(nn.Module):
    def __init__(self, network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True, uncertainty=False):
        super(VONet, self).__init__()

        if network==0: # PWCNet
            from .PWC import PWCDCNet as FlowNet
            self.flowNet     = FlowNet(uncertainty=uncertainty)
        elif network==2:
            from .FlowNet2 import FlowNet2 as FlowNet
            self.flowNet     = FlowNet(middleblock=3)
        elif network==3:
            from .StereoFlowNet import FlowNet
            self.flowNet     = FlowNet(uncertainty=uncertainty)
        else:
            print('Flow network should be 0 or 2..')

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        unc = 1 if uncertainty else 0
        self.flowPoseNet = FlowPoseNet(intrinsic=intrinsic, down_scale=down_scale, config=config, uncertainty=unc)

        self.network = network
        self.intrinsic = intrinsic
        self.flowNormFactor = flowNormFactor
        self.down_scale = down_scale
        self.uncertainty = uncertainty

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, x, only_flow=False, only_pose=False, gt_flow=False):
        '''
        x[0]: rgb frame t-1 and t
        x[1]: intrinsics
        x[2]: flow t-1 -> t (optional)
        '''
        # import ipdb;ipdb.set_trace()
        if not only_pose: # forward flownet
            flow_out, unc_out = self.flowNet(x[0])
            if only_flow:
                return flow_out, unc_out

            if self.network == 0:
                if self.down_scale:
                    flow = flow_out[0]
                    if self.uncertainty:
                        unc = unc_out[0]
                else:
                    flow = F.interpolate(flow_out[0], scale_factor=4, mode='bilinear', align_corners=True)
                    if self.uncertainty:
                        unc = F.interpolate(unc_out[0], scale_factor=4, mode='bilinear', align_corners=True)
            elif self.network ==2 or self.network==3:
                if self.down_scale:
                    flow_out = F.interpolate(flow_out, scale_factor=0.25, mode='bilinear', align_corners=True)
                    if self.uncertainty:
                        unc = F.interpolate(unc_out, scale_factor=0.25, mode='bilinear', align_corners=True)
                        unc = 0.5 * torch.tanh(-unc*0.5-2)+0.5 # mapping unc to 0-1
                flow = flow_out
        else:
            assert(gt_flow) # when only_pose==True, we should provide gt-flow as input
            assert(len(x)>2)
            flow_out = None

        if gt_flow:
            flow_input = x[2]
        else:
            flow_input = flow * self.flowNormFactor

        if self.uncertainty:
            flow_input = torch.cat((flow_input, unc), dim=1)

        if self.intrinsic:
            flow_input = torch.cat( ( flow_input, x[1] ), dim=1 )
        
        pose = self.flowPoseNet( flow_input )

        return flow_out, pose


class MultiCamVONet(nn.Module):
    def __init__(self, flowNormFactor=1.0, fixflow=True):
        super(MultiCamVONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, uncertainty=0, stereo=2)

        self.flowNormFactor = flowNormFactor

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, imgA, imgB, imgC, intrinsic, extrinsic):
        # import ipdb;ipdb.set_trace()
        flowAB, _ = self.flowNet(torch.cat([imgA, imgB], dim=1))
        flowAC, _ = self.flowNet(torch.cat([imgA, imgC], dim=1))
                
        flowAB = flowAB[0] * self.flowNormFactor
        flowAC = flowAC[0] * self.flowNormFactor

        x = torch.cat([flowAB, flowAC, intrinsic], dim=1)
        pose = self.flowPoseNet(x, extrinsic=extrinsic)

        return flowAB, flowAC, pose


    # def get_flow_loss(self, netoutput, target, criterion, mask=None, small_scale=False):
    #     '''
    #     small_scale: the target flow and mask are down scaled (when in forward_vo)
    #     '''
    #     if self.network == 0: # pwc net
    #         # netoutput 1/4, 1/8, ..., 1/32 size flow
    #         # if mask is not None:
    #         return self.flowNet.calc_loss(netoutput, target, criterion, mask) # To be tested
    #         # else:
    #         #     return self.flowNet.get_loss(netoutput, target, criterion, small_scale=small_scale)
    #     else: 
    #         if mask is not None:
    #             # if small_scale:
    #             #     mask = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=True)
    #             valid_mask = mask<128
    #             valid_mask = valid_mask.expand(target.shape)
    #             return criterion(netoutput[valid_mask], target[valid_mask])
    #         else:
    #             return criterion(netoutput, target)

    def get_flow_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        '''
        Note: criterion is not used when uncertainty is included
        '''
        if mask is not None: 
            output_ = output[mask]
            target_ = target[mask]
            if unc is not None:
                unc = unc[mask]
        else:
            output_ = output
            target_ = target

        if unc is None:
            return criterion(output_, target_), criterion(output_, target_)
        else: # if using uncertainty, then no mask 
            diff = torch.abs( output_ - target_) # hard code L1 loss
            loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
            loss = torch.mean(diff)
            return  loss_unc/(1.0+lamb), loss


if __name__ == '__main__':
    
    voflownet = VONet(network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True) # 
    voflownet.cuda()
    voflownet.eval()
    print (voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    x, y = np.ogrid[:448, :640]
    # print (x, y, (x+y))
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print (img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    intrin = imgInput[:,:2,:112,:160].copy()

    imgTensor = torch.from_numpy(imgInput)
    intrinTensor = torch.from_numpy(intrin)
    print (imgTensor.shape)
    stime = time.time()
    for k in range(100):
        flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
        print (flow.data.shape, pose.data.shape)
        print (pose.data.cpu().numpy())
        print (time.time()-stime)
    print (time.time()-stime)/100
