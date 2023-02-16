
import torch 
import torch.nn as nn
import torch.nn.functional as F


class VONet(nn.Module):
    def __init__(self, flowNormFactor=1.0, fixflow=True):
        super(VONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, stereo=0)

        self.flowNormFactor = flowNormFactor

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, imgA, imgC, intrinsic):
        # import ipdb;ipdb.set_trace()
        flowAC, _ = self.flowNet(torch.cat([imgA, imgC], dim=1))
        flowAC = flowAC[0] * self.flowNormFactor

        x = torch.cat([flowAC, intrinsic], dim=1)
        pose = self.flowPoseNet(x)

        return flowAC, pose


class MultiCamVONet(nn.Module):
    def __init__(self, flowNormFactor=1.0, fix_parts=("flow"), stereo=2, sep_feat=True):
        super(MultiCamVONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, stereo=stereo, fix_parts=fix_parts, sep_feat=sep_feat)
        # self.flowPoseNet = FlowPoseNet(inputnum=4)

        self.flowNormFactor = flowNormFactor

        if "flow" in fix_parts:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, imgA, imgB, imgC, intrinsic, extrinsic):
        # import ipdb;ipdb.set_trace()
        flowAB, _ = self.flowNet(torch.cat([imgA, imgB], dim=1))
        flowAC, _ = self.flowNet(torch.cat([imgA, imgC], dim=1))
                
        flowAB = flowAB[0] * self.flowNormFactor
        flowAC = flowAC[0] * self.flowNormFactor

        x = torch.cat([flowAB, flowAC, intrinsic], dim=1)
        # x = torch.cat([flowAC, intrinsic], dim=1)
        pose = self.flowPoseNet(x, extrinsic=extrinsic)
        # pose = self.flowPoseNet(x)

        return flowAB, flowAC, pose

    # def get_flow_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
    #     '''
    #     Note: criterion is not used when uncertainty is included
    #     '''
    #     if mask is not None: 
    #         output_ = output[mask]
    #         target_ = target[mask]
    #         if unc is not None:
    #             unc = unc[mask]
    #     else:
    #         output_ = output
    #         target_ = target

    #     if unc is None:
    #         return criterion(output_, target_), criterion(output_, target_)
    #     else: # if using uncertainty, then no mask 
    #         diff = torch.abs( output_ - target_) # hard code L1 loss
    #         loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
    #         loss = torch.mean(diff)
    #         return  loss_unc/(1.0+lamb), loss


# if __name__ == '__main__':
#     voflownet = VONet(network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True) # 
#     voflownet.cuda()
#     voflownet.eval()
#     print (voflownet)
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import time

#     x, y = np.ogrid[:448, :640]
#     # print (x, y, (x+y))
#     img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
#     img = img.astype(np.float32)
#     print (img.dtype)
#     imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
#     intrin = imgInput[:,:2,:112,:160].copy()

#     imgTensor = torch.from_numpy(imgInput)
#     intrinTensor = torch.from_numpy(intrin)
#     print (imgTensor.shape)
#     stime = time.time()
#     for k in range(100):
#         flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
#         print (flow.data.shape, pose.data.shape)
#         print (pose.data.cpu().numpy())
#         print (time.time()-stime)
#     print (time.time()-stime)/100
