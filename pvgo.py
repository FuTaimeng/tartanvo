import os
import torch
import warnings
import argparse
import numpy as np
import pypose as pp
from torch import nn
from pgo_dataset import G2OPGO, VOPGO, PVGO_Dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseVelGraph(nn.Module):
    def __init__(self, nodes, vels, device, loss_weight, stop_frames=[]):
        super().__init__()
        assert nodes.size(0) == vels.size(0)
        # self.nodes = pp.Parameter(nodes)
        # self.vels = torch.nn.Parameter(vels)
        self.para_nodes = pp.Parameter(nodes[1:])
        self.para_vels = torch.nn.Parameter(vels[1:])
        self.node0 = nodes[0].cpu()
        self.vel0 = vels[0].cpu()
        # self.nodes = torch.cat([self.node0.view(1, -1), self.para_nodes], dim=0)
        # self.vels = torch.cat([self.vel0.view(1, -1), self.para_vels], dim=0)
        self.device = device

        assert len(loss_weight) == 4
        # loss weight hyper para
        self.l1 = loss_weight[0]
        self.l2 = loss_weight[1]
        self.l3 = loss_weight[2]
        self.l4 = loss_weight[3]

        self.stop_frames = stop_frames


    def nodes(self):
        return torch.cat([self.node0.to(self.device).view(1, -1), self.para_nodes], dim=0)
    
    def vels(self):
        return torch.cat([self.vel0.to(self.device).view(1, -1), self.para_vels], dim=0)


    def forward(self, edges, poses, imu_drots, imu_dtrans, imu_dvels, dts):
        nodes = self.nodes()
        vels = self.vels()
        
        # E = edges.size(0)
        # M = nodes.size(0) - 1
        # assert E == poses.size(0)
        # assert M == imu_drots.size(0) == imu_dtrans.size(0) == imu_dvels.size(0)
        
        # pose graph constraint
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        pgerr = error.Log().tensor()

        # adj vel constraint
        adjvelerr = imu_dvels - torch.diff(vels, dim=0)

        # imu rot constraint
        node1 = nodes.rotation()[ :-1]
        node2 = nodes.rotation()[1:  ]
        error = imu_drots.Inv() @ node1.Inv() @ node2
        imuroterr = error.Log().tensor()

        # trans vel constraint
        transvelerr = torch.diff(nodes.translation(), dim=0) - (vels[:-1] * dts.view(-1, 1) + imu_dtrans)

        # stop constraint
        stopvelerr = vels[self.stop_frames]

        # print("pvgo errs:")
        # print('pgerr:       ', pgerr.size(), pgerr[5].detach().cpu().numpy())
        # print('adjvelerr:   ', adjvelerr.size(), adjvelerr[5].detach().cpu().numpy())
        # print('imuroterr:   ', imuroterr.size(), imuroterr[5].detach().cpu().numpy())
        # print('transvelerr: ', transvelerr.size(), transvelerr[5].detach().cpu().numpy())

        # test_run
        return torch.cat((  self.l1 * pgerr.view(-1), 
                            self.l2 * adjvelerr.view(-1), 
                            self.l3 * imuroterr.view(-1),
                            self.l4 * transvelerr.view(-1),
                            self.l2*100 * stopvelerr.view(-1)  ), dim=0)

        # # test_run_pg
        # return pgerr.view(-1, 1)

        # # test_run_pg-imurot
        # return torch.cat((pgerr.view(-1, 1), imuroterr.view(-1, 1)), dim=0)


    def vo_loss(self, edges, poses):
        nodes = self.nodes()
        vels = self.vels()

        node1 = nodes[edges[..., 0]].detach()
        node2 = nodes[edges[..., 1]].detach()
        motion = node1.Inv() @ node2
        error = poses.Inv() @ motion 
        # trans_loss = torch.norm(error.translation(), dim=1, p=1) / torch.norm(motion.translation(), dim=1, p=1)
        # rot_loss = torch.norm(error.rotation(), dim=1, p=1)
        # loss = rot_loss + trans_loss
        loss = torch.norm(error, dim=1, p=1)
        return loss


def run_pvgo(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dt_, 
                device='cuda:0', radius=1e4, loss_weight=(1,1,1,1), stop_frames=[]):

    data = PVGO_Dataset(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dt_, device)
    nodes, vels = data.nodes, data.vels
    edges, poses = data.edges, data.poses
    imu_drots, imu_dtrans, imu_dvels = data.imu_drots, data.imu_dtrans, data.imu_dvels
    dt, node0, vel0 = data.dt, data.node0, data.vel0

    graph = PoseVelGraph(nodes, vels, device, loss_weight, stop_frames).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual:
        # TODO: weights
        loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dt))
        scheduler.step(loss)
        # graph.nodes[0].data = node0
        # graph.vels[0].data = vel0

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize(input=(edges, poses), weight=infos)

    # fix_nodes = initial_node0 @ graph.nodes[0].detach().Inv() @ graph.nodes.detach()
    # fix_nodes = fix_nodes.cpu().numpy()
    # np.savetxt(os.path.join(savedir, 'pgo_pose.txt'), fix_nodes)
    nodes_np = graph.nodes().detach().cpu().numpy()
    vels_np = graph.vels().detach().cpu().numpy()

    loss = graph.vo_loss(edges, data.poses_withgrad)
    return loss, nodes_np, vels_np
