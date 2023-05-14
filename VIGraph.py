import torch
from torch import nn

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

from IMUModel import IMUModel


def to_SE3(x):
    if isinstance(x, pp.LieTensor):
        if x.ltype == pp.SE3_type:
            return x.to(dtype=torch.float32)
        elif x.ltype == pp.se3_type:
            return x.Exp().to(dtype=torch.float32)
        else:
            raise RuntimeError('Cannot convert to SE3: {}'.format(x))
    else:
        return pp.SE3(x, dtype=torch.float32)


class VIGraph(nn.Module):
    def __init__(self, visual_motions, visual_links, imu_model, loss_weight):
        super().__init__()

        assert isinstance(visual_motions, pp.LieTensor) and visual_motions.ltype == pp.SE3_type
        assert isinstance(visual_links, torch.Tensor)
        assert isinstance(imu_model, IMUModel)

        assert visual_motions.shape[0] == visual_links.shape[0]
        assert torch.max(visual_links) < imu_model.num_keyframes

        self.vmot = visual_motions.detach()
        self.vmot_withgrad = visual_motions
        
        irot, _, ivel = imu_model.trajectory()
        pos = torch.cumsum(torch.cat((
            imu_model.init_pos[None, :], 
            irot[:-1].Act(self.vmot.translation())
        ), dim=0), dim=0)
        pose = pp.SE3(torch.cat((pos, irot.tensor()), dim=1))

        self.pose = pp.Parameter(pose)
        self.vel = torch.nn.Parameter(ivel)
        self.vlink = visual_links
        self.imu_model = imu_model

        assert len(loss_weight) == 4
        # loss weight hyper para
        self.l1 = loss_weight[0]
        self.l2 = loss_weight[1]
        self.l3 = loss_weight[2]
        self.l4 = loss_weight[3]

    def forward(self):
        # parameters
        pose = self.pose
        vel = self.vel
        
        # IMU knowledge
        imu_drot, imu_dpos, imu_dvel = imu_model.trajectory(return_delta=True)
        
        # pose graph constraint
        pose1 = pose[self.vlink[:, 0]]
        pose2 = pose[self.vlink[:, 1]]
        error = self.vmot.Inv() @ pose1.Inv() @ pose2
        pgerr = error.Log().tensor()

        # adj vel constraint
        adjvelerr = imu_dvel - torch.diff(vel, dim=0)

        # imu rot constraint
        pose1 = pose.rotation()[ :-1]
        pose2 = pose.rotation()[1:  ]
        error = imu_drot @ pose1.Inv() @ pose2
        imuroterr = error.Log().tensor()

        # trans vel constraint
        transvelerr = imu_dpos - torch.diff(pose.translation(), dim=0)

        # pvgo loss
        return torch.cat((  self.l1 * pgerr.view(-1), 
                            self.l2 * adjvelerr.view(-1), 
                            self.l3 * imuroterr.view(-1), 
                            self.l4 * transvelerr.view(-1)  ), dim=0)

    def vo_loss(self):
        pose = self.pose
        vel = self.vel

        pose1 = pose[self.vlink[:, 0]].detach()
        pose2 = pose[self.vlink[:, 1]].detach()
        error = self.vmot_withgrad.Inv() @ pose1.Inv() @ pose2
        error = error.Log().tensor()

        trans_loss = torch.sum(error[:, :3]**2, dim=1)
        rot_loss = torch.sum(error[:, 3:]**2, dim=1)

        return trans_loss, rot_loss

    def align_to_imu_init(self):
        source = self.pose[0].detach()
        target = self.imu_model.init_pose()
        aligned_vel = target.rotation() @ source.rotation().Inv() @ self.vel
        aligned_pose = target @ source.Inv() @ self.pose
        return aligned_pose, aligned_vel

    def motions(self):
        return self.pose[self.vlink[:, 0]].Inv() @ self.pose[self.vlink[:, 1]]


def graph_optimization(graph, radius=1e4):
    assert isinstance(graph, VIGraph)

    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual:
        loss = optimizer.step(input=())
        scheduler.step(loss)

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize()

    trans_loss, rot_loss = graph.vo_loss()

    pose, vel = graph.align_to_imu_init()
    motion = graph.motions()
    pose = pose.detach().cpu().numpy()
    vel = vel.detach().cpu().numpy()
    motion = motion.detach().cpu().numpy()

    return trans_loss, rot_loss, pose, vel, motion


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from Datasets.TrajFolderDataset import TrajFolderDatasetPVGO

    dataroot = '/user/taimengf/projects/kitti_raw/2011_09_30/2011_09_30_drive_0034_sync'
    ds = TrajFolderDatasetPVGO(datadir=dataroot, datatype='kitti', start_frame=0, end_frame=8)
    print('Load data done.')

    t0 = time.time()
    imu_model = IMUModel(gyro_measurments=ds.gyros, accel_measurments=ds.accels, 
        deltatimes=ds.imu_dts, keyframe_idx=ds.rgb2imu_sync, device='cpu',
        init_rot=ds.imu_init['rot'], init_pos=ds.imu_init['pos'], init_vel=ds.imu_init['vel'])
    t1 = time.time()
    print('Init IMU model done. Time:', t1 - t0)

    imu_rot, imu_pos, imu_vel = imu_model.trajectory()

    gt_poses = pp.SE3(ds.poses)
    links = torch.tensor([[i, i+1] for i in range(gt_poses.shape[0]-1)])
    gt_motions = gt_poses[links[:, 0]].Inv() @ gt_poses[links[:, 1]]

    graph = VIGraph(visual_motions=gt_motions, visual_links=links, imu_model=imu_model, 
                    loss_weight=(1,1,10,1))

    trans_loss, rot_loss, pgo_pose, pgo_vel, pgo_motion = graph_optimization(graph)

    imu_pos = imu_pos.detach().numpy()
    pgo_pos = pgo_pose[:, :3]
    gt_pos = ds.poses[:, :3]

    plt.figure('XY')
    plt.plot(imu_pos[:, 0], imu_pos[:, 1], color='r')
    plt.plot(pgo_pos[:, 0], pgo_pos[:, 1], color='b')
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], color='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('graph_optm_test_traj_XY.png')
