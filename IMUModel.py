import torch
from torch import nn

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau


def skew_symmetric(x):
    return torch.tensor([   
        0,     -x[2],  x[1],
         x[2],     0, -x[0],
        -x[1],  x[0],     0
    ], dtype=x.dtype, device=x.device).view(3, 3)

def F_t(R_t, a_hat_t, bias_a_t, w_hat_t, bias_w_t):
    F = torch.zeros((15, 15), dtype=R_t.dtype, device=R_t.device)
    F[0:3, 3:6] = torch.eye(3)
    F[3:6, 6:9] = -R_t @ skew_symmetric(a_hat_t - bias_a_t)
    F[3:6, 9:12] = -R_t
    F[6:9, 6:9] = -skew_symmetric(w_hat_t - bias_w_t)
    F[6:9, 12:15] = -torch.eye(3)
    return F


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    else:
        return torch.tensor(x, dtype=torch.float32)


class IMUModel(nn.Module):
    def __init__(self, gyro_measurments, accel_measurments, deltatimes, keyframe_idx=None, 
                    init_gyro_bias=None, init_accel_bias=None, gravity=(0,0,9.81007), 
                    init_rot=None, init_pos=None, init_vel=None,
                    device='cuda', para_list=('gyro_bias', 'accel_bias', 'gravity')):

        super().__init__()

        gyro_measurments = to_tensor(gyro_measurments)
        accel_measurments = to_tensor(accel_measurments)
        assert len(gyro_measurments.shape) == 2 and len(accel_measurments.shape) == 2
        assert gyro_measurments.shape[0] == accel_measurments.shape[0]
        assert gyro_measurments.shape[1] == 3 and accel_measurments.shape[1] == 3
        N = gyro_measurments.shape[0]

        if isinstance(deltatimes, float):
            deltatimes = torch.tensor([deltatimes] * (N-1))
        else:
            deltatimes = to_tensor(deltatimes)
            # if deltatimes.shape == (N-1,):
            #     deltatimes = torch.cat((deltatimes, torch.tensor([deltatimes[-1]])))
            assert deltatimes.shape == (N-1,)

        if keyframe_idx is None:
            keyframe_idx = [i for i in range(N)]
        else:
            assert max(keyframe_idx) < N
        K = len(keyframe_idx)

        if init_gyro_bias is None:
            init_gyro_bias = torch.zeros(3)
        else:
            init_gyro_bias = to_tensor(init_gyro_bias)
            assert init_gyro_bias.shape == (3,)

        if init_accel_bias is None:
            init_accel_bias = torch.zeros(3)
        else:
            init_accel_bias = to_tensor(init_gyro_bias)
            assert init_accel_bias.shape == (3,)

        gravity = to_tensor(gravity)
        assert gravity.shape == (3,)

        if init_rot is not None:
            if isinstance(init_rot, pp.LieTensor):
                assert init_rot.ltype == pp.SO3Type
                init_rot = init_rot.to(dtype=torch.float32)
            else:
                init_rot = to_tensor(init_rot)
                assert init_rot.shape == (4,)
                init_rot = pp.SO3(init_rot)
        
        if init_pos is not None:
            init_pos = to_tensor(init_pos)
            assert init_pos.shape == (3,)

        if init_vel is not None:
            init_vel = to_tensor(init_vel)
            assert init_vel.shape == (3,)

        keyframe_deltatimes = torch.tensor([
            torch.sum(deltatimes[keyframe_idx[i]:keyframe_idx[i+1]]) for i in range(len(keyframe_idx)-1)
        ])

        self.a_hat = accel_measurments.to(device)
        self.w_hat = gyro_measurments.to(device)
        self.dt = deltatimes.to(device)
        self.kf = keyframe_idx
        self.kf_dt = keyframe_deltatimes.to(device)
        
        self.bias_a = init_accel_bias.repeat(N, 1).to(device)   
        self.bias_w = init_gyro_bias.repeat(N, 1).to(device)
        self.g = gravity.to(device)
        self.init_rot = init_rot.to(device)
        self.init_pos = init_pos.to(device)
        self.init_vel = init_vel.to(device)
        self.device = device

        self.num_measurements = N
        self.num_keyframes = K

        for para in para_list:
            if para == 'accel_bias':
                self.bias_a = nn.Parameter(self.bias_a)
            elif para == 'gyro_bias':
                self.bias_w = nn.Parameter(self.bias_w)
            elif para == 'gravity':
                self.g = nn.Parameter(self.g)
            else:
                supported_para_types = ('gyro_bias', 'accel_bias', 'gravity')
                raise ValueError('Unknown para type in IMUModel: {}. Supported types: {}'.format(para, supported_para_types))
        
        self.__preintegrate()

    def __preintegrate(self):
        self.preintegrator = pp.module.IMUPreintegrator(gravity=0., prop_cov=False, reset=True)
        self.preintegrator = self.preintegrator.to(self.device)

        M = self.num_keyframes - 1
        alpha = torch.zeros(M, 3).to(self.device)
        beta = torch.zeros(M, 3).to(self.device)
        gamma = pp.identity_SO3(M).to(self.device)
        Jac = torch.zeros(M, 15, 15).to(self.device)

        for i in range(M):
            a, b, c, J = self.__propergate(self.kf[i], self.kf[i+1])
            alpha[i] = a
            beta[i] = b
            gamma[i] = c
            Jac[i] = J

        self.alpha_hat = alpha
        self.beta_hat = beta
        self.gamma_hat = gamma
        self.Jac = Jac

    def __propergate(self, start_idx, end_idx):
        dt = self.dt[start_idx:end_idx]
        acc = self.a_hat[start_idx:end_idx]
        gyro = self.w_hat[start_idx:end_idx]
        b_a = self.bias_a[start_idx:end_idx]
        b_w = self.bias_w[start_idx:end_idx]

        state = self.preintegrator(dt=dt[:, None], acc=acc, gyro=gyro)

        J = torch.eye(15).to(self.device)
        for i in range(end_idx - start_idx):
            R = state['rot'][..., i, :].squeeze().matrix()
            F = F_t(R, acc[i], b_a[i], gyro[i], b_w[i])
            J = (torch.eye(15).to(self.device) + dt[i] * F) @ J
        
        alpha = state['pos'][..., -1, :].squeeze()
        beta = state['vel'][..., -1, :].squeeze()
        gamma = state['rot'][..., -1, :].squeeze()

        return alpha, beta, gamma, J

    def init_pose(self):
        assert self.init_pos is not None and self.init_rot is not None
        return pp.SE3(torch.cat((self.init_pos, self.init_rot.tensor())))

    def alpha(self, k0=None, k1=None):
        if k0 == None and k1 == None:
            k0 = 0
            k1 = self.Jac.shape[0]
        elif k1 == None:
            k1 = k0 + 1
        J_alpha_ba = self.Jac[k0:k1, 0:3, 9:12]
        J_alpha_bw = self.Jac[k0:k1, 0:3, 12:15]
        b_a = self.bias_a[self.kf[k0:k1], :, None]
        b_w = self.bias_w[self.kf[k0:k1], :, None]
        res = self.alpha_hat[k0:k1, :, None] \
              + J_alpha_ba @ b_a \
              + J_alpha_bw @ b_w
        return res.squeeze()
    
    def beta(self, k0=None, k1=None):
        if k0 == None and k1 == None:
            k0 = 0
            k1 = self.Jac.shape[0]
        elif k1 == None:
            k1 = k0 + 1
        J_beta_ba = self.Jac[k0:k1, 3:6, 9:12]
        J_beta_bw = self.Jac[k0:k1, 3:6, 12:15]
        b_a = self.bias_a[self.kf[k0:k1], :, None]
        b_w = self.bias_w[self.kf[k0:k1], :, None]
        res = self.beta_hat[k0:k1, :, None] \
              + J_beta_ba @ b_a \
              + J_beta_bw @ b_w
        return res.squeeze()

    def gamma(self, k0=None, k1=None):
        if k0 == None and k1 == None:
            k0 = 0
            k1 = self.Jac.shape[0]
        elif k1 == None:
            k1 = k0 + 1
        J_gamma_bw = self.Jac[k0:k1, 6:9, 12:15]
        b_w = self.bias_w[self.kf[k0:k1], :, None]
        dtheta = 0.5 * J_gamma_bw @ b_w
        dtheta = dtheta.squeeze()
        dquat = pp.SO3(torch.cat((dtheta, torch.ones(k1-k0, 1).to(self.device)), dim=1))
        res = self.gamma_hat[k0:k1] @ dquat
        return res.squeeze()

    def world_dpos(self, R_wb, vel_w):
        M = self.num_keyframes - 1
        return vel_w[:M] * self.kf_dt[:, None] \
               - 0.5 * self.g[None, :] * self.kf_dt[:, None]**2 \
               + R_wb[:M].Act(self.alpha())
    
    def world_dvel(self, R_wb):
        M = self.num_keyframes - 1
        return -self.g[None, :] * self.kf_dt[:, None] \
               + R_wb[:M].Act(self.beta())

    def world_drot(self):
        return self.gamma()

    def trajectory(self, return_delta=False):
        assert self.init_rot is not None and \
               self.init_vel is not None and \
               self.init_pos is not None

        drot = self.world_drot()
        rot = [self.init_rot]
        for dr in drot:
            rot.append(rot[-1] @ dr)
        rot = torch.stack(rot)

        dvel = self.world_dvel(rot)
        vel = torch.cumsum(torch.cat((self.init_vel[None, :], dvel), dim=0), dim=0)

        dpos = self.world_dpos(rot, vel)
        pos = torch.cumsum(torch.cat((self.init_pos[None, :], dpos), dim=0), dim=0)

        if not return_delta:
            return rot, pos, vel
        else:
            return drot, dpos, dvel

    def forward(self, gt_rot=None, gt_pos=None, gt_vel=None, weights=(1,1,1)):
        rot, pos, vel = self.trajectory()

        loss = []
        if gt_rot is not None:
            loss.extend(weights[0] * (gt_rot.Inv() @ rot).Log())
        if gt_pos is not None:
            loss.extend(weights[1] * (gt_pos - pos) ** 2)
        if gt_vel is not None:
            loss.extend(weights[2] * (gt_vel - vel) ** 2)
        loss = torch.stack(loss)

        return loss


def imu_model_optimization(imu_model, radius=1e4, gt_rot=None, gt_pos=None, gt_vel=None, weights=(1,1,1)):
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(imu_model, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    if gt_rot is not None:
        gt_rot = gt_rot.to(imu_model.device)
    if gt_pos is not None:
        gt_pos = gt_pos.to(imu_model.device)
    if gt_vel is not None:
        gt_vel = gt_vel.to(imu_model.device)

    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual:
        loss = optimizer.step(input=(gt_rot, gt_pos, gt_vel, weights))
        scheduler.step(loss)

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize()


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from Datasets.TrajFolderDataset import TrajFolderDatasetPVGO

    torch.autograd.set_detect_anomaly(True)

    dataroot = '/user/taimengf/projects/kitti_raw/2011_09_30/2011_09_30_drive_0034_sync'
    ds = TrajFolderDatasetPVGO(datadir=dataroot, datatype='kitti', start_frame=0, end_frame=80)
    print('Load data done.')

    t0 = time.time()
    imu_model = IMUModel(gyro_measurments=ds.gyros, accel_measurments=ds.accels, 
        deltatimes=ds.imu_dts, keyframe_idx=ds.rgb2imu_sync, device='cuda:0',
        init_rot=ds.imu_init['rot'], init_pos=ds.imu_init['pos'], init_vel=ds.imu_init['vel'])
    t1 = time.time()
    print('Init IMU model done. Time:', t1 - t0)

    rot, pos, vel = imu_model.trajectory()

    gt_pos = torch.tensor(ds.poses[:, :3])
    gt_rot = pp.SO3(ds.poses[:, 3:])

    t2 = time.time()
    imu_model_optimization(imu_model, gt_rot=gt_rot, gt_pos=gt_pos, weights=(1,1,0))
    t3 = time.time()
    print('IMU optimization done. Time:', t3 - t2)

    print('accel bias ({})'.format(imu_model.bias_a.shape), imu_model.bias_a)
    print('gyro bias ({})'.format(imu_model.bias_w.shape), imu_model.bias_w)
    print('gravity', imu_model.g)

    rot2, pos2, vel2 = imu_model.trajectory()

    rot_errs = torch.norm((gt_rot.Inv() @ rot).Log(), dim=1) * 180 / 3.14
    pos_errs = torch.norm(gt_pos - pos, dim=1)
    rot2_errs = torch.norm((gt_rot.Inv() @ rot2).Log(), dim=1) * 180 / 3.14
    pos2_errs = torch.norm(gt_pos - pos2, dim=1)
    print('Rot Errs:', torch.mean(rot_errs), torch.mean(rot2_errs))
    print('Pos Errs:', torch.mean(pos_errs), torch.mean(pos2_errs))

    pos = pos.detach().numpy()
    pos2 = pos2.detach().numpy()
    gt_pos = gt_pos.numpy()

    plt.figure('XY')
    plt.plot(pos[:, 0], pos[:, 1], color='r')
    plt.plot(pos2[:, 0], pos2[:, 1], color='b')
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], color='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('imu_model_test_traj_XY.png')

    plt.figure('XZ')
    plt.plot(pos[:, 0], pos[:, 2], color='r')
    plt.plot(pos2[:, 0], pos2[:, 2], color='b')
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], color='g')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig('imu_model_test_traj_XZ.png')

    plt.figure('YZ')
    plt.plot(pos[:, 1], pos[:, 2], color='r')
    plt.plot(pos2[:, 0], pos2[:, 2], color='b')
    plt.plot(gt_pos[:, 1], gt_pos[:, 2], color='g')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.savefig('imu_model_test_traj_YZ.png')
