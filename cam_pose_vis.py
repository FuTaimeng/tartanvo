from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def normalize_3d(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    lim_len = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    xlim = (xlim[0]+xlim[1])/2 - lim_len/2, (xlim[0]+xlim[1])/2 + lim_len/2
    ylim = (ylim[0]+ylim[1])/2 - lim_len/2, (ylim[0]+ylim[1])/2 + lim_len/2
    zlim = (zlim[0]+zlim[1])/2 - lim_len/2, (zlim[0]+zlim[1])/2 + lim_len/2
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


# def read_pose(fname):
#     Rs = []
#     qs = []
#     ts = []
#     with open(fname) as f:
#         for line in f:
#             line = line.rstrip()
#             if not line:
#                 continue
#             chunks = line.split(' ')
#             q = [float(chunks[k]) for k in (1, 2, 3, 0)]
#             t = [float(chunks[k]) for k in range(4, 7)]
#             qs.append(q)
#             ts.append(t)
#     Rs = Rotation.from_quat(qs)
#     ts = np.array(ts)
#     return Rs, ts


def read_pose_tartanvo(fname, correct_rot=False):
    Rs = []
    qs = []
    ts = []
    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            chunks = line.split(' ')
            q = [float(chunks[k]) for k in range(3, 7)]
            t = [float(chunks[k]) for k in range(0, 3)]
            qs.append(q)
            ts.append(t)
    Rs = Rotation.from_quat(qs)
    ts = np.array(ts)

    if correct_rot:
        tf = Rotation.from_matrix(np.array([0,0,1, 0,-1,0, 1,0,0]).reshape(3, 3))
        Rs = Rs * tf
        # new_Rs = []
        # for R in Rs:
        #     new_R = np.dot(R.as_matrix(), tf)
        #     new_Rs.append(new_R)
        # Rs = Rotation.from_matrix(new_Rs)
    return Rs, ts


def read_pose_rotvec(fname, correct_rot=False):
    Rs = []
    rs = []
    ts = []
    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            chunks = line.split(' ')
            r = [float(chunks[k]) for k in range(3, 6)]
            t = [float(chunks[k]) for k in range(0, 3)]
            rs.append(r)
            ts.append(t)
    Rs = Rotation.from_rotvec(rs)
    ts = np.array(ts)

    if correct_rot:
        tf = Rotation.from_matrix(np.array([0,0,1, 0,-1,0, 1,0,0]).reshape(3, 3))
        Rs = Rs * tf
        # new_Rs = []
        # for R in Rs:
        #     new_R = np.dot(R.as_matrix(), tf)
        #     new_Rs.append(new_R)
        # Rs = Rotation.from_matrix(new_Rs)
    return Rs, ts


# def read_link(fname, sample_step):
#     links = []
#     with open(fname) as f:
#         for line in f:
#             line = line.rstrip()
#             if not line:
#                 continue
#             chunks = line.split(' ')
#             id1 = int(chunks[0][-len('000000.color.png'):-len('.color.png')]) // sample_step
#             id2 = int(chunks[1][-len('000000.color.png'):-len('.color.png')]) // sample_step
#             print("graph link:", (id1, id2))
#             links.append([id1, id2])
#     return links


def read_link_tartanvo(fname):
    return np.loadtxt(fname, dtype=np.int)


def transform_pose(R_tar, t_tar, idx, Rs, ts):
    new_Rs = []
    new_ts = []
    for i in range(len(ts)):
        new_R = R_tar * Rs[idx].inv() * Rs[i]
        new_t = (R_tar * Rs[idx].inv()).apply(ts[i]) - (R_tar * Rs[idx].inv()).apply(ts[idx]) + t_tar
        new_Rs.append(new_R.as_quat())
        new_ts.append(new_t)
    new_Rs = Rotation.from_quat(new_Rs)
    new_ts = np.array(new_ts)
    return new_Rs, new_ts


def draw_pose(ax, Rs, ts, color, line=True, point=True, dir_line_scale=0.1, dir_line_step=1, vels=None, vel_scale=0.1, vel_col='yellow'):
    N = len(ts)
    if line:
        ax.plot3D(ts[:, 0], ts[:, 1], ts[:, 2], color=color)
    if point:
        ax.scatter3D(ts[:, 0], ts[:, 1], ts[:, 2], color=color, s=5)

    if dir_line_scale > 0:
        for i in range(0, len(Rs), dir_line_step):
            R = Rs[i]
            v = R.as_matrix() * dir_line_scale
            line_color = ['r', 'g', 'b']
            for k in range(3):
                l = np.concatenate((ts[i].reshape(3, 1), ts[i].reshape(3, 1) + v[:, k].reshape(3, 1)), axis=1)
                ax.plot3D(l[0, :], l[1, :], l[2, :], color=line_color[k], linewidth=0.5)

    if vels is not None:
        for i in range(len(Rs)-1):
            l = np.concatenate((ts[i].reshape(3, 1), ts[i].reshape(3, 1) + vels[i].reshape(3, 1) * vel_scale), axis=1)
            ax.plot3D(l[0, :], l[1, :], l[2, :], color=vel_col, linewidth=0.5)


def draw_edge(ax, ts, links, color):
    for l in links:
        ax.plot3D([ts[l[0], 0], ts[l[1], 0]], [ts[l[0], 1], ts[l[1], 1]], [ts[l[0], 2], ts[l[1], 2]], color=color)


def pose2motion(R1, t1, R2, t2):
    return R1.inv() * R2, R1.inv().apply(t2-t1)

def motions2poses(mRs, mts, R=None, t=None):
    if R is None or t is None:
        R = Rotation.identity()
        t = np.zeros(3)
    qs = [R.as_quat()]
    ts = [t]
    for i in range(len(mRs)):
        t = R.apply(mts[i]) + t
        R = R * mRs[i]
        qs.append(R.as_quat())
        ts.append(t)
    Rs = Rotation.from_quat(qs)
    ts = np.array(ts)
    return Rs, ts

def motion_err(Rs1, ts1, ismotion1, Rs2, ts2, ismotion2, links=None):
    if links is None:
        links = [(i, i+1) for i in range(len(Rs1)-1)]

    errs = np.zeros((len(links), 2))
    norms = np.zeros((len(links), 2))
    for i, l in enumerate(links):
        if ismotion1:
            mR1, mt1 = Rs1[i], ts1[i]
        else:
            mR1, mt1 = pose2motion(Rs1[l[0]], ts1[l[0]], Rs1[l[1]], ts1[l[1]])
        if ismotion2:
            mR2, mt2 = Rs2[i], ts2[i]
        else:
            mR2, mt2 = pose2motion(Rs2[l[0]], ts2[l[0]], Rs2[l[1]], ts2[l[1]])
        rot_err = np.rad2deg((mR1.inv() * mR2).magnitude())
        trans_err = np.linalg.norm(mt2 - mt1)
        errs[i] = [rot_err, trans_err]
        norms[i] = [np.rad2deg(mR1.magnitude()), np.linalg.norm(mt1)]
    return errs, norms


def relposenet(ax, input_dir, sample_step):
    Rs_gt, ts_gt = read_pose(input_dir + '/gt_abs_poses.txt')
    draw_pose(ax, Rs_gt, ts_gt, 'g', dir_line_scale=0)

    Rs_fwd, ts_fwd = read_pose(input_dir + '/fwd_abs_poses.txt')
    idx = 0
    Rs_fwd, ts_fwd = transform_pose(Rs_gt[idx], ts_gt[idx], idx, Rs_fwd, ts_fwd)
    draw_pose(ax, Rs_fwd, ts_fwd, 'r', dir_line_scale=0)
    ax.scatter3D(ts_fwd[idx, 0], ts_fwd[idx, 1], ts_fwd[idx, 2], color='yellow', s=10)

    Rs_pgo, ts_pgo = read_pose(input_dir + '/pgo_abs_poses.txt')
    draw_pose(ax, Rs_pgo, ts_pgo, 'b', dir_line_scale=0)

    links = read_link(input_dir + '/pg_gt.txt', sample_step)
    draw_edge(ax, ts_fwd, links[len(Rs_gt)-1:], 'yellow')


def tartanvo(ax, input_dir, ds_name):
    try:
        Rs_gt, ts_gt = read_pose_tartanvo('{}/{}_tartanvo_1914_gt.txt'.format(input_dir, ds_name), correct_rot=True)
        Rs_fwd, ts_fwd = read_pose_tartanvo('{}/{}_tartanvo_1914.txt'.format(input_dir, ds_name))
        Rs_pgo, ts_pgo = read_pose_tartanvo(input_dir + '/pgo_save/poses.txt')
        links = read_link_tartanvo('{}/{}_tartanvo_1914_link.txt'.format(input_dir, ds_name))
        mRs_est, mts_est = read_pose_tartanvo('{}/{}_tartanvo_1914_motion.txt'.format(input_dir, ds_name))
    except:
        Rs_gt, ts_gt = read_pose_tartanvo('{}/gt_pose.txt'.format(input_dir, ds_name), correct_rot=False)
        Rs_fwd, ts_fwd = read_pose_tartanvo('{}/0/pose.txt'.format(input_dir, ds_name))
        try:
            Rs_pgo, ts_pgo = read_pose_tartanvo('{}/0/pgo_pose.txt'.format(input_dir))
        except:
            Rs_pgo, ts_pgo = Rs_gt, ts_gt
        links = read_link_tartanvo('{}/link.txt'.format(input_dir, ds_name))
        mRs_est, mts_est = read_pose_tartanvo('{}/0/motion.txt'.format(input_dir, ds_name))

    min_idx = np.min(links[len(Rs_gt)-1:])
    max_idx = np.max(links[len(Rs_gt)-1:])

    Rs_gt = Rs_gt[min_idx:max_idx+1]
    ts_gt = ts_gt[min_idx:max_idx+1]
    Rs_fwd = Rs_fwd[min_idx:max_idx+1]
    ts_fwd = ts_fwd[min_idx:max_idx+1]
    if len(ts_pgo) != len(ts_gt):
        Rs_pgo = Rs_pgo[min_idx:max_idx+1]
        ts_pgo = ts_pgo[min_idx:max_idx+1]
    mask = [np.all(links[i]>=min_idx) and np.all(links[i]<=max_idx) for i in range(len(links))]
    mRs_est = mRs_est[mask]
    mts_est = mts_est[mask]
    links = links[mask] - min_idx
    
    align_idx = 0
    Rs_fwd, ts_fwd = transform_pose(Rs_gt[align_idx], ts_gt[align_idx], align_idx, Rs_fwd, ts_fwd)
    Rs_pgo, ts_pgo = transform_pose(Rs_gt[align_idx], ts_gt[align_idx], align_idx, Rs_pgo, ts_pgo)

    # test = motion_err(Rs_fwd, ts_fwd, False, mRs_est, mts_est, True, links[:len(Rs_fwd)-1])
    # print(test)

    motion_err_fwd = motion_err(Rs_gt, ts_gt, False, mRs_est, mts_est, True, links)
    motion_err_pgo = motion_err(Rs_gt, ts_gt, False, Rs_pgo, ts_pgo, False, links)
    motion_err_comp = np.concatenate((motion_err_fwd[:, 0].reshape(-1, 1), motion_err_pgo[:, 0].reshape(-1, 1), (motion_err_pgo[:, 0] - motion_err_fwd[:, 0]).reshape(-1, 1),
                                      motion_err_fwd[:, 1].reshape(-1, 1), motion_err_pgo[:, 1].reshape(-1, 1), (motion_err_pgo[:, 1] - motion_err_fwd[:, 1]).reshape(-1, 1)), axis=1)

    draw_pose(ax, Rs_gt, ts_gt, 'g', dir_line_scale=1, dir_line_step=10)
    draw_pose(ax, Rs_fwd, ts_fwd, 'r', dir_line_scale=1, dir_line_step=10)
    draw_pose(ax, Rs_pgo, ts_pgo, 'b', dir_line_scale=1, dir_line_step=10)

    ax.scatter3D(ts_gt[align_idx, 0], ts_gt[align_idx, 1], ts_gt[align_idx, 2], color='yellow', s=50)
    draw_edge(ax, ts_fwd, links[len(Rs_gt)-1:], 'yellow')
    
    np.savetxt(input_dir + '/motion_err_fwd.txt', motion_err_fwd, fmt='%.3f')
    np.savetxt(input_dir + '/motion_err_pgo.txt', motion_err_pgo, fmt='%.3f')
    np.savetxt(input_dir + '/motion_err_comp.txt', motion_err_comp, fmt='%.3f')

    def myhist(ax, x, s):
        a = (np.max(x) - np.min(x)) / s
        mn = floor(min(x)/a) * a
        mx = ceil(max(x)/a) * a
        ax.hist(x, bins=np.arange(mn, mx, a))

    def errhist(name, rot, trans):
        plt.figure(name)
        ax_rot = plt.subplot(1,2,1)
        myhist(ax_rot, rot, 20)
        ax_trans = plt.subplot(1,2,2)
        myhist(ax_trans, trans, 20)
    
    errhist('err-adj', motion_err_comp[:len(ts_pgo)-1, 0], motion_err_comp[:len(ts_pgo)-1, 3])
    errhist('err-loop', motion_err_comp[len(ts_pgo)-1:, 0], motion_err_comp[len(ts_pgo)-1:, 3])
    errhist('errdiff-adj', motion_err_comp[:len(ts_pgo)-1, 2], motion_err_comp[:len(ts_pgo)-1, 5])
    errhist('errdiff-loop', motion_err_comp[len(ts_pgo)-1:, 2], motion_err_comp[len(ts_pgo)-1:, 5])

    fig = plt.figure('err-3d')
    ax = plt.axes(projection='3d')
    ax.plot3D(ts_fwd[:, 0], ts_fwd[:, 1], ts_fwd[:, 2], color='red')
    ax.plot3D(ts_gt[:, 0], ts_gt[:, 1], ts_gt[:, 2], color='green')
    color = (motion_err_comp[:len(ts_pgo)-1, 2]>=0)*4-2 + (motion_err_comp[:len(ts_pgo)-1, 5]>=0)*2-1
    color_points = ax.scatter3D(ts_pgo[:-1, 0], ts_pgo[:-1, 1], ts_pgo[:-1, 2], c=color, s=5)
    fig.colorbar(color_points)
    normalize_3d(ax)

    plt.figure('pose')


def align_scale(src, tar):
    N = len(src)
    s = []
    for i in range(N-1):
        s1 = np.linalg.norm(src[i+1] - src[i])
        s2 = np.linalg.norm(tar[i+1] - tar[i])
        s.append(s1 / s2)
    factor = np.median(s)
    print('resacle factor:', factor)
    return tar * factor


def vis_imu_pose(ax, input_dir, col='r', dir_line_scale=0):
    Rs_gt, ts_gt = read_pose_tartanvo(input_dir + '/gt_pose.txt', correct_rot=False)
    # Rs_gt = Rs_gt[850:900]
    # ts_gt = ts_gt[850:900]
    N = len(ts_gt)
    draw_pose(ax, Rs_gt, ts_gt, 'yellow', dir_line_scale=dir_line_scale, dir_line_step=1)

    # Rs_fwd, ts_fwd = read_pose_tartanvo('train_results/vopgo--pwc+11_200k--lr=1e-4' + '/0/pose.txt')
    # Rs_fwd = Rs_fwd[:N]
    # ts_fwd = ts_fwd[:N]
    # Rs_fwd, ts_fwd = transform_pose(Rs_gt[0], ts_gt[0], 0, Rs_fwd, ts_fwd)
    # draw_pose(ax, Rs_fwd, ts_fwd, 'b', dir_line_scale=0, dir_line_step=1)

    # Rs_imu, ts_imu = read_pose_tartanvo(input_dir + '/imu_pose_full.txt')
    # vels_imu = np.loadtxt(input_dir + '/imu_pred_vel_full.txt')
    # vels_gt = np.loadtxt(input_dir + '/imu_vel_world.txt')
    # Rs_imu = Rs_imu[::10]
    # ts_imu = ts_imu[::10]
    # vels_imu = vels_imu[::10]
    # vels_gt = vels_gt[::10]
    # # Rs_imu = Rs_imu[8:15]
    # # ts_imu = ts_imu[8:15]
    # # ts_imu = align_scale(ts_gt, ts_imu)
    # align_idx = 20
    # # Rs_imu, ts_imu = transform_pose(Rs_gt[align_idx], ts_gt[align_idx], align_idx, Rs_imu, ts_imu)
    # # for i in range(len(vels_imu)):
    # #     vels_imu[i] = Rs_imu[i].apply(vels_imu[i])
    # draw_pose(ax, Rs_imu, ts_imu, 'r', line=False, point=True, dir_line_scale=0, dir_line_step=10, vels=vels_imu, vel_scale=0.1, vel_col='cyan')
    # draw_pose(ax, Rs_imu, ts_imu, 'r', line=False, point=True, dir_line_scale=0, dir_line_step=10, vels=vels_gt, vel_scale=0.1, vel_col='b')

    # x = ts_imu[0]
    # xs = []
    # dt = 1.0/10.0 / 10.0
    # for i in range(len(ts_imu)-1):
    #     xs.append(x)
    #     x = x + vels_imu[i] * dt
    # xs = np.stack(xs, axis=0)
    # draw_pose(ax, Rs_imu, xs, 'b', line=False, point=True, dir_line_scale=0, dir_line_step=1)
     
    Rs_imu, ts_imu = read_pose_tartanvo(input_dir + '/imu_pose.txt')
    draw_pose(ax, Rs_imu, ts_imu, col, dir_line_scale=dir_line_scale, dir_line_step=1)

    errs, norms = motion_err(Rs_imu, ts_imu, False, Rs_gt, ts_gt, False)
    print("mean rot err:", np.mean(errs[:, 0]))
    print("mean trans err:", np.mean(errs[:, 1]))
    print("mean rot norm:", np.mean(norms[:, 0]))
    print("mean trans norm:", np.mean(norms[:, 1]))

    return

    vel_gt = np.loadtxt(input_dir + '/imu_gt_vel.txt')
    vel_gt = vel_gt[::10]
    vel_imu = np.loadtxt(input_dir + '/imu_vel.txt')

    vel_err = np.linalg.norm(vel_gt - vel_imu, axis=1)

    # mRs_imu, mts_imu = read_pose_tartanvo(input_dir + '/imu_motion.txt')
  
    # print((Rs_gt.inv() * Rs_imu).magnitude())
    # print(np.linalg.norm(ts_gt - ts_imu, axis=1))
    # errs, norms = motion_err(Rs_gt, ts_gt, False, Rs_imu, ts_imu, False)
    errs, norms = motion_err(Rs_gt, ts_gt, True, Rs_imu, ts_imu, True)
    print("mean rot err:", np.mean(errs[:, 0]))
    print("mean trans err:", np.mean(errs[:, 1]))
    print("mean rot norm:", np.mean(norms[:, 0]))
    print("mean trans norm:", np.mean(norms[:, 1]))

    print(errs[::10, 0].reshape(-1))
    print(errs[::10, 1].reshape(-1))

    # dt = 1.0 / 10
    # at = 2000 * dt
    # rel_trans_err = errs[2000, 1]
    # vel_err = rel_trans_err / dt
    # acc_err = vel_err / at
    # print(acc_err)

    # vo_errs, vo_norms = motion_err(Rs_gt, ts_gt, False, Rs_fwd, ts_fwd, False)
    # print("vo mean rot err:", np.mean(vo_errs[:, 0]))
    # print("vo mean trans err:", np.mean(vo_errs[:, 1]))

    plt.figure('imu-err')
    plt.subplot(1, 3, 1)
    # plt.hist(errs[:, 0])
    x = [i for i in range(len(errs))]
    plt.plot(x, errs[:, 0])
    plt.subplot(1, 3, 2)
    # plt.hist(errs[:, 1])
    plt.plot(x, errs[:, 1])
    plt.subplot(1, 3, 3)
    plt.plot(x, vel_err[:len(x)])

    # Rs_imu, ts_imu = read_pose_tartanvo(input_dir + '/imu_pose_0.txt')
    # ts_imu = align_scale(ts_gt, ts_imu)
    # Rs_imu, ts_imu = transform_pose(Rs_gt[0], ts_gt[0], 0, Rs_imu, ts_imu)
    # draw_pose(ax, Rs_imu, ts_imu, 'r', dir_line_scale=0, dir_line_step=10)

    # Rs_imu, ts_imu = read_pose_tartanvo(input_dir + '/imu_pose_9.8.txt')
    # ts_imu = align_scale(ts_gt, ts_imu)
    # Rs_imu, ts_imu = transform_pose(Rs_gt[0], ts_gt[0], 0, Rs_imu, ts_imu)
    # draw_pose(ax, Rs_imu, ts_imu, 'r', dir_line_scale=0, dir_line_step=10)

    # Rs_imu, ts_imu = read_pose_tartanvo(input_dir + '/imu_pose_-9.8.txt')
    # ts_imu = align_scale(ts_gt, ts_imu)
    # Rs_imu, ts_imu = transform_pose(Rs_gt[0], ts_gt[0], 0, Rs_imu, ts_imu)
    # draw_pose(ax, Rs_imu, ts_imu, 'r', dir_line_scale=0.2, dir_line_step=10)


def vis_train_pose(ax, input_dir, steps):
    # Rs_gt, ts_gt = read_pose_tartanvo(input_dir + '/gt_pose.txt', correct_rot=False)

    # R1 = Rotation.identity()
    # t1 = np.zeros(3)
    # R2 = Rotation.identity()
    # t2 = np.zeros(3)

    Rs, ts = read_pose_tartanvo(input_dir + '/imu_pose.txt')
    R1 = Rs[0]
    t1 = ts[0]
    R2 = Rs[0]
    t2 = ts[0]

    cnt = 0
    for i in range(0, steps):
        Rs, ts = read_pose_tartanvo(input_dir + '/' + str(i) + '/pose.txt')
        Rs, ts = transform_pose(R1, t1, 0, Rs, ts)
        draw_pose(ax, Rs, ts, 'b' if i%2 else 'cyan', dir_line_scale=1, dir_line_step=1)
        cnt += len(Rs)
        R1 = Rs[-1]
        t1 = ts[-1]

        Rs, ts = read_pose_tartanvo(input_dir + '/' + str(i) + '/gt_pose.txt')
        Rs, ts = transform_pose(R2, t2, 0, Rs, ts)
        draw_pose(ax, Rs, ts, 'g', dir_line_scale=1, dir_line_step=1)
        R2 = Rs[-1]
        t2 = ts[-1]

    Rs, ts = read_pose_tartanvo(input_dir + '/imu_pose.txt')
    Rs = Rs[:cnt]
    ts = ts[:cnt]
    # draw_pose(ax, Rs, ts, 'r', dir_line_scale=1, dir_line_step=1)

    vels = np.loadtxt(input_dir + '/imu_gt_vel.txt')
    vels = vels[:cnt]
    vel2t = [ts[0]]
    for v in vels:
        vel2t.append(vel2t[-1] + v*0.1)
    vel2t = np.stack(vel2t)
    draw_pose(ax, Rs, vel2t, 'r', dir_line_scale=1, dir_line_step=1)


def vis_train_motion(ax, input_dir, step=10):
    for k in range(1, step+1):
        mRs_gt, mts_gt = read_pose_rotvec(input_dir + '/' + str(k) + '/gt_motion.txt', correct_rot=False)
        if k == 1:
            Rs_gt, ts_gt = motions2poses(mRs_gt, mts_gt)
        else:
            Rs_gt, ts_gt = motions2poses(mRs_gt, mts_gt, Rs_gt[-1], ts_gt[-1])
        draw_pose(ax, Rs_gt, ts_gt, 'g', dir_line_scale=0, dir_line_step=1)

        mRs, mts = read_pose_rotvec(input_dir + '/' + str(k) + '/motion.txt')
        if k == 1:
            Rs, ts = motions2poses(mRs, mts)
        else:
            Rs, ts = motions2poses(mRs, mts, Rs[-1], ts[-1])
        draw_pose(ax, Rs, ts, 'r', dir_line_scale=0, dir_line_step=1)


def calib_fps():
    input_dir = 'train_results/test_run'
    Rs_gt, ts_gt = read_pose_tartanvo(input_dir + '/gt_pose.txt', correct_rot=False)

    imu_vels = np.loadtxt(input_dir+'/imu_vel.txt')

    for i in range(50):
        dx = ts_gt[i+1, :3] - ts_gt[i, :3]
        v = np.mean(imu_vels[i*10:(i+1)*10, :], axis=0)
        ndx = np.linalg.norm(dx)
        nv = np.linalg.norm(v)
        fps = 1 / (ndx / nv)
        # print(dx, ndx)
        # print(v, nv)
        print(fps)

    cam_time = np.load(input_dir + '/cam_time.npy')
    print(cam_time[:50])
    imu_time = np.load(input_dir + '/imu_time.npy')
    print(imu_time[:50])


def vis_accel_gyro():
    input_dir = 'train_results/test_run'
    accels = np.loadtxt(input_dir + '/imu_accel.txt')
    gyros = np.loadtxt(input_dir + '/imu_gyro.txt')

    norm_accels = np.linalg.norm(accels, axis=1)
    norm_gyros = np.linalg.norm(gyros, axis=1)

    plt.figure('accel')
    # data = norm_accels[80:150]
    # xs = [i for i in range(80,150)]
    data = norm_accels
    xs = [i for i in range(len(data))]
    plt.plot(xs, data)


def exam_scale():
    input_dir = 'train_results/test_run'
    Rs_gt, ts_gt = read_pose_tartanvo('{}/gt_pose.txt'.format(input_dir), correct_rot=False)
    N = len(ts_gt)
    Rs_fwd, ts_fwd = read_pose_tartanvo('{}/0/pose.txt'.format(input_dir))
    try:
        Rs_pgo, ts_pgo = read_pose_tartanvo('{}/0/pgo_pose.txt'.format(input_dir))
    except:
        Rs_pgo, ts_pgo = Rs_gt, ts_gt
    links = read_link_tartanvo('{}/link.txt'.format(input_dir))
    mRs_est, mts_est = read_pose_tartanvo('{}/0/motion.txt'.format(input_dir))

    mRs_gt = []
    mts_gt = []
    for l in links:
        mR, mt = pose2motion(Rs_gt[l[0]], ts_gt[l[0]], Rs_gt[l[1]], ts_gt[l[1]])
        mRs_gt.append(mR.as_quat())
        mts_gt.append(mt)
    mRs_gt = Rotation.from_quat(mRs_gt)
    mts_gt = np.stack(mts_gt, axis=0)

    print(len(mRs_gt), len(mRs_est))
    assert len(mRs_gt) == len(mts_gt) == len(mRs_est) == len(mts_est)

    scales = np.linalg.norm(mts_est, axis=1) / np.linalg.norm(mts_gt, axis=1)
    print(scales)
    
    plt.figure('scale')
    plt.subplot(1, 2, 1)
    plt.hist(scales[:N])
    plt.subplot(1, 2, 2)
    plt.hist(scales[N:])


def imu_delta():
    input_dir = 'train_results/test_run'
    delta = np.loadtxt(input_dir + '/imu_delta.txt')
    acc_delta = delta[:, :3]
    gyro_delta = delta[:, 3:]

    print('acc delta mean:', np.mean(acc_delta, axis=0))
    print('acc delta median:', np.median(acc_delta, axis=0))
    print('gyro delta mean:', np.mean(gyro_delta, axis=0))
    print('gyro delta median:', np.median(gyro_delta, axis=0))


def calc_imu_bias():
    input_dir = 'train_results/test_imu'
    Rs_gt, ts_gt = read_pose_tartanvo('{}/gt_pose.txt'.format(input_dir), correct_rot=False)
    vel_gt_world = np.loadtxt(input_dir + '/imu_gt_vel_world.txt')
    Rs_imu, ts_imu = read_pose_tartanvo('{}/imu_pose.txt'.format(input_dir), correct_rot=False)
    accel_imu = np.loadtxt(input_dir + '/imu_accel.txt')

    draw_pose(ax, Rs_gt, ts_gt, 'g', dir_line_scale=0.1, dir_line_step=1)
    draw_pose(ax, Rs_imu[::10], ts_imu[::10], 'r', dir_line_scale=0.1, dir_line_step=1)

    imu_dt = 1.0 / 100
    accel_gt_world = np.diff(vel_gt_world, axis=0) / imu_dt
    accel_gt_world += np.array([0, 0, -9.81007])
    print(accel_gt_world.shape)
    # accel_gt = Rs_imu[:-1].inv().apply(accel_gt_world)
    accel_gt = accel_gt_world
    accel_imu = Rs_imu[:-1].apply(accel_imu)
    print(accel_gt.shape)

    accel_diff = accel_gt - accel_imu[:-1]
    mean_bias = np.mean(accel_diff, axis=0)
    median_bias = np.median(accel_diff, axis=0)
    # mean_bias = median_bias
    accel_diff_after = accel_gt - (accel_imu[:-1] + mean_bias)

    print(np.median(np.linalg.norm(accel_diff, axis=1)))
    print(np.median(np.linalg.norm(accel_diff_after, axis=1)))

    plt.figure('norm diff')
    plt.subplot(1, 2, 1)
    plt.hist(np.linalg.norm(accel_diff, axis=1))
    plt.subplot(1, 2, 2)
    plt.hist(np.linalg.norm(accel_diff_after, axis=1))

    print(np.mean(accel_diff, axis=0))
    print(np.mean(accel_diff_after, axis=0))

    print(mean_bias)


def vis_imu_motion(ax, input_dir):
    vel_gt = np.loadtxt(input_dir + '/imu_gt_vel_world.txt')
    vel_gt = vel_gt[::10]
    print('vel_gt:', vel_gt.shape)

    Rs_gt, ts_gt = read_pose_tartanvo(input_dir + '/gt_pose.txt')
    print('gt_pose:', len(Rs_gt))

    # dvel_pred = np.loadtxt(input_dir + '/imu_dvel.txt')
    dvel_pred = np.loadtxt(input_dir + '/imu_vel.txt')
    dvel_pred = dvel_pred[1:]
    print('dvel_pred:', dvel_pred.shape)

    mRs_pred, mts_pred = read_pose_tartanvo(input_dir+'/imu_motion.txt')

    accel = np.loadtxt(input_dir + '/imu_accel.txt')
    accel = accel[::10]
    print('accel:', accel.shape)

    vel = vel_gt[0, :]
    print(vel, vel_gt[0])
    print('err:', vel_gt[0]-vel, np.linalg.norm(vel_gt[0]-vel))
    for i in range(5, 6):
    # for i in range(dvel_pred.shape[0]):
        # dvel = Rs_gt[i].apply(dvel_pred[i])
        dvel = dvel_pred[i]
        acc = Rs_gt[i].apply(accel[i]) + np.array([0, 0, 9.81007])
        print('\tdvel:', dvel, np.linalg.norm(dvel))
        print('\t:', (ts_gt[i+1]-ts_gt[i])*10, (ts_gt[i+2]-ts_gt[i+1])*10)
        print('\tvel_gt:', vel_gt[i], vel_gt[i+1])
        print('\tdvel_gt:', vel_gt[i+1]-vel_gt[i], np.linalg.norm(vel_gt[i+1]-vel_gt[i]))
        print('\tacc:', acc, np.linalg.norm(acc))
        print('\tmotion:', mts_pred[i], np.linalg.norm(mts_pred[i]))
        vel += dvel
        print(vel, vel_gt[i+1])
        print('err:', vel_gt[i+1]-vel, np.linalg.norm(vel_gt[i+1]-vel))


plt.figure('pose')
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# tartanvo(ax, 'euroc-st=1-itv=10-strict', 'euroc')
# tartanvo(ax, 'euroc-st=1-adj=2.1-short', 'euroc')
# tartanvo(ax, 'tartanair-st=1-itv=100-gtloop', 'tartanair')
# tartanvo(ax, 'tartanair-st=1-adj=2.1-short', 'tartanair')
# tartanvo(ax, 'train_results/test_run', 'tartanair')
# tartanvo(ax, 'train_results/test_run_pwc+1_1_200000', 'tartanair')
# vis_imu_pose(ax, 'train_results/test_euroc/t9.81/test', 'r', 0.01)
# vis_imu_pose(ax, 'train_results/test_euroc/tib/test', 'purple', 0.01)
# vis_imu_pose(ax, 'train_results/t0b/test', 'yellow')
# vis_imu_pose(ax, 'train_results/tib/test', 'purple')
# vis_imu_pose(ax, 'train_results/tflu/test', 'r', 1)
# vis_imu_pose(ax, 'train_results/txyz/test', 'r', 1)
# vis_train_pose(ax, 'train_results/test_run', [0])
# vis_train_pose(ax, 'train_results/vopgo--pwc+11_5k--lr=1e-5', [0, 40])
# calib_fps()
# vis_accel_gyro()
# exam_scale()
# imu_delta()
# calc_imu_bias()
# vis_imu_motion(ax, 'train_results/test_run')

# vis_train_pose(ax, 'test_results/test', 1)
vis_train_pose(ax, 'test_results/test30', 1)

normalize_3d(ax)

plt.show()
