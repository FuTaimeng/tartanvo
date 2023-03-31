import cv2
import numpy as np
import pypose as pp

import torch
from torch.masked import masked_tensor

from timer import Timer


def proj(x):
    return x / x[..., -1:]

def scale_from_disp_flow(disp, flow, motion, fx, fy, cx, cy, baseline):
    height, width = disp.shape[-2:]

    T = pp.SE3(motion.detach())
    # T.requires_grad = True

    disp_th = 1
    flow_th = 0.1*height
    flow_norm = torch.linalg.norm(flow, dim=0)
    mask = torch.logical_and(disp >= disp_th, flow_norm <= flow_th)
    # mask2 = torch.stack([mask, mask])
    # m_disp = masked_tensor(disp, mask)
    # m_flow = masked_tensor(flow, mask2)
    m_disp = torch.where(mask, disp, disp_th)
    # m_disp = disp
    # m_flow = flow

    # m_disp_gray = to_image(m_disp.numpy())
    # cv2.imwrite('m_disp_gray.png', m_disp_gray)

    z = fx*baseline / m_disp

    # z_gray = to_image(z.numpy()*10)
    # cv2.imwrite('z_gray.png', z_gray)
    
    u_lin = torch.linspace(0, width-1, width)
    v_lin = torch.linspace(0, height-1, height)
    u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
    uv = torch.stack([u, v])
    uv1 = torch.stack([u, v, torch.ones_like(u)])

    # u_gray = to_image(u.numpy()*0.5)
    # v_gray = to_image(v.numpy()*0.5)
    # cv2.imwrite('u_gray.png', u_gray)
    # cv2.imwrite('v_gray.png', v_gray)    

    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3)
    K_inv = torch.linalg.inv(K)

    P = z.unsqueeze(-1) * (K_inv.unsqueeze(0).unsqueeze(0) @ uv1.permute(1, 2, 0).unsqueeze(-1)).squeeze()

    R = T.Inv().rotation()
    t = T.Inv().translation()
    # scale = torch.linalg.norm(t)
    # t_norm = t / scale
    t_norm = torch.nn.functional.normalize(t, dim=0)
    a = (K @ t_norm.view(3, 1)).squeeze().unsqueeze(0).unsqueeze(0)
    b = (K.unsqueeze(0).unsqueeze(0) @ (R.unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)).squeeze()
    # print(a.shape, b.shape)

    f = (flow + uv).permute(1, 2, 0)
    M1 = a[..., 2] * f[..., 0] - a[..., 0]
    w1 = b[..., 0] - b[..., 2] * f[..., 0]
    M2 = a[..., 2] * f[..., 1] - a[..., 1]
    w2 = b[..., 1] - b[..., 2] * f[..., 1]
    # print(M.shape, w.shape)

    # print(torch.sum(mask))
    m_M1 = M1.view(-1)[mask.view(-1)]
    m_M2 = M2.view(-1)[mask.view(-1)]
    m_w1 = w1.view(-1)[mask.view(-1)]
    m_w2 = w2.view(-1)[mask.view(-1)]

    M = torch.stack([m_M1, m_M2]).view(-1, 1)
    w = torch.stack([m_w1, m_w2]).view(-1, 1)
    s = 1 / torch.sum(M * M) * M.t() @ w
    s = s.item()

    # print(s, scale)

    T = pp.SE3(torch.cat([s * t, R.tensor()]))

    reproj = K.unsqueeze(0).unsqueeze(0) @ proj(T.Inv().unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)
    reproj = reproj.squeeze().permute(2, 0, 1)[:2, ...]

    # reproj_rgb = np.concatenate([reproj.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj.shape[1:]), axis=-1)], axis=-1)*0.5
    # reproj_rgb = to_image(reproj_rgb)
    # uv_rgb = np.concatenate([uv.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(uv.shape[1:]), axis=-1)], axis=-1)*0.5
    # uv_rgb = to_image(uv_rgb)
    # cv2.imwrite('reproj_rgb.png', reproj_rgb)
    # cv2.imwrite('uv_rgb.png', uv_rgb)

    r = reproj - (flow + uv)

    # flow_dest = flow + uv
    # flow_dest_rgb = np.concatenate([flow_dest.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(flow_dest.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('flow_dest_rgb.png', flow_dest_rgb)
    # reproj_flow = reproj - uv
    # reproj_flow_rgb = np.concatenate([reproj_flow.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj_flow.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('reproj_flow_rgb.png', reproj_flow_rgb)

    return r, s


path = '/user/taimengf/projects/tartanair/TartanAir/abandonedfactory/Easy/P000'

def to_image(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def tartan2kitti(pose):
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T = pp.from_matrix(T, ltype=pp.SE3_type)
    if len(pose.shape) == 2:
        T = T.view(1, 7)

    new_pose = T @ pose @ T.Inv()

    return new_pose

if __name__ == '__main__':
    timer = Timer()

    flow = np.load(path+'/flow/000000_000001_flow.npy')
    depth = np.load(path+'/depth_left/000000_left_depth.npy')
    disp = 320*0.25 / depth

    # img1 = cv2.imread(path+'/image_left/000000_left.png')
    # img2 = cv2.imread(path+'/image_left/000001_left.png')
    # cv2.imwrite('img1.png', img1)
    # cv2.imwrite('img2.png', img2)

    # img1_warp = warp_flow(img1, flow)
    # cv2.imwrite('img1_warp.png', img1_warp)

    # flow_rgb = np.concatenate([flow*10, np.expand_dims(np.zeros(flow.shape[:-1]), axis=-1)], axis=2)
    # flow_rgb = to_image(flow_rgb)
    # depth_gray = to_image(depth*10)
    # disp_gray = to_image(disp)
    # cv2.imwrite('flow_rgb.png', flow_rgb)
    # cv2.imwrite('depth_gray.png', depth_gray)
    # cv2.imwrite('disp_gray.png', disp_gray)

    # print('disp max min', np.max(disp), np.min(disp))
    # print('depth max min', np.max(depth), np.min(depth))

    flow = torch.from_numpy(flow).permute(2, 0, 1)
    disp = torch.from_numpy(disp)

    poses = np.loadtxt(path+'/pose_left.txt')
    poses = pp.SE3(poses)
    # poses = tartan2kitti(poses)
    motion = poses[0].Inv() @ poses[1]
    motion = tartan2kitti(motion)
    t = motion.translation()
    scale = torch.linalg.norm(t)

    # print(motion)

    timer.tic('fwd')
    r, s = disp_flow_ba(disp, flow, motion, 320, 320, 320, 240, 0.25)
    timer.toc('fwd')

    # img1_reproj = warp_flow(img1, reproj_flow.permute(1, 2, 0).numpy())
    # cv2.imwrite('img1_reproj.png', img1_reproj)

    print('r.shape', r.shape)
    print('max', torch.max(r), 'min', torch.min(r))
    print('s', s, 'scale', scale)
    print('fwd time', timer.last('fwd'))

    # r_np = r.detach().permute(1, 2, 0).numpy()
    # r_gray = np.linalg.norm(r_np, axis=2)
    # r_gray = to_image(r_gray)
    # r_rgb = np.concatenate([r_np, np.expand_dims(np.zeros(r_np.shape[:-1]), axis=-1)], axis=2)
    # r_rgb = to_image(r_rgb)
    # cv2.imwrite('r_rgb.png', r_rgb)
    # cv2.imwrite('r_gray.png', r_gray)


