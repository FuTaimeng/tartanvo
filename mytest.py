import torch
import pypose as pp
import numpy as np
from scipy.spatial.transform import Rotation

R_mag = []
t_mag = []

for step in range(10, 10000, 10):
    fname = 'train_multicamvo/multicamvo_lr=5e-6_batch=32_step=10000_SepFeatEncoder_s=7500/debug/{}/gt_motion.txt'.format(step)
    gt_motion = np.loadtxt(fname)

    R = Rotation.from_rotvec(gt_motion[:, 3:])
    t = gt_motion[:, :3]

    R_mag.extend(np.rad2deg(R.magnitude()))
    t_mag.extend(np.linalg.norm(t, axis=1))

print(len(R_mag), len(t_mag))
print(np.mean(R_mag), np.var(R_mag))
print(np.mean(t_mag), np.var(t_mag))
