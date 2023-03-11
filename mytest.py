import torch
import pypose as pp
import numpy as np
from scipy.spatial.transform import Rotation

fname = 'train_multicamvo/Dist-1A16-block/log_P0.txt'
with open(fname, 'r') as f:
    lines = f.readlines()

ts = []
for line in lines:
    sp = line.split()
    for ch in sp:
        if ch.startswith('ife:'):
            val = float(ch[4:])
            ts.append(val)

print(len(ts))
print(sum(ts)/len(ts)*100)