import os,torch
import numpy as np
import pypose as pp
import torch.utils.data as Data


class G2OPGO(Data.Dataset):
    def __init__(self, root, dataname, device='cpu'):
        super().__init__()

        def info2mat(info):
            mat = np.zeros((6,6))
            ix = 0
            for i in range(mat.shape[0]):
                mat[i,i:] = info[ix:ix+(6-i)]
                mat[i:,i] = info[ix:ix+(6-i)]
                ix += (6-i)
            return mat
        self.dtype = torch.get_default_dtype()
        filename = os.path.join(root, dataname)
        ids, nodes, edges, poses, infos = [], [], [], [], []
        with open(filename) as f:
            for line in f:
                line = line.split()
                if line[0] == 'VERTEX_SE3:QUAT':
                    ids.append(torch.tensor(int(line[1]), dtype=torch.int64))
                    nodes.append(pp.SE3(np.array(line[2:], dtype=np.float64)))
                elif line[0] == 'EDGE_SE3:QUAT':
                    edges.append(torch.tensor(np.array(line[1:3], dtype=np.int64)))
                    poses.append(pp.SE3(np.array(line[3:10], dtype=np.float64)))
                    infos.append(torch.tensor(info2mat(np.array(line[10:], dtype=np.float64))))

        self.ids = torch.stack(ids)
        self.nodes = torch.stack(nodes).to(self.dtype).to(device)
        self.edges = torch.stack(edges).to(device) # have to be LongTensor
        self.poses = torch.stack(poses).to(self.dtype).to(device)
        self.infos = torch.stack(infos).to(self.dtype).to(device)
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)


class VOPGO(Data.Dataset):
    def __init__(self, poses_np, motions, links, infomats=None, device='cpu'):
        super().__init__()

        N = poses_np.shape[0]
        M = len(links)
        self.dtype = torch.get_default_dtype()

        self.ids = torch.arange(0, N, dtype=torch.int64).view(-1, 1)
        self.nodes = pp.SE3(poses_np).to(self.dtype).to(device)
        self.edges = torch.tensor(links, dtype=torch.int64).to(device)
        self.poses = pp.SE3(motions.detach()).to(self.dtype).to(device)
        self.poses_withgrad = pp.SE3(motions).to(self.dtype).to(device)
        if infomats is not None:
            raise NotImplementedError
        else:
            self.infos = torch.stack([torch.eye(6)]*M).to(self.dtype).to(device)
            
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)


class PVGO_Dataset():
    def __init__(self, nodes, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, dts, device='cpu'):
        N = nodes.shape[0]
        M = motions.shape[0]
        self.dtype = torch.get_default_dtype()
        self.device = device

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.detach().clone()

        self.ids = torch.arange(0, N, dtype=torch.int64).view(-1, 1)
        self.edges = torch.tensor(links, dtype=torch.int64).to(device)
        self.poses = self.to_SE3(motions.detach())
        self.poses_withgrad = self.to_SE3(motions)
        
        # No use
        self.infos = torch.stack([torch.eye(6)] * M).to(self.dtype).to(device)

        self.imu_drots = self.to_SO3(imu_drots_np)
        self.imu_dtrans = self.to_tensor(imu_dtrans_np)
        self.imu_dvels = self.to_tensor(imu_dvels_np)
        self.dts = self.to_tensor(dts)
        
        init_with_imu_rot = True
        if init_with_imu_rot:
            if isinstance(nodes, pp.LieTensor):
                rot = nodes[0].rotation()
            else:
                rot = self.to_SO3(nodes[0, 3:])
            rots = [rot]
            for drot in self.imu_drots:
                rot = rot @ drot
                rots.append(rot)
            rots = torch.stack(rots)
            if isinstance(nodes, pp.LieTensor):
                trans = nodes.translation()
            else:
                trans = self.to_tensor(nodes[:, :3])
            # print(N, rots.size(0), trans.size(0))
            assert N == rots.size(0) == trans.size(0)
            self.nodes = self.to_SE3(torch.cat([trans, rots.tensor()], dim=1))
        else:
            self.nodes = self.to_SE3(nodes)
            
        vels_ = torch.diff(self.nodes.translation(), dim=0) / self.dts.view(-1, 1)
        self.vels = torch.cat((vels_, vels_[-1].view(1, 3)), dim=0)
            
        assert N == self.ids.size(0) == self.nodes.size(0) == self.vels.size(0)
        assert M == self.edges.size(0) == self.poses.size(0)
        assert N-1 == self.imu_drots.size(0) == self.imu_dtrans.size(0) == self.imu_dvels.size(0)
    
    def align_nodes(self, rot, trans, idx, nodes):
        tar = pp.SE3(np.concatenate((trans, rot))).to(nodes.dtype).to(nodes.device)
        return tar @ nodes[idx].Inv() @ nodes

    def to_SE3(self, x):
        if isinstance(x, pp.LieTensor):
            if x.ltype == pp.SE3_type:
                return x.to(self.dtype).to(self.device)
            elif x.ltype == pp.se3_type:
                return x.Exp().to(self.dtype).to(self.device)
        elif x.shape[-1] == 7:
            return pp.SE3(x).to(self.dtype).to(self.device)
        elif x.shape[-1] == 6:
            return pp.se3(x).Exp().to(self.dtype).to(self.device)
        return None

    def to_SO3(self, x):
        if isinstance(x, pp.LieTensor):
            if x.ltype == pp.SO3_type:
                return x.to(self.dtype).to(self.device)
            elif x.ltype == pp.so3_type:
                return x.Exp().to(self.dtype).to(self.device)
        elif x.shape[-1] == 4:
            return pp.SO3(x).to(self.dtype).to(self.device)
        elif x.shape[-1] == 3:
            return pp.so3(x).Exp().to(self.dtype).to(self.device)
        return None

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.dtype).to(self.device)
        return torch.tensor(x).to(self.dtype).to(self.device)

