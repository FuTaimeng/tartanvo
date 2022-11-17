import torch
import pypose as pp

# a = pp.randn_SE3(1).tensor()
# a.requires_grad = True
# print(a)
# b = 5 * a
# print(b)
# c = pp.SE3(b)
# print(c)
# d = pp.SE3(b.detach())
# print(d)

a = torch.tensor([1, 2, 3, 4]).view(2,2)
b = torch.cat((a, a), dim=1)
print(b)