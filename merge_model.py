import torch

def print_dict(state_dict):
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())


# posenet_dir = './train_multicamvo/StereoVO_AllEnv_Proc2_lrDec0.5/models/StereoVO_AllEnv_Proc2_lrDec0.5_B32_lr6.000e-05_Oadam/'
# posenet_name = 'StereoVO_AllEnv_Proc2_lrDec0.5_B32_lr6.000e-05_Oadam_st10000.pkl'
# posenet_dict = torch.load(posenet_dir + posenet_name)

# base_file = './models/43_6_2_vonet_30000.pkl'
# base_dict = torch.load(base_file)

# # print_dict(posenet_dict)
# # print_dict(base_dict)

# merged_dict = {}
# for k, v in posenet_dict.items():
#     kk = k.replace('module.', 'module.flowPoseNet.')
#     merged_dict[kk] = v
# for k, v in base_dict.items():
#     if k.startswith('module.flowNet.') or k.startswith('module.stereoNet.'):
#         merged_dict[k] = v

# print_dict(merged_dict)

# torch.save(merged_dict, './models/' + posenet_name)


base_file = './models/cvt_tartanvo_1914.pkl'
base_dict = torch.load(base_file)

stereo_name = './models/43_6_2_vonet_30000.pkl'
stereo_dict = torch.load(stereo_name)

for k, v in stereo_dict.items():
    if k.startswith('module.stereoNet.'):
        base_dict[k] = v
print_dict(base_dict)

torch.save(base_dict, './models/stereo_cvt_tartanvo_1914.pkl')