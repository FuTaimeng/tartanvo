import torch
import torch.nn as nn
import torch.optim as optim

from VOFlowNet import VOFlowRes


def print_dict(state_dict):
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())


def cvt_state_dict(keep_others=True):
    net = TartanVO()
    # net = VOFlowRes(intrinsic=True, down_scale=True, config=1, stereo=int(stereo))
    # print_dict(net.state_dict())

    pretrain_name = 'tartanvo_1914.pkl'
    pretrain_dict = torch.load('models/'+pretrain_name)
    print_dict(pretrain_dict)

    replace_dict = {
        'firstconv':'feat_net',
        'layer1':'feat_net.3',
        'layer2':'feat_net.4',
        'layer3':'feat_net.5',
        'layer4':'feat_net.6',
        'layer5':'feat_net.7',
        'voflow_trans':'voflow_trans',
        'voflow_rot':'voflow_rot'
    }

    state_dict = {}
    for k in pretrain_dict:
        flag = False
        for st in replace_dict:
            if k.startswith(st) or k.startswith('module.flowPoseNet.'+st):
                kk = k.replace(st, replace_dict[st])
                state_dict[kk] = pretrain_dict[k]
                flag = True
                break
        if not flag and keep_others:
            state_dict[k] = pretrain_dict[k]

    if pretrain_name == '43_6_2_vonet_30000.pkl':
        state_dict['feat_net.0.0.weight'] = state_dict['feat_net.0.0.weight'][:, (0,1,3,4), ...]

    if stereo==2.2 or stereo==3:
        temp = {}
        for k in state_dict:
            if k.startswith('feat_net'):
                kk = k.replace('feat_net', 'feat_net2')
                temp[kk] = state_dict[k]
        state_dict.update(temp)

    model_dict = net.state_dict()
    for k in state_dict:
        if k not in model_dict:
            print('[{}] in pretrain but not in model'.format(k))
        elif state_dict[k].size() != model_dict[k].size():
            print('[{}] size mismatch: {} - {}'.format(k, state_dict[k].size(), model_dict[k].size()))

    for k in model_dict:
        if k not in state_dict:
            print('[{}] in model but not in pratrain'.format(k))
            # if k.endswith('weight'):
            #     print('\tinit with kaiming_normal_')
            #     w = torch.rand_like(model_dict[k])
            #     nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
            # else:
            #     print('\tinit to zeros')
            #     w = torch.zeros_like(model_dict[k])
            # state_dict[k] = w

    save_name = 'cvt_'+pretrain_name
    torch.save(state_dict, 'models/' + save_name)



stereo = 0
cvt_state_dict()

# save_name = 'multicamvo_posenet_init_stereo={}.pkl'.format(stereo)
# x = torch.load('model_init/' + save_name)



# save_name = '43_6_2_vonet_30000.pkl'
# x = torch.load('model_init/' + save_name)
# # print_dict(x)
# for k in x:
#     print(k, x[k].shape)