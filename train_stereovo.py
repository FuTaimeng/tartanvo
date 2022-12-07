
import cv2
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np
from Datasets.data_roots import *
from Datasets.MultiDatasets import EndToEndStereoMultiDatasets, FlowMultiDatasets, StereoMultiDatasets
import random

# from scipy.io import savemat
np.set_printoptions(precision=4, threshold=10000, suppress=True)

import time # for testing

from Network.StereoVONet import StereoVONet

class TrainStereoVONet(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainStereoVONet, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'vonet'

        # import ipdb;ipdb.set_trace()
        # stereonorm = 80.0 / args.stereo_baseline_x_focal * 2.5 # 80: baseline x focal in tartanair; 2.5: the difference of normalization in stereonet and posenet
        stereonorm = 0.02 # the norm factor for the stereonet
        self.vonet = StereoVONet(network=args.network, intrinsic=self.args.intrinsic_layer, 
                            flowNormFactor=1.0, stereoNormFactor=stereonorm, poseDepthNormFactor=0.25, 
                            down_scale=args.downscale_flow, config=args.resvo_config, 
                            fixflow=args.fix_flow, fixstereo=args.fix_stereo, autoDistTarget=args.auto_dist_target)

        # load stereo
        if args.load_stereo_model:
            modelname0 = self.args.working_dir + '/models/' + args.stereo_model
            self.load_model(self.vonet.stereoNet, modelname0)

        # load flow
        if args.load_flow_model:
            modelname1 = self.args.working_dir + '/models/' + args.flow_model
            if args.flow_model.endswith('tar'): # load pwc net
                data = torch.load(modelname1)
                self.vonet.flowNet.load_state_dict(data)
                print('load pwc network...')
            else:
                self.load_model(self.vonet.flowNet, modelname1)

        # load pose
        if args.load_pose_model:
            modelname2 = self.args.working_dir + '/models/' + args.pose_model
            self.load_model(self.vonet.flowPoseNet, modelname2)

        # load the whole model
        if self.args.load_model:
            modelname = self.args.working_dir + '/models/' + self.args.model_name
            self.load_model(self.vonet, modelname)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.network==0 or args.network==1: # pwcnet
            flowmean = None
            flowstd = None

        self.pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013] # hard code, use when save motionfile when testing

        self.LrDecrease = [int(self.args.train_step/2), 
                            int(self.args.train_step*3/4), 
                            int(self.args.train_step*7/8)]
        self.lr = self.args.lr
        self.lr_flow = self.args.lr_flow
        self.lr_stereo = self.args.lr_stereo

        if not self.args.test: 
            if self.args.train_vo: # dataloader for end2end flow vo
                self.trainDataloader = EndToEndStereoMultiDatasets(self.args.data_file, self.args.train_data_type, self.args.train_data_balence, 
                                                args, self.args.batch_size, self.args.worker_num,  
                                                mean=mean, std=std)
                self.voflowOptimizer = optim.Adam(self.vonet.parameters(), lr = self.lr)
            if self.args.train_flow: # dataloader for flow 
                self.trainFlowDataloader = FlowMultiDatasets(self.args.flow_file, self.args.flow_data_type, self.args.flow_data_balence,
                                                        self.args, self.args.batch_size, self.args.worker_num,
                                                        mean = flowmean, std = flowstd)
                self.flowOptimizer = optim.Adam(self.vonet.flowNet.parameters(),lr = self.lr_flow)
            if self.args.train_stereo: # dataloader for stereo 
                self.trainStereoDataloader = StereoMultiDatasets(self.args.stereo_file, self.args.stereo_data_type, self.args.stereo_data_balence,
                                                        self.args, self.args.batch_size, self.args.worker_num)
                self.stereoOptimizer = optim.Adam(self.vonet.stereoNet.parameters(),lr = self.lr_stereo)

            self.testDataloader = EndToEndStereoMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std)
        else: 
            self.testDataloader = EndToEndStereoMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std, shuffle= (not args.test_traj))

        self.criterion = nn.L1Loss()

        if self.args.multi_gpu>1:
            self.vonet = nn.DataParallel(self.vonet)

        self.vonet.cuda()

    def initialize(self):
        super(TrainStereoVONet, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('flow', 100)
        self.add_accumulated_value('stereo', 100)
        self.add_accumulated_value('pose', 100)
        self.add_accumulated_value('vo_flow', 100)
        self.add_accumulated_value('vo_stereo', 100)

        self.add_accumulated_value('test', 1)
        self.add_accumulated_value('t_flow', 1)
        self.add_accumulated_value('t_stereo', 1)
        self.add_accumulated_value('t_pose', 1)

        self.add_accumulated_value('t_trans', 1)
        self.add_accumulated_value('t_rot', 1)
        self.add_accumulated_value('trans', 100)
        self.add_accumulated_value('rot', 100)
        self.append_plotter("loss", ['loss', 'test'], [True, False])
        self.append_plotter("loss_flow", ['flow', 'vo_flow', 't_flow'], [True, True, False])
        self.append_plotter("loss_pose", ['pose', 't_pose'], [True, False])
        self.append_plotter("loss_stereo", ['stereo', 'vo_stereo', 't_stereo'], [True, True, False])
        self.append_plotter("trans_rot", ['trans', 'rot', 't_trans', 't_rot'], [True, True, False, False])

        if self.args.test_traj: # additional plot for testing
            self.add_accumulated_value('trans_norm', 100)
            self.add_accumulated_value('rot_norm', 100)
            self.append_plotter("loss_norm", ['trans_norm', 'rot_norm'], [True, True])

        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        self.count = 0
        self.test_count = 0
        self.epoch = 0

        super(TrainStereoVONet, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.vonet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward_stereo(self, sample, mask=False, stereo_norm=0.02): # stereo_norm is currently hard coded
        # this is used for pure stereo training and testing
        leftTensor = sample['img0'].squeeze(1).cuda()
        rightTensor = sample['img1'].squeeze(1).cuda()
        output = self.vonet(None, None, leftTensor,rightTensor, only_stereo=True)

        if self.args.no_gt: # run test w/o GT file
            if self.args.network==0 and self.stereonet.training: # PSMNet + training
                output = output[2]
            return 0, output/stereo_norm

        targetdisp = sample['disp0'].squeeze(1).cuda()
        targetdisp = targetdisp * stereo_norm # normalize the output for numerical stability

        if mask:
            valid_mask = targetdisp>0 # in kitti dataset, value 0 is set to unmesured pixels
        else:
            valid_mask = None

        if self.args.multi_gpu>1:
            loss = self.vonet.module.get_stereo_loss(output, targetdisp, self.criterion, mask=valid_mask)
        else:
            loss = self.vonet.get_stereo_loss(output, targetdisp, self.criterion, mask=valid_mask)

        return loss/stereo_norm, output/stereo_norm

    def forward_flow(self, sample, use_mask=False): 
        img1Tensor = sample['img0'][:,0,:,:,:].cuda()
        img2Tensor = sample['img0'][:,1,:,:,:].cuda()
        output = self.vonet(img1Tensor,img2Tensor, x0_stereo=None, x1_stereo=None, only_flow=True)
        targetflow = sample['flow'].squeeze(1).cuda()

        if not use_mask:
            mask = None
        else:
            mask = sample['fmask'].squeeze(1).cuda()
        if self.args.multi_gpu>1:
            loss = self.vonet.module.get_flow_loss(output, targetflow, self.criterion, mask=mask)
        else:
            loss = self.vonet.get_flow_loss(output, targetflow, self.criterion, mask=mask) #flow_loss(output, targetflow, use_mask, mask)

        return loss/self.args.normalize_output, output

    def forward_vo(self, sample, use_mask=False, stereo_norm=0.02):
        if self.args.network ==0 or self.args.network ==1: # PWC-Net
            img0_flow   = sample['img0'][:,0,:,:,:].cuda()
            img1_flow   = sample['img0'][:,1,:,:,:].cuda()
        else: # 
            img0_flow   = sample['img0_norm'][:,0,:,:,:].cuda()
            img1_flow   = sample['img0_norm'][:,1,:,:,:].cuda()
        intrinsic = sample['intrinsic'].squeeze(1).cuda()
        if 'scale_w' in sample:
            scale_w = sample['scale_w'].cuda()
            scale_w = scale_w.view(scale_w.shape + (1,1))
        else:
            scale_w = 1.0

        img0_stereo   = sample['img0_norm'][:,0,:,:,:].cuda()
        img1_stereo   = sample['img1_norm'].squeeze(1).cuda()

        flow, mask = None, None
        if 'flow' in sample:
            flow = sample['flow'].squeeze(1).cuda()
            if use_mask:
                mask = sample['fmask'].squeeze(1).cuda()

        disp = None
        if 'disp0' in sample:
            disp = sample['disp0'].squeeze(1).cuda()
            disp = disp * stereo_norm
            disp_scale = disp * scale_w
        # import ipdb;ipdb.set_trace()
        blxfx = sample['blxfx'].view((sample['blxfx'].shape[0], 1, 1, 1)).cuda()
        if random.random()>self.args.vo_gt_flow: 
            flow_output, stereo_output, pose_output = self.vonet(img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsic, 
                                                                 scale_w=scale_w, scale_disp=self.args.scale_disp,
                                                                 blxfx = blxfx)
        else:
            flow_output, stereo_output, pose_output = self.vonet(img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsic, 
                                                                 scale_w=scale_w, gt_flow=flow, gt_disp=disp, 
                                                                 scale_disp=self.args.scale_disp,
                                                                 blxfx = blxfx)
        pose_output_np = pose_output.data.cpu().detach().numpy()

        if self.args.no_gt: 
            return 0., 0., 0., 0.,0., pose_output_np

        # import ipdb;ipdb.set_trace()
        # calculate flow loss
        if self.args.multi_gpu>1:
            flowloss = self.vonet.module.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output
            stereoloss = self.vonet.module.get_stereo_loss(stereo_output, disp_scale, self.criterion) / stereo_norm
        else:
            flowloss = self.vonet.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output #flow_loss(flow_output, flow, use_mask, mask, small_scale=self.args.downscale_flow )/self.args.normalize_output
            stereoloss = self.vonet.get_stereo_loss(stereo_output, disp_scale, self.criterion) / stereo_norm
        # calculate vo loss
        motion = sample['motion'].squeeze(1)

        lossPose = self.criterion(pose_output, motion.cuda())
        diff = torch.abs(pose_output.data.cpu().detach() - motion)
        trans_loss = diff[:,:3].mean().item()
        rot_loss = diff[:,3:].mean().item()

        # # for debug: 
        # if flowloss.item()>50:
        #     from Datasets.utils import tensor2img, visflow
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225]
        #     img1 = tensor2img(img0_flow[0].cpu(),mean,std)
        #     img2 = tensor2img(img1_flow[0].cpu(),mean,std)
        #     tflow = flow.cpu().squeeze().numpy().transpose(1,2,0) * 20.0
        #     oflow = flow_output[0].cpu().squeeze().numpy().transpose(1,2,0) * 20.0
        #     countstr = str(self.count)
        #     cv2.imwrite(countstr+'_img1.png', img1)
        #     cv2.imwrite(countstr+'_img2.png', img2)
        #     np.save(countstr+'_tflow.npy', tflow)
        #     np.save(countstr+'_oflow.npy', tflow)
        #     tflowvis = visflow(tflow)
        #     oflowvis = visflow(oflow)
        #     cv2.imwrite(countstr+'_tflowvis.png', tflowvis)
        #     cv2.imwrite(countstr+'_oflowvis.png', oflowvis)


        return flowloss, stereoloss, lossPose, trans_loss, rot_loss, pose_output_np


    def train(self):
        super(TrainStereoVONet, self).train()

        self.count = self.count + 1
        self.vonet.train()

        starttime = time.time()

        # train flow
        if self.args.train_flow: # not a vo only training
            flowsample, flowmask = self.trainFlowDataloader.load_sample()
            self.flowOptimizer.zero_grad()
            flowloss, _ = self.forward_flow(flowsample, use_mask=flowmask)
            flowloss.backward()
            self.flowOptimizer.step()
            self.AV['flow'].push_back(flowloss.item(), self.count)

        flowtime = time.time() 

        # train stereo
        if self.args.train_stereo: # not a vo only training
            stereosample, stereomask = self.trainStereoDataloader.load_sample()
            self.stereoOptimizer.zero_grad()
            stereoloss, _ = self.forward_stereo(stereosample, mask=stereomask)
            stereoloss.backward()
            self.stereoOptimizer.step()
            self.AV['stereo'].push_back(stereoloss.item(), self.count)

        stereotime = time.time() 

        if self.args.train_vo: # not a flow only training
            self.voflowOptimizer.zero_grad()
            sample, vo_flowmask = self.trainDataloader.load_sample()
            loadtime = time.time()
            flowloss, stereoloss, poseloss, trans_loss, rot_loss, _ = self.forward_vo(sample, use_mask=vo_flowmask)
            loss = poseloss
            if not self.args.fix_flow:
                loss = loss + flowloss * self.args.lambda_flow 
            if not self.args.fix_stereo:
                loss = loss + stereoloss * self.args.lambda_stereo #             

            loss.backward()
            self.voflowOptimizer.step()

            # import ipdb;ipdb.set_trace()
            self.AV['loss'].push_back(loss.item(), self.count)
            self.AV['vo_flow'].push_back(flowloss.item(), self.count)
            self.AV['vo_stereo'].push_back(stereoloss.item(), self.count)
            self.AV['pose'].push_back(poseloss.item(), self.count)
            self.AV['trans'].push_back(trans_loss, self.count)
            self.AV['rot'].push_back(rot_loss, self.count)

        nntime = time.time()

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                if self.args.train_vo:
                    self.lr = self.lr*0.4
                    for param_group in self.voflowOptimizer.param_groups: 
                        param_group['lr'] = self.lr
                if self.args.train_flow:
                    self.lr_flow = self.lr_flow*0.4
                    for param_group in self.flowOptimizer.param_groups: 
                        param_group['lr'] = self.lr_flow
                if self.args.train_stereo:
                    self.lr_stereo = self.lr_stereo * 0.4
                    for param_group in self.stereoOptimizer.param_groups: 
                        param_group['lr'] = self.lr_stereo

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr:%.6f - time(%.2f,%.2f,%.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr, flowtime-starttime, stereotime-flowtime, nntime-stereotime))

        if self.count % self.args.plot_interval == 0: 
            self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()

    def test(self):
        super(TrainStereoVONet, self).test()
        self.test_count += 1

        self.vonet.eval()
        sample, mask = self.testDataloader.load_sample()

        with torch.no_grad():
            flowloss, stereoloss, poseloss, trans_loss, rot_loss, motion = self.forward_vo(sample, use_mask=mask)

        finish = self.test_count*motion.shape[0]>= self.testDataloader.datalens[0]
        motion_unnorm = motion.squeeze() * self.pose_norm

        if self.args.no_gt:
            if self.test_count % self.args.print_interval == 0:
                self.logger.info("  TEST %s #%d - output : %s"  % (self.args.exp_prefix[:-1], 
                    self.test_count, motion_unnorm))
            return 0, 0, 0, 0, 0, 0, motion_unnorm, finish

        loss = flowloss * self.args.lambda_flow + stereoloss * self.args.lambda_stereo + poseloss  # 

        lossnum = loss.item()
        self.AV['test'].push_back(lossnum, self.count)
        self.AV['t_flow'].push_back(flowloss.item(), self.count)
        self.AV['t_stereo'].push_back(stereoloss.item(), self.count)
        self.AV['t_pose'].push_back(poseloss.item(), self.count)
        self.AV['t_trans'].push_back(trans_loss, self.count)
        self.AV['t_rot'].push_back(rot_loss, self.count)

        self.logger.info("  TEST %s #%d - (loss, flow, stereo, pose, rot, trans) %.4f  %.4f  %.4f  %.4f  %.4f  %.4f"  % (self.args.exp_prefix[:-1], 
            self.test_count, loss.item(), flowloss.item(), stereoloss.item(), poseloss.item(), rot_loss, trans_loss))

        return lossnum, flowloss.item(), stereoloss.item(), poseloss.item(), trans_loss, rot_loss, motion_unnorm, finish

    def finalize(self):
        super(TrainStereoVONet, self).finalize()
        if self.count < self.args.train_step and not self.args.test and not self.args.test_traj:
            self.dumpfiles()

        if self.args.test and not self.args.no_gt:
            self.logger.info('The average loss values: (t-trans, t-rot, t-flow, t-pose)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['test'].last_avg(100), 
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100),
                self.AV['t_flow'].last_avg(100),
                self.AV['t_pose'].last_avg(100)))

        else:
            self.logger.info('The average loss values: (loss, trans, rot, test, t_trans, t_rot)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans'].last_avg(100),
                self.AV['rot'].last_avg(100),
                self.AV['test'].last_avg(100),
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100)))


if __name__ == '__main__':
    args = get_args()

    if args.use_int_plotter:
        plottertype = 'Int'
    else:
        plottertype = 'Visdom'
    try:
        # Instantiate an object for MyWF.
        trainVOFlow = TrainStereoVONet(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)
        trainVOFlow.initialize()

        if args.test:
            errorlist = []
            motionlist = []
            finish = False
            while not finish:
                error0, error1, error2, error3, error4, error5, motion, finish = trainVOFlow.test()
                errorlist.append([error0, error1, error2, error3, error4, error5])
                motionlist.append(motion)
                if ( trainVOFlow.test_count == args.test_num ):
                    break
            errorlist = np.array(errorlist)
            print("Test reaches the maximum test number (%d)." % (args.test_num))
            print("Loss statistics: loss/flow/stereo/pose/trans/rot: (%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f)" % (errorlist[:,0].mean(),
                            errorlist[:,1].mean(), errorlist[:,2].mean(), errorlist[:,3].mean(), errorlist[:,4].mean(), errorlist[:,5].mean()))

            if args.test_traj:
                # save motion file
                outputdir_prefix = args.test_output_dir+'/'+args.model_name.split('vonet')[0]+args.val_file.split('/')[-1].split('.txt')[0] # trajtest/xx_xx_euroc_xx
                motionfilename = outputdir_prefix +'_output_motion.txt'
                motions = np.array(motionlist)
                np.savetxt(motionfilename, motions)
                # visualize the file 
                # import ipdb;ipdb.set_trace()
                from error_analysis import evaluate_trajectory
                from evaluator.transformation import motion_ses2pose_quats, pose_quats2motion_ses
                from Datasets.utils import per_frame_scale_alignment
                gtposefile = args.gt_pose_file
                gtposes = np.loadtxt(gtposefile)
                gtmotions = pose_quats2motion_ses(gtposes)
                # estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
                estposes = motion_ses2pose_quats(motions)
                evaluate_trajectory(gtposes, estposes, trajtype=args.test_data_type, outfilename=outputdir_prefix, scale=False, medir_dir=args.test_output_dir)
        else: # Training
            while True:
                trainVOFlow.train()
                if (trainVOFlow.count >= args.train_step):
                    break

        trainVOFlow.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainVOFlow.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")


