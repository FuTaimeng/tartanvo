#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:tesla_v100-pcie-32gb:2
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:2
###SBATCH --gres=gpu:nvidia_a16:12

#SBATCH --mem=10000

#SBATCH --job-name="train_multicamvo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

#SBATCH --mail-user=taimengf@buffalo.edu
#SBATCH --mail-type=ALL

###SBATCH --requeue


source ~/.bashrc
conda activate impe-learning


# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data
# data_dir=/data/tartanair
data_dir=~/projects/tartanair/TartanAir

lr=1e-5
batch=32
step=100000

root_dir=train_multicamvo
train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_FixNormBug
# train_name="test_4e-5_1000_tunetrans"
# train_name=test
# continue_from=multicamvo_lr=1e-5_batch=32_step=100000_10Scenes_s=29000
# train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_10Scenes_s=44000

rm -r ${root_dir}/${train_name}
mkdir ${root_dir}/${train_name}

export CUDA_VISIBLE_DEVICES=0

# train
python train_multicamvo.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --flow-model-name models/pwc_net.pth.tar \
    --pose-model-name models/multicamvo_posenet_init_stereo=2.2.pkl \
    --batch-size ${batch} \
    --worker-num 1 \
    --data-root ${data_dir} \
    --train-step ${step} \
    --print-interval 10 \
    --snapshot-interval 500 \
    --lr ${lr} \
    --device cuda \
    --mode train-all \
    --debug-flag 012 \
    --random-intrinsic 800 \
    --hsv-rand 0.2 \
    --use-stereo 2.2 \
    --fix_model_parts 'flow' 'feat' 'rot' \
| tee ${root_dir}/${train_name}/log.txt

# continue
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name ${root_dir}/${continue_from}/models/multicamvo_posenet_15000.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step ${step} \
#     --print-interval 10 \
#     --snapshot-interval 100 \
#     --lr ${lr} \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 012 \
#     --random-intrinsic 800 \
#     --hsv-rand 0.2 \
#     --use-stereo 2.2 \
#     --fix_model_parts 'flow' 'feat' 'rot' \
#     --vo-optimizer rmsprop \
# | tee ${root_dir}/${train_name}/log.txt

# test
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name train_multicamvo/multicamvo_lr=4e-5_batch=64_step=10000_tunetrans/models/multicamvo_posenet_100.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step 1 \
#     --print-interval 1 \
#     --snapshot-interval 100 \
#     --lr ${lr} \
#     --lr-decay-rate 0.5 \
#     --lr-decay-point '()' \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 012 \
#     --random-intrinsic 800 \
#     --hsv-rand 0.2 \
# | tee ${root_dir}/${train_name}/log.txt

# cd ${root_dir}
# zip -r -q ${train_name}.zip ${train_name} 
