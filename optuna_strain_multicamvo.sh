#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

###SBATCH --gpus=nvidia_a100-pcie-40gb:2
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
###SBATCH --gres=gpu:nvidia_a16:1

#SBATCH --mem=32000

#SBATCH --job-name="train_multicamvo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

#SBATCH --mail-user=taimengf@buffalo.edu
#SBATCH --mail-type=ALL

###SBATCH --requeue

###SBATCH --array=1-4


source ~/.bashrc
conda activate impe-learning

data_dir=/user/taimengf/projects/tartanair/TartanAir

batch=16
step=100

nick_name=Dist-4A16-B16
train_name=${nick_name}

# export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
export CUDA_VISIBLE_DEVICES=8,9,10,11

python optuna_train_multicamvo2.py \
    --flow-model-name ./models/pwc_net.pth.tar \
    --batch-size ${batch} \
    --worker-num 2 \
    --data-root ${data_dir} \
    --print-interval 5 \
    --snapshot-interval 500 \
    --mode train-all \
    --random-intrinsic 800 \
    --hsv-rand 0.2 \
    --use-stereo 2.2 \
    --fix_model_parts 'flow' 'feat' 'rot' \
    --result-dir ./train_multicamvo \
    --train-name ${train_name} \
    --debug-flag '' \
    --pose-model-name ./models/multicamvo_posenet_init_stereo=0.pkl \
    --train-step ${step} \
    --test-interval 50 \
    --lr 6e-6 \
    --world-size 4 \
    --not-write-log \
    --tcp-port 65532

    # --tuning-val 'lr' \
    # --lr-lb  1e-7 \
    # --lr-ub  1e-3 \
    # --enable-pruning \
    # --trail-num 10 \
    
    # --lr 6e-6 \
    # --trail-num 1 \
    # --enable-decay \

    # --out-to-cml \
    # --not-write-log \

    # --load-study \
    # --study-name multicamvo_B32_St500_optuna_lr_testset_dev3090_Feb_27_2023_12_44_00 \
    # --start-iter 100000 \
