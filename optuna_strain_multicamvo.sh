#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:2
###SBATCH --gres=gpu:nvidia_a16:1

#SBATCH --mem=32000

#SBATCH --job-name="train_multicamvo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

#SBATCH --mail-user=taimengf@buffalo.edu
#SBATCH --mail-type=ALL

###SBATCH --requeue

#SBATCH --array=1-4


source ~/.bashrc
conda activate impe-learning

data_dir=/user/taimengf/projects/tartanair/TartanAir

batch=32
step=5000

nick_name=ExMLP-ExNorm
train_name=${nick_name}_optuna[nel,ntl]

# CUDA_VISIBLE_DEVICES=0

python optuna_train_multicamvo2.py \
    --flow-model-name ./models/pwc_net.pth.tar \
    --batch-size ${batch} \
    --worker-num 4 \
    --data-root ${data_dir} \
    --print-interval 5 \
    --snapshot-interval 500 \
    --device cuda \
    --mode train-all \
    --random-intrinsic 800 \
    --hsv-rand 0.2 \
    --use-stereo 2.2 \
    --fix_model_parts 'flow' 'feat' 'rot' \
    --result-dir ./train_multicamvo \
    --train-name ${train_name} \
    --debug-flag '' \
    --pose-model-name ./models/multicamvo_posenet_init_stereo=2.2.pkl \
    --train-step ${step} \
    --test-interval 50 \
    --tuning-val 'extrinsic_encoder_layers' 'trans_head_layers' \
    --enable-pruning \
    --trail-num 2 \
    --out-to-cml \
    --lr 6e-6 \
    --load-study \

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
