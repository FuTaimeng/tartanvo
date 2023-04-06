#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

###SBATCH --gpus=nvidia_a100-pcie-40gb:2
#SBATCH --gres=gpu:2
###SBATCH --gres=gpu:tesla_v100-pcie-32gb:2
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
###SBATCH --gres=gpu:nvidia_a16:1

#SBATCH --mem=80000

#SBATCH --job-name="train_multicamvo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

#SBATCH --mail-user=taimengf@buffalo.edu
#SBATCH --mail-type=ALL

###SBATCH --requeue

###SBATCH --array=1-2


source ~/.bashrc
conda activate impe-learning

# data_dir=/user/taimengf/projects/tartanair/TartanAir
data_dir=/user/taimengf/projects/tartanair/TartanAir/abandonedfactory/Easy/P000

batch=1
step=10

nick_name=CalcScale
train_name=${nick_name}

# export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
export CUDA_VISIBLE_DEVICES=8,9,10,11

python optuna_train_multicamvo2.py \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --batch-size ${batch} \
    --worker-num 2 \
    --data-root ${data_dir} \
    --print-interval 5 \
    --snapshot-interval 500 \
    --mode train-all \
    --random-intrinsic 0 \
    --hsv-rand 0.2 \
    --use-stereo 1 \
    --result-dir ./train_multicamvo \
    --train-name ${train_name} \
    --debug-flag '' \
    --train-step ${step} \
    --test-interval 50 \
    --world-size 1 \
    --lr 1e-6 \
    --stereo-data-type 's' \
    --vo-optimizer adam \
    --fix-model-parts 'flow' 'stereo' \
    --not-write-log

    # --flow-model-name ./models/pwc_net.pth.tar \
    # --use-stereo 2.2 \
    # --fix_model_parts 'flow' 'feat' 'rot' \
    # --pose-model-name ./models/multicamvo_posenet_init_stereo=0.pkl \

    # --tuning-val 'lr' \
    # --lr-lb  1e-7 \
    # --lr-ub  1e-3 \
    # --enable-pruning \
    # --trial-num 10 \
    # --load-study \
    
    # --lr 6e-6 \
    # --enable-decay \

    # --out-to-cml \
    # --not-write-log \

