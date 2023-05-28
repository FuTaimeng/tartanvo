#!/bin/sh

#SBATCH --cluster=ub-hpc
###SBATCH --cluster=faculty

#SBATCH --partition=general-compute --qos=general-compute
###SBATCH --partition=scavenger --qos=scavenger

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

###SBATCH --gres=gpu:1
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
###SBATCH --gres=gpu:tesla_v100-pcie-16gb:2
###SBATCH --gres=gpu:nvidia_a16:12

#SBATCH --mem=40000

#SBATCH --job-name="train_vo_pvgo"

###SBATCH --output= "result_$(date +"%Y_%m_%d_%k_%M_%S").out"

###SBATCH --mail-user=taimengf@buffalo.edu
###SBATCH --mail-type=ALL

###SBATCH --requeue


source ~/.bashrc
conda activate impe-learning
# conda activate /projects/academic/cwx/shaoshus/packages/miniconda3/envs/shaoshu

CUDA_VISIBLE_DEVICES=1


ds_name=MH_04_difficult

data_dir=/user/taimengf/projects/euroc/${ds_name}/mav0

loss_weight='(1,0.1,10,1)'
rot_w=1
trans_w=0.1
lr=1e-5
epoch=1
train_portion=1
project_name=euroc_${ds_name}_p${train_portion}
train_name=sr_imu0b

rm -r train_results/${project_name}/${train_name}
mkdir -p train_results/${project_name}/${train_name}
rm -r train_results_models/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}

python train.py \
    --result-dir train_results/${project_name}/${train_name} \
    --save-model-dir train_results_models/${project_name}/${train_name} \
    --project-name ${project_name} \
    --train-name ${train_name} \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --batch-size 8 \
    --worker-num 2 \
    --data-root ${data_dir} \
    --start-frame 100 \
    --end-frame -1 \
    --train-epoch ${epoch} \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --mode train-all \
    --use-stereo 1 \
    --data-type euroc \
    --fix-model-parts 'flow' 'stereo' \
    --use-pvgo \
    --rot-w ${rot_w} \
    --trans-w ${trans_w} \
    --delay-optm \
    --train-portion ${train_portion}
# | tee train_results/${project_name}/${train_name}/log.txt
