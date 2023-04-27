
# data_dir=data/EuRoC_V102
# data_dir=/user/taimengf/projects/tartanair/TartanAir/abandonedfactory/Easy/P000
# data_dir=/user/taimengf/projects/kitti_raw/2011_10_03/2011_10_03_drive_0042_sync
data_dir=/user/taimengf/projects/kitti_raw/2011_09_30/2011_09_30_drive_0016_sync
# data_dir=/user/taimengf/projects/kitti_raw/2011_09_30/2011_09_30_drive_0016_sync

loss_weight='(10,0,1,0)'
lr=1e-5
project_name=test_16
train_name=noskip_IMUxyz_canny_${loss_weight}

rm -r train_results/${project_name}/${train_name}
mkdir -p train_results/${project_name}/${train_name}

CUDA_VISIBLE_DEVICES=5
CUDA_LAUNCH_BLOCKING=1

python train.py \
    --result-dir train_results/${project_name}/${train_name} \
    --project-name ${project_name} \
    --train-name ${train_name} \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --batch-size 8 \
    --worker-num 2 \
    --data-root ${data_dir} \
    --start-frame 0 \
    --end-frame -1 \
    --train-step 1000 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --mode train-all \
    --use-stereo 1 \
    --data-type kitti \
    --fix-model-parts 'flow' 'stereo' \
    --use-pvgo \
| tee train_results/${project_name}/${train_name}/log.txt
    