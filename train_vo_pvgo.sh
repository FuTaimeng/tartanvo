# data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000
# data_type=tartanvo
data_dir=~/projects/euroc/MH_01_easy/mav0
data_type=euroc
# data_dir=~/projects/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync
# data_type=kitti

loss_weight='(0.01,0.1,10,1)'
lr=1e-5
optm=adam
# train_name=new_imu-vo-pvgo_lw=${loss_weight}_optm=${optm}-lr=${lr}_imuscale
train_name=test

rm -r train_results/${train_name}
mkdir train_results/${train_name}

python train.py \
    --data-root ${data_dir} \
    --data-type ${data_type} \
    --result-dir train_results/${train_name} \
    --train-name ${train_name} \
    --flow-model-name models/pwc_net.pth.tar \
    --pose-model-name models/multicamvo_B32_St100000_optuna_lr_dev3090_Feb_21_2023_01_13_55_B32_lr3_843e-06_posenet_100000.pkl \
    --batch-size 32 \
    --worker-num 4 \
    --start-frame 0 \
    --end-frame 500 \
    --train-step 10 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --mode train \
    --vo-optimizer ${optm} \
    --use-stereo 2.2 \
| tee train_results/${train_name}/log.txt

    # --only-backpropagate-loop-edge \jingtong0
    