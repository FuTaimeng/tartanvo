
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000
data_dir=~/projects/euroc/MH_01_easy/mav0

loss_weight='(0.01,0.1,10,1)'
lr=1e-5
optm=adam
# train_name=new_imu-vo-pvgo_lw=${loss_weight}_optm=${optm}-lr=${lr}_imuscale
train_name=test

rm -r train_results/${train_name}
mkdir train_results/${train_name}

python train.py \
    --data-root ${data_dir} \
    --data-type euroc \
    --result-dir train_results/${train_name} \
    --train-name ${train_name} \
    --flow-model-name pwc_net.pth.tar \
    --pose-model-name multicamvo_posenet_init_stereo=0.pkl \
    --batch-size 64 \
    --worker-num 4 \
    --start-frame 0 \
    --end-frame 50 \
    --train-step 1 \
    --print-interval 1 \
    --snapshot-interval 100 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --mode train \
    --vo-optimizer ${optm} \
| tee train_results/${train_name}/log.txt

    # --only-backpropagate-loop-edge \jingtong0
    