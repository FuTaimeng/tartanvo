
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000
data_dir=~/projects/tartanair/TartanAir/abandonedfactory/Easy/P000

loss_weight='(0.01,0.1,10,1)'
lr=1e-5
optm=adam
train_name=test

rm -r train_results/${train_name}
mkdir train_results/${train_name}

python train.py \
    --result-dir train_results/${train_name} \
    --train-name ${train_name} \
    --flow-model-name models/pwc_net.pth.tar \
    --pose-model-name models/multicamvo_posenet_init_stereo=3.pkl \
    --batch-size 1 \
    --worker-num 1 \
    --image-dir ${data_dir}/image_left \
    --pose-file ${data_dir}/pose_left.txt \
    --sample-step 1 \
    --start-frame 0 \
    --end-frame 32 \
    --train-step 200 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --imu-dir ${data_dir}/imu \
    --device cuda \
    --loss-weight ${loss_weight} \
    --mode test \
    --vo-optimizer ${optm} \
| tee train_results/${train_name}/log.txt

    # --only-backpropagate-loop-edge \jingtong0
    