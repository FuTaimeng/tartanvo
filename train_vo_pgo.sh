
# data_dir=data/EuRoC_V102
data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000

loss_weight='(0.01,0.1,10,1)'
train_name=new_imu-vo-pvgo_lw=${loss_weight}

rm -r train_results/${train_name}
mkdir train_results/${train_name}

python train.py \
    --result-dir train_results/${train_name} \
    --train-name ${train_name} \
    --flow-model-name pwc_net.pth.tar \
    --pose-model-name 1_1_sample_voflow_200000.pkl \
    --batch-size 1 \
    --worker-num 1 \
    --test-dir ${data_dir}/image_left \
    --pose-file ${data_dir}/pose_left.txt \
    --sample-step 1 \
    --start-frame 850 \
    --end-frame 880 \
    --train-step 10 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr 1e-4 \
    --imu-dir ${data_dir}/imu \
    --device cuda:3 \
    --loss-weight ${loss_weight} \
> train_results/${train_name}/log.txt

    # --only-backpropagate-loop-edge \jingtong0
    