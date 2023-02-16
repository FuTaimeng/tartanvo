
# data_dir=data/EuRoC_V102
data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000

loss_weight='(0.01,0.01,1,0.1)'
lr=1e-5
root_dir=test_results
# train_name=imu-vo-pvgo_lw=${loss_weight}_imuscale_loopclosure
train_name=temp

rm -r ${root_dir}/${train_name}
mkdir ${root_dir}/${train_name}

python train.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --vo-model-name 4_2_2_vonet_5000.pkl \
    --batch-size 1 \
    --worker-num 1 \
    --image-dir ${data_dir}/image_left \
    --right-image-dir ${data_dir}/image_right \
    --pose-file ${data_dir}/pose_left.txt \
    --sample-step 1 \
    --start-frame 850 \
    --end-frame 900 \
    --train-step 200 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --imu-dir ${data_dir}/imu \
    --device cuda:3 \
    --loss-weight ${loss_weight} \
    --mode test \
    --use-stereo 1 \
    # --use-loop-closure \
    # --use-stop-constraint \
# > ${root_dir}/${train_name}/log.txt

    # --only-backpropagate-loop-edge \
    # --pose-model-name 1_1_sample_voflow_200000.pkl \