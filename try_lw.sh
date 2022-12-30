
# data_dir=data/EuRoC_V102
data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000

for l1 in 0.01 0.1 1 ;  
do  
    for l2 in 0.01 0.1 1 ;
    do
        for l3 in 1 10 100;
        do
            for l4 in 0.1 1 10 ;
            do

loss_weight=\(${l1},${l2},${l3},${l4}\)
train_name=imu-stereovo-pvgo_lw=${loss_weight}

echo ${train_name}

root_dir=try_lw_stereo
rm -r ${root_dir}/${train_name}
mkdir -p ${root_dir}/${train_name}

# python train.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name pwc_net.pth.tar \
#     --pose-model-name 1_1_sample_voflow_200000.pkl \
#     --batch-size 1 \
#     --worker-num 1 \
#     --test-dir ${data_dir}/image_left \
#     --pose-file ${data_dir}/pose_left.txt \
#     --sample-step 1 \
#     --start-frame 850 \
#     --end-frame 900 \
#     --train-step 1 \
#     --print-interval 1 \
#     --snapshot-interval 10 \
#     --lr 1e-4 \
#     --imu-dir ${data_dir}/imu \
#     --device cuda:3 \
#     --loss-weight ${loss_weight} \
# > ${root_dir}/${train_name}/log.txt

python train.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --vo-model-name 43_6_2_vonet_30000.pkl \
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
    --lr 1e-4 \
    --imu-dir ${data_dir}/imu \
    --device cuda:4 \
    --loss-weight ${loss_weight} \
    --mode test \
    --use-stereo \
> ${root_dir}/${train_name}/log.txt

            done
        done
    done
done  


    