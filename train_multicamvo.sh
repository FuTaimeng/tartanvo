
# data_dir=data/EuRoC_V102
data_dir=/data/datasets/wenshanw/tartan_data
lr=1e-4
root_dir=train_results
train_name=multicamvo_lr=${lr}
# train_name=temp

rm -r ${root_dir}/${train_name}
mkdir ${root_dir}/${train_name}

CUDA_VISIBLE_DEVICES=2,3 \
python train_multicamvo.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --flow-model-name pwc_net.pth.tar \
    --batch-size 16 \
    --worker-num 1 \
    --data-root ${data_dir} \
    --train-step 1 \
    --print-interval 1 \
    --snapshot-interval 1 \
    --lr ${lr} \
    --device cuda \
    --mode train-all \
# > ${root_dir}/${train_name}/log.txt
