
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data
data_dir=/data/tartanair

lr=1e-4
batch=64
step=1000

root_dir=train_multicamvo
train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}
# continue_from=multicamvo_lr=2e-4_batch=128_step=1000
# train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_continue=\(${continue_from}\)

rm -r ${root_dir}/${train_name}
mkdir ${root_dir}/${train_name}

export CUDA_VISIBLE_DEVICES=0

# train
python train_multicamvo.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --flow-model-name models/pwc_net.pth.tar \
    --pose-model-name models/multicamvo_posenet_init.pkl \
    --batch-size ${batch} \
    --worker-num 1 \
    --data-root ${data_dir} \
    --train-step ${step} \
    --print-interval 10 \
    --snapshot-interval 100 \
    --lr ${lr} \
    --lr-decay-rate 0.4 \
    --lr-decay-point '()' \
    --device cuda \
    --mode train-all \
    --debug-flag 0 \
    --random-intrinsic 600 \
    --hsv-rand 0.2 \
| tee ${root_dir}/${train_name}/log.txt

# continue
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name ${root_dir}/${continue_from}/models/multicamvo_posenet_1000.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step ${step} \
#     --print-interval 10 \
#     --snapshot-interval 100 \
#     --lr ${lr} \
#     --lr-decay-rate 0.4 \
#     --lr-decay-point '(0.5, 0.75, 0.875)' \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 0 \
# | tee ${root_dir}/${train_name}/log.txt

# test
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name models/multicamvo_posenet_init.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step ${step} \
#     --print-interval 1 \
#     --snapshot-interval 1 \
#     --lr ${lr} \
#     --lr-decay-rate 0.4 \
#     --lr-decay-point '(0.5, 0.75, 0.875)' \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 01234 \
#     --random-intrinsic 600 \
#     --hsv-rand 0.2 \
# | tee ${root_dir}/${train_name}/log.txt

# cd ${root_dir}
# zip -r -q ${train_name}.zip ${train_name} 
