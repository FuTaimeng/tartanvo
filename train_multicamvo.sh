
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data
data_dir=/data/tartanair

lr=1e-5
batch=32
step=10000

root_dir=train_multicamvo
# train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_SepFeatEncoder
# train_name="test_4e-5_1000_tunetrans"
# train_name=all_frames
continue_from=multicamvo_lr=1e-5_batch=32_step=10000_SepFeatEncoder
train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_SepFeatEncoder_s=5000

rm -r ${root_dir}/${train_name}
mkdir ${root_dir}/${train_name}

export CUDA_VISIBLE_DEVICES=0

# train
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name models/multicamvo_posenet_init_stereo=2.2.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step ${step} \
#     --print-interval 10 \
#     --snapshot-interval 100 \
#     --lr ${lr} \
#     --lr-decay-rate 0.5 \
#     --lr-decay-point 0.5 0.75 \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 0 \
#     --random-intrinsic 800 \
#     --hsv-rand 0.2 \
#     --use-stereo 2.2 \
#     --fix_model_parts 'flow' 'feat' 'rot' \
# | tee ${root_dir}/${train_name}/log.txt

# continue
python train_multicamvo.py \
    --result-dir ${root_dir}/${train_name} \
    --train-name ${train_name} \
    --flow-model-name models/pwc_net.pth.tar \
    --pose-model-name ${root_dir}/${continue_from}/models/multicamvo_posenet_5000.pkl \
    --batch-size ${batch} \
    --worker-num 1 \
    --data-root ${data_dir} \
    --train-step ${step} \
    --print-interval 10 \
    --snapshot-interval 100 \
    --lr ${lr} \
    --device cuda \
    --mode train-all \
    --debug-flag 012 \
    --random-intrinsic 800 \
    --hsv-rand 0.2 \
    --use-stereo 2.2 \
    --fix_model_parts 'flow' 'feat' 'rot' \
| tee ${root_dir}/${train_name}/log.txt

# test
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --flow-model-name models/pwc_net.pth.tar \
#     --pose-model-name train_multicamvo/multicamvo_lr=4e-5_batch=64_step=10000_tunetrans/models/multicamvo_posenet_100.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step 1 \
#     --print-interval 1 \
#     --snapshot-interval 100 \
#     --lr ${lr} \
#     --lr-decay-rate 0.5 \
#     --lr-decay-point '()' \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 012 \
#     --random-intrinsic 800 \
#     --hsv-rand 0.2 \
# | tee ${root_dir}/${train_name}/log.txt

# cd ${root_dir}
# zip -r -q ${train_name}.zip ${train_name} 
