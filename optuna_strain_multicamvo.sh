
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data
# data_dir=/data/tartanair
data_dir=/home/data/tartanair/TartanAir_comb

# lr=1e-5
batch=32
step=100000

root_dir=train_multicamvo
# train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_SepFeatEncoder
# train_name="test_4e-5_1000_tunetrans"
# train_name=all_frames
# continue_from=multicamvo_lr=1e-5_batch=32_step=100000_SepFeatEncoder_s=12500
# train_name = debug_autotuna_lr_batch=${batch}_step=${step}_10Scenes_s=29000

train_name=multicamvo_B${batch}_St${step}_optuna_lr

# rm -r ${root_dir}/${train_name}
# mkdir ${root_dir}/${train_name}

# export CUDA_VISIBLE_DEVICES=0
echo "train_name: ${train_name}"

# train
# train
python optuna_train_multicamvo2.py \
    --flow-model-name ./models/pwc_net.pth.tar \
    --batch-size ${batch} \
    --worker-num 4 \
    --data-root ${data_dir} \
    --print-interval 10 \
    --snapshot-interval 1000 \
    --device cuda \
    --mode train-all \
    --random-intrinsic 800 \
    --hsv-rand 0.2 \
    --use-stereo 2.2 \
    --fix_model_parts 'flow' 'feat' 'rot' \
    --result-dir ./train_multicamvo \
    --train-name ${train_name} \
    --debug-flag '' \
    --pose-model-name ./models/multicamvo_posenet_15000.pkl \
    --train-step ${step} \
    --trail-num 5 \
    --enable-pruning

    #  --out-to-cml
# python train_multicamvo.py \
#     --result-dir ${root_dir}/${train_name} \
#     --train-name ${train_name} \
#     --pose-model-name models/multicamvo_posenet_init_stereo=2.2.pkl \
#     --batch-size ${batch} \
#     --worker-num 1 \
#     --data-root ${data_dir} \
#     --train-step ${step} \
#     --print-interval 10 \
#     --snapshot-interval 500 \
#     --lr ${lr} \
#     --device cuda \
#     --mode train-all \
#     --debug-flag 012 \
#     --random-intrinsic 800 \
#     --hsv-rand 0.2 \
#     --use-stereo 2.2 \
#     --fix_model_parts 'flow' 'feat' 'rot' \
# | tee ${root_dir}/${train_name}/log.txt


