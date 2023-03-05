
# data_dir=data/EuRoC_V102
# data_dir=/data/datasets/wenshanw/tartan_data
# data_dir=/data/tartanair
data_dir=/home/data2/TartanAir/TartanAir_comb
# data_dir=/user/shaoshus/projects/tartanair/TartanAir

# lr=1e-5
batch=32
step=2000

root_dir=train_multicamvo
# train_name=multicamvo_lr=${lr}_batch=${batch}_step=${step}_SepFeatEncoder
# train_name="test_4e-5_1000_tunetrans"
# train_name=all_frames
# continue_from=multicamvo_lr=1e-5_batch=32_step=100000_SepFeatEncoder_s=12500
# train_name = debug_autotuna_lr_batch=${batch}_step=${step}_10Scenes_s=29000
# train_name=multicamvo_B${batch}_St${step}_optuna_lr_mlp_encoder


# train
train_name=multicamvo_B${batch}_St${step}_lr_mlp_encoder
echo "working dir: ${PWD}"
echo "train_name: ${train_name}"
# tuning lr
python optuna_train_multicamvo2.py \
    --flow-model-name ./models/pwc_net.pth.tar \
    --batch-size ${batch} \
    --worker-num 4 \
    --data-root ${data_dir} \
    --print-interval 5 \
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
    --pose-model-name ./models/multicamvo_B32_St100000_optuna_fix_lr_dev3090_Feb_28_2023_14_39_24_B32_lr6_103e-06_posenet_100000.pkl \
    --train-step ${step} \
    --test-interval 10 \
    --gpu-id 1 \
    --vo-optimizer 'rmsprop' \
    --tuning-val 'lr' \
    --lr-lb  1e-7 \
    --lr-ub  1e-3 \
    --enable-pruning \
    --trail-num 10 \
    --out-to-cml \
    # --load-study \
    # --study-name multicamvo_B32_St2000_with_100000_model_dev3090_Mar_03_2023_21_42_55 \
   

    # fix lr
    # --lr 1e-5	 \
    # --trail-num 10 \
    # --out-to-cml \
    # --trunk-value 0.9939128725683963 \
    # --not-write-log \
    # --enable-decay \
    
    # continue traning
    # --start-iter 100000 \ 
    # --not-write-log
    # --pose-model-name ./models/multicamvo_B32_St100000_optuna_lr_dev3090_Feb_21_2023_01_13_55_B32_lr3_843e-06_posenet_100000.pkl \
