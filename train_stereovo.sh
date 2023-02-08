CUDA_VISIBLE_DEVICES=3 \
python train_stereo_vo_wf.py \
    --exp-prefix 4_2_3_ \
    --use-int-plotter \
    --batch-size 64 \
    --worker-num 1 \
    --train-step 20000 \
    --snapshot 5000 \
    --multi-gpu 1 \
    --lr 0.0001 --lr-decay \
    --fix-flow \
    --fix-stereo \
    --train-vo \
    --data-file data/tartan_train.txt \
    --train-data-type tartan \
    --val-file data/tartan_test.txt \
    --image-height 448 \
    --image-width 640 \
    --normalize-output 0.05 \
    --downscale-flow \
    --intrinsic-layer \
    --resvo-config 1 \
    --network 1 \
    --random-intrinsic 600 \
    --hsv-rand 0.2 \
    --load-model \
    --model-name 4_2_2_vonet_5000.pkl \
    --platform cluster \


    # --load-flow-model \
    # --flow-model 26_1_2_flow_430000.pkl \
    # --load-stereo-model \
    # --stereo-model 5_5_4_stereo_30000.pkl \