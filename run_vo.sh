
data_dir=data/EuRoC_V102
# data_dir=/data/datasets/tartanair/abandonedfactory/Easy/P000

python vo_trajectory_from_folder.py \
    --model-name tartanvo_1914.pkl \
    --batch-size 1 \
    --worker-num 1 \
    --test-dir ${data_dir}/image_left \
    --pose-file ${data_dir}/pose_left.txt \
    --sample-step 1 \
    --start-frame 850 \
    --end-frame 900 \
    --euroc \
    