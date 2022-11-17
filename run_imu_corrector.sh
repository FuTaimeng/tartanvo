data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000

python imu_corrector.py \
    --device cuda:2 \
    --batch-size 4 \
    --max_epoches 100 \
    --dataroot ${data_dir}/imu \
    --dataname ${data_dir}/pose_left.txt