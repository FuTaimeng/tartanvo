source ~/.bashrc
conda activate impe-learning

ds_date=2011_09_30
ds_idx=27
data_dir=/user/taimengf/projects/kitti_raw/${ds_date}/${ds_date}_drive_00${ds_idx}_sync

loss_weight='(1,0.1,10,0.1)'
rot_w=1
trans_w=0.1
lr=3e-6
epoch=10
train_portion=1

use_scale=false
if [ "$use_scale" = true ]; then
    exp_type='mono'
else
    exp_type='stereo'
fi

project_name=test
train_name=${rot_w}Ra_${trans_w}ta_delayOptm_lr=${lr}_${loss_weight}_${exp_type}

echo -e "\n=============================================="
echo "project name = ${project_name}"
echo "train name = ${train_name}"
echo "=============================================="

rm -r train_results/${project_name}/${train_name}
mkdir -p train_results/${project_name}/${train_name}
rm -r train_results_models/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}

if [ "$use_scale" = true ]; then
    # mono
    python train.py \
        --result-dir train_results/${project_name}/${train_name} \
        --save-model-dir train_results_models/${project_name}/${train_name} \
        --project-name ${project_name} \
        --train-name ${train_name} \
        --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
        --batch-size 8 \
        --worker-num 2 \
        --data-root ${data_dir} \
        --start-frame 0 \
        --end-frame -1 \
        --train-epoch ${epoch} \
        --print-interval 1 \
        --snapshot-interval 1 \
        --lr ${lr} \
        --loss-weight ${loss_weight} \
        --mode train-all \
        --use-stereo 1 \
        --data-type kitti \
        --fix-model-parts 'flow' 'stereo' \
        --use-pvgo \
        --rot-w ${rot_w} \
        --trans-w ${trans_w} \
        --delay-optm \
        --train-portion ${train_portion} \
        --use-gt-scale
else
    # stereo not use scale
    python train.py \
        --result-dir train_results/${project_name}/${train_name} \
        --save-model-dir train_results_models/${project_name}/${train_name} \
        --project-name ${project_name} \
        --train-name ${train_name} \
        --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
        --batch-size 8 \
        --worker-num 2 \
        --data-root ${data_dir} \
        --start-frame 0 \
        --end-frame -1 \
        --train-epoch ${epoch} \
        --print-interval 1 \
        --snapshot-interval 1 \
        --lr ${lr} \
        --loss-weight ${loss_weight} \
        --mode train-all \
        --use-stereo 1 \
        --data-type kitti \
        --fix-model-parts 'flow' 'stereo' \
        --use-pvgo \
        --rot-w ${rot_w} \
        --trans-w ${trans_w} \
        --delay-optm \
        --train-portion ${train_portion}
fi
