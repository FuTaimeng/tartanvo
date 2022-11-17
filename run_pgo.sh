work_dir=/home/user/workspace/tartanvo
ds_name=euroc_tartanvo_1914
# ds_name=tartanair_tartanvo_1914

python pgo.py \
    --device cuda:0 \
    --save ${work_dir}/results/pgo_save \
    --dataroot ${work_dir}/results \
    --dataname ${ds_name}.g2o
