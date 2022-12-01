data_dir=/data/datasets/wenshanw/tartan_data/abandonedfactory/Data/P000

result_dir=test_results/dpvo_test
rm -r ${result_dir}
mkdir ${result_dir}

python dpvo_pvgo.py \
--imagedir=${data_dir}/image_left \
--calib=dpvo_calib/tartan.txt \
--stride=1 \
--resultdir=${result_dir} \
--config=dpvo_config/default.yaml \
--network=models/dpvo.pth

