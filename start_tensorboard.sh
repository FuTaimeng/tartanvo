# to 200000 steps
cat start_tensorboard.sh
cd tensorboard

# tensorboard --logdir=./multicamvo_B32_St500_optuna_lr_new_archi_dev3090_Feb_25_2023_17_28_29

# tensorboard --logdir=./multicamvo_batch_64_step_50000_optuna_lr_dev3090_Feb_18_2023_03_21_51



# add testing dataset
# tensorboard --logdir=./multicamvo_B32_St2000_optuna_lr_testset_dev3090_Feb_27_2023_02_06_42
# debug
# tensorboard --logdir=./multicamvo_B32_St500_optuna_lr_testset_dev3090_Feb_27_2023_11_07_14
# debug 2
# tensorboard --logdir=./multicamvo_B32_St500_optuna_lr_testset_dev3090_Feb_27_2023_12_44_00
# debug 3
tensorboard --logdir=./multicamvo_B32_St2000_optuna_lr_testset_dev3090_Feb_27_2023_16_30_40