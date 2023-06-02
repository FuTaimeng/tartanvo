#!/bin/bash

# dates=("2011_10_03" )
# drives=("42")

dates=("2011_10_03" "2011_10_03" "2011_10_03" "2011_09_30" "2011_09_30" "2011_09_30" "2011_09_30" "2011_09_30" "2011_09_30" "2011_09_30")
drives=("27" "42" "34" "16" "18" "20" "27" "28" "33" "34")

# dates=("2011_10_03" "2011_09_30")
# drives=("27" "27")

# ds_date=2011_10_03
# ds_idx=42 


for i in ${!dates[*]}; do
  echo "${dates[$i]} ${drives[$i]}"
done


# Path to the file to be modified
file_path="./train_vo_pvgo_kitti.script"


for i in ${!dates[*]}; do
    ds_date=${dates[$i]}
    ds_idx=${drives[$i]}

    # # Break the loop if ds_name is equal to "MH_03_medium"
    # if [[ "$i" == "1" ]]; then
    #     break
    # fi

    # Replace the placeholders with actual values in the file. The changes are saved into a new file.
    sed -i "s/ds_date=DATE_NAME/ds_date=$ds_date/g" "$file_path"
    sed -i "s/ds_idx=IDX_NAME/ds_idx=$ds_idx/g" "$file_path"

    sbatch train_vo_pvgo_kitti.script

    sed_command="s/ds_date=$ds_date/ds_date=DATE_NAME/g"
    sed -i "$sed_command" "$file_path"

    sed_command="s/ds_idx=$ds_idx/ds_idx=IDX_NAME/g"
    sed -i "$sed_command" "$file_path"

done