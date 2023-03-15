#!/bin/bash

params=(
    "./data_dcase2023_task3/model_monitor/202303xxxxxxxx/params_202303xxxxxxxx_0010000.pth"
)

for param in "${params[@]}"; do
    for batch_size in 16; do
        p=(${param});
        CUDA_VISIBLE_DEVICES=0 python seld.py \
        -eval \
        -evalwt ./data_dcase2023_task3/list_dataset/dcase2023t3_foa_devtest.txt \
        -em ${p[0]} \
        -b ${batch_size};
    done
done
