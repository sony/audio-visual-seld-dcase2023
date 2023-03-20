#!/bin/bash

for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=0 python seld.py \
    -train -val \
    -twt ./data_dcase2023_task3/list_dataset/dcase2023t3_mic_devtrain_audiovisual.txt \
    -valwt ./data_dcase2023_task3/list_dataset/dcase2023t3_mic_devtest.txt \
    -b ${batch_size} \
    -s 1000 -i 20000;
done
