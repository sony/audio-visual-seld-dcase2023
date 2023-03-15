#!/bin/bash

for batch_size in 16; do
    CUDA_VISIBLE_DEVICES=0 python seld.py \
    -train -val \
    -b ${batch_size} \
    -s 1000 -i 10000;
done
