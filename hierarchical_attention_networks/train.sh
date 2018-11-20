#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python config.py \
    --mode train \
    --gpu 3 \
    --para_max_num 5 \
    --para_max_length 10 \
    --batch_size 1024 \
    --init_lr 0.1 \
    --patience 5 \
    --num_steps 1 \
    --checkpoint 1
