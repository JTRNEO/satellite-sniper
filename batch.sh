#!/bin/bash

root_dir=/home/gxdai/satellite/satellite/002_sliced_image

for file in $root_dir/*
do
    echo $file
    python demo_mask.py --cfg configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --im_path $file --dataset satellite
done