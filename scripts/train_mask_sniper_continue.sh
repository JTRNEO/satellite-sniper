#!/bin/bash

CUDA_VISIBLE_DEVICES=10 python main_train.py --cfg /sniper/configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --set TRAIN.USE_NEG_CHIPS False TRAIN.RESUME True TRAIN.begin_epoch 3
