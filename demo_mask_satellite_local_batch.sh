export LD_LIBRARY_PATH=/home/gxdai/nvidia/cuda-9.0/lib64:/home/gxdai/nvidia/cudnn74/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/gxdai/nvidia/cuda-9.0/lib64/stubs:$LD_LIBRARY_PATH
python demo_mask_batch.py --cfg configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --img_dir_path data/demo_batch/images --dataset satellite
