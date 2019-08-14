#!bin/bash


for i in /sniper/data/sniper_infer/*

do
var=$i
var=${var##*/}
txt="dum.txt"
if [ "$var" == "$txt" ];then
continue
else

  rm -rf /sniper/data/demo_batch/images
  mkdir /sniper/data/demo_batch/images

  rm -rf /sniper/data/demo_batch/batch_results
  mkdir /sniper/data/demo_batch/batch_results
  
  echo 'crop on'$i
  
  python /sniper/crop.py --dataset $i --image_size 1024 --stride 1024 
  
  echo 'begin inference'
  python demo_mask_batch.py --cfg configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --img_dir_path data/demo_batch/images --dataset satellite
  
  python /sniper/puzzle.py --dataset $i --image_size 1024 --stride 1024  
  echo 'done!'
fi
done

