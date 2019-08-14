#!bin/bash
echo 'begin'
rm -rf /sniper/data/Patches
mkdir /sniper/data/Patches

rm -rf /sniper/data/result
mkdir /sniper/data/result
python Clip.py

for i in /sniper/data/Patches/*

do
  rm -rf /sniper/data/demo_batch/images
  mkdir /sniper/data/demo_batch/images

  rm -rf /sniper/data/demo_batch/batch_results
  mkdir /sniper/data/demo_batch/batch_results
  
  echo 'crop on'$i
  
  python /sniper/crop.py --dataset $i --image_size 1024 --stride 1024 
  
  echo 'begin inference'
  python demo_mask_batch.py --cfg configs/faster/sniper_res101_e2e_mask_pred_satellite.yml --img_dir_path data/demo_batch/images --dataset satellite
  
  python /sniper/puzzle_Aden.py --dataset $i --image_size 1024 --stride 1024  
  echo 'Patches done'

done

python Merge.py
rm -rf /sniper/data/Patches
mkdir /sniper/data/Patches

rm -rf /sniper/data/result
mkdir /sniper/data/result

rm -rf /sniper/data/demo_batch/images
mkdir /sniper/data/demo_batch/images

rm -rf /sniper/data/demo_batch/batch_results
mkdir /sniper/data/demo_batch/batch_results

echo 'finished'
