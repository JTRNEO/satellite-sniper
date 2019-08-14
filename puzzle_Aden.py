import time
import cv2
import os
import numpy as np
import argparse
#import getfile 
# Uncomment next two lines if you do not have Kube-DNS working. 
# import os 
# host = os.getenv("REDIS_SERVICE_HOST") 

#path=getfile.get_image_path() 
#stride=getfile.get_image_size() 
#image_size=getfile.get_image_size() 
def parse_args():
    parser=argparse.ArgumentParser(description='puzzledeeplab')
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--stride',required=True,type=int,default=1024)
    parser.add_argument('--image_size',required=True,type=int,default=1024)
    args=parser.parse_args()
    return args


def puzzle():
   args=parse_args()
   path=args.dataset
   n=0

   image=cv2.imread(path)
   prefix=os.path.split(path)
   h,w,_=image.shape
   padding_h=(h//args.stride+1)*args.stride
   padding_w=(w//args.stride+1)*args.stride
   mask_whole = np.zeros((padding_h,padding_w,3),dtype=np.uint8)

   for i in range(padding_h//args.stride):
        for j in range(padding_w//args.stride):
               n+=1
               
               img_dir= '/sniper/data/demo_batch/batch_results/'+str(n)+'_detections'+'.png'
               if os.path.isfile(img_dir):  
                 mask=cv2.imread(img_dir) 
              
                 mask=mask.reshape((args.image_size,args.image_size,3)).astype(np.uint8) 
                 mask_whole[i*args.stride:i*args.stride+args.image_size,j*args.stride:j*args.stride+args.image_size,:] = mask[:,:,:] 
               else: 
                 mask=cv2.imread('/sniper/data/demo_batch/images/'+str(n)+'.png') 
              
                 mask=mask.reshape((args.image_size,args.image_size,3)).astype(np.uint8) 
                 mask_whole[i*args.stride:i*args.stride+args.image_size,j*args.stride:j*args.stride+args.image_size,:] = mask[:,:,:] 
                
               

   cv2.imwrite('/sniper/data/result/'+prefix[-1].split('.')[0]+'.tif',mask_whole[0:h,0:w,:])
if __name__ == '__main__':
    puzzle()

