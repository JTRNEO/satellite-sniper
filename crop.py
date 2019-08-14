import cv2
import random
import numpy as np
import os
#import redis
#from getfile import get_image_size,get_stride,get_image_path,get_image_bs
import argparse
#path=get_image_path()
#stride=get_stride()
#image_size=get_image_size()
#batch_size=get_image_bs()
def parse_args():
    parser=argparse.ArgumentParser(description='inference')
    parser.add_argument('--dataset',type=str,required=True)

    parser.add_argument('--stride',required=True,type=int,default=1024)
    parser.add_argument('--image_size',required=True,type=int,default=1024)
    parser.add_argument('--batch_size',type=int,default=1)
    args=parser.parse_args()
    return args
def crop():
        args=parse_args()
       # r = redis.Redis(host='redis', port=6379, decode_responses=True)
        ###############################
       # b = redis.Redis(host='redis', port=6379, decode_responses=True)

       #####################################
        #path_name=os.path.split(args.image_name)
       ##########################################
        #os.makedirs('/mnt/geojson/'+path_name[1].split('.')[0])
       ##########################################
        #f=open('/mnt/path.txt','w')
       # f.write('./mnt/geojson/'+path_name[1].split('.')[0]+'/')
       # f.close()

        path=args.dataset
        image=cv2.imread(path)
        h,w,_=image.shape
        padding_h=(h//args.stride+1)*args.stride
        padding_w=(w//args.stride+1)*args.stride
        padding_img=np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:]=image[:,:,:]

        n=0
        image=np.asarray(padding_img,'f')
        for i in range(padding_h//args.stride):
           for j in range(padding_w//args.stride):
              crop=padding_img[i*args.stride:i*args.stride+args.image_size,j*args.stride:j*args.stride+args.image_size,:]
              n+=1
              cv2.imwrite('/sniper/data/demo_batch/images/'+str(n)+'.png',crop)
        print('finished!')
if __name__ == '__main__':

    crop()

