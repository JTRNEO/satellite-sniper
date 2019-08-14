from tqdm import tqdm
import copy
import pickle
import os
import importlib
import math
from pydoc import locate

import rediswq
import cv2
import sys

import re
import time
import os
os.chdir("/sniper/service")
sys.path.append('/sniper/SNIPER-mxnet/')

from GeoImage import GeoImage
from ContourProcessor import ContourProcessor
from config import serviceConfig, Cache

import pdb




def run_process(serviceConfig, outputDir, Model):

    if os.path.isdir(outputDir) is False:
        os.makedirs(outputDir)
    
    cache_path = os.path.join(outputDir,'full_results.pkl')

    geoImg = GeoImage(serviceConfig.image_path, gap=200)
    
    px, py = geoImg.pixelSize # tuple(px,py)
    pixel_size = (px,py)
    
    model = None if serviceConfig.cacheStrategy == Cache.read else Model(serviceConfig.modelStorePath)

    geojson = ContourProcessor(geoImg, outputDir)

    readCacheIterator = None
    if serviceConfig.cacheStrategy == Cache.read:
        with open(cache_path, 'rb') as input:
            cachedReadResults = pickle.load(input)
        readCacheIterator = iter(cachedReadResults)


    cacheWriteResults = []

    #pdb.set_trace()

    for xyOffset in tqdm(geoImg.getSplits(), desc='Processing {}'.format('Model Cache' if serviceConfig.cacheStrategy == Cache.read else 'Image')):

        left, up = xyOffset

        if serviceConfig.cacheStrategy == Cache.read:
            result = next(readCacheIterator)
        else:
            img = geoImg.getCv2ImgFromSplit(xyOffset)
            result = model.infere(img, imageId='left-{}_up-{}'.format(left, up), pixel_size=pixel_size)
            if serviceConfig.cacheStrategy == Cache.write: cacheWriteResults.append(copy.deepcopy(result))

        patchGeom = geojson.addPatchBoundary(left, up)
        for feature in result:
            #geojson.addFeature(left, up, feature)
            geojson.addFeature(left, up, feature, patchGeom)
            # print("Feature added")

    if serviceConfig.cacheStrategy == Cache.write:
        with open(cache_path, 'wb') as output:
            pickle.dump(cacheWriteResults, output, pickle.HIGHEST_PROTOCOL)


    geojson.cleanUp()


    # do comparison with the ground truth if it is given
    if serviceConfig.groundTruthGeoJson != None:
        geojson.compareGT(serviceConfig.groundTruthGeoJson, serviceConfig.gtMappings, serviceConfig.modelMappings)


def main():
    
    # Import your model
    #################################################
    """
    A general way of importing the model, so instead of this:
    >> from service.MaskRCNN import MaskRCNN
    >> from service.PANet import PANet 

    You can just specify the model name in config.py and it's
    imported automatically using the model name.

    """
    print("Importing SNIPER Model...")
    from Sniper_v2 import SNIPER as Model
    print("Import Complete") 
    #if serviceConfig.cacheStrategy == Cache.write:
    #    MODEL_NAME = serviceConfig.modelName
    #    modulename = "models." + MODEL_NAME + "." + MODEL_NAME
    #    # mod = __import__(modulename, fromlist=[MODEL_NAME])
    #    mod = locate(modulename)
    #    Model = getattr(mod, MODEL_NAME)
    #    #################################################

    #    print("MODEL:",MODEL_NAME)
    #else:
    #    Model = None

    img_ext = [".tif",".TIF",".tiff",".TIFF"]

    print("DIR:{}".format(os.path.basename(os.path.normpath(serviceConfig.img_dir_path))))

    # Assert the image directory exists
    assert os.path.isdir(serviceConfig.img_dir_path) is True
    img_list = [i for i in os.listdir(serviceConfig.img_dir_path) if i[-4:] in img_ext]

    outputListDir = os.path.join(os.path.dirname(serviceConfig.img_dir_path),os.path.basename(serviceConfig.img_dir_path), '{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion))#+ "_Results") # 1_Results ##os.path.join(os.path.dirname(serviceConfig.img_dir_path),'{}/{}/'.format(serviceConfig.modelName, serviceConfig.modelVersion))

    if os.path.isdir(outputListDir) is False:
        os.makedirs(outputListDir)

    print("Number of images in directory = {} \n".format(len(img_list)))

    for img in img_list:

        # Overwrite image path in config.py
        serviceConfig.image_path = os.path.join(serviceConfig.img_dir_path,img)

        print("IMAGE:{}".format(os.path.basename(os.path.normpath(serviceConfig.image_path))))

        img_output_dir = os.path.join(outputListDir,os.path.basename(serviceConfig.image_path)[:-4])

        # Create an outputDir for each image in img_list
        if os.path.isdir(img_output_dir) is False:
            os.makedirs(img_output_dir)

        run_process(serviceConfig,img_output_dir,Model)

        print("\n")



if __name__ == "__main__":
    main()

