# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
import sys
sys.path.insert(0, './lib')
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
# from inference import Tester
from inference_mask import Tester
from symbols.faster import *
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import ipdb
import time
def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--img_dir_path', dest='img_dir_path', help='Path to the image directory', type=str,
                            default='data/demo_batch/images')
    arg_parser.add_argument('--dataset', dest='dataset', help='choose categories belong to which dataset', type=str,
                            default='satellite')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    arg_parser.add_argument('--batch_size', dest='batch_size', help='Batch Size',default=1)
    
    return arg_parser.parse_args()

coco_names = [u'BG', u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                       u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant',
                       u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                       u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
                       u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports\nball', u'kite', u'baseball\nbat',
                       u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine\nglass',
                       u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange',
                       u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
                       u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote',
                       u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book',
                       u'clock', u'vase', u'scissors', u'teddy bear', u'hair\ndrier', u'toothbrush']

dota_names = [u'BG', u'plane', u'baseball-diamond', u'bridge', u'ground-track-field', u'small-vehicle', u'large-vehicle', u'ship', u'tennis-court',
               u'basketball-court', u'storage-tank',  u'soccer-ball-field', u'roundabout', u'harbor', u'swimming-pool', u'helicopter']

satellite_names = [u'BG', u'Planes', u'Ships', u'Helicopter', u'Vehicles', u'Bridges', u'Buildings',
                       u'Parking Lots', u'Satellite Dish', u'Solar Panels', u'Storage Tank', u'Swimming Pool',
                       u'Sports Stadium/Field', u'Shipping Containers', u'Crane', u'Train', u'Mil Vehicles',
                       u'Missiles/Missile Systems', u'Comms Towers']

#satellite_names = [u'BG', u'Buildings']

def main():
    
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    # Use just the first GPU for demo
    context = [mx.gpu(int(config.gpus[0]))]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Pack db info
    db_info = EasyDict()
    db_info.name = 'coco'
    db_info.result_path = 'data/demo/batch_results'

    assert args.dataset in ['coco', 'dota', 'satellite']
    if args.dataset == 'coco':
        db_info.classes = coco_names
    elif args.dataset == 'dota':
        db_info.classes = dota_names
    elif args.dataset == 'satellite':
        db_info.classes = satellite_names

    db_info.num_classes = len(db_info.classes)

    #roidb = []

    for img in os.listdir(args.img_dir_path):

        start = time.time()

        im_path = os.path.join(args.img_dir_path,img)

        # Get image dimensions
        width, height = Image.open(im_path).size

        # Pack image info
        roidb = [{'image': im_path, 'width': width, 'height': height, 'flipped': False}]
        #r = [{'image': im_path, 'width': width, 'height': height, 'flipped': False}]
        #roidb.append(r)

        # Creating the Logger
        logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

        
        # Create the model
        sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
        sym_inst = sym_def(n_proposals=400, test_nbatch=1)
        sym = sym_inst.get_symbol_rcnn(config, is_train=False)
        test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=args.batch_size, nGPUs=1, threads=1,
                                   crop_size=None, test_scale=config.TEST.SCALES[0],
                                   num_classes=db_info.num_classes)
        # Create the module
        shape_dict = dict(test_iter.provide_data_single)
        sym_inst.infer_shape(shape_dict)
        mod = mx.mod.Module(symbol=sym,
                            context=context,
                            data_names=[k[0] for k in test_iter.provide_data_single],
                            label_names=None)
        # TODO: just to test the change of order, for refactor
        mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

        # Initialize the weights
        model_prefix = os.path.join(output_path, args.save_prefix)
        arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                            convert=True, process=True)
        mod.init_params(arg_params=arg_params, aux_params=aux_params)

        # Create the tester
        tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=args.batch_size)

        # Sequentially do detection over scales
        # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
        all_detections= []
        all_masks = []
        for s in config.TEST.SCALES:
            # Set tester scale
            tester.set_scale(s)
            # Perform detection
            detections, masks = tester.get_detections(vis=False, evaluate=False, cache_name=None)
            all_detections.append(detections)
            all_masks.append(masks)
            # all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

        # Aggregate results from multiple scales and perform NMS
        tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=args.batch_size)
        file_name, out_extension = os.path.splitext(os.path.basename(im_path))
        all_detections, all_masks = tester.aggregateSingle(all_detections, all_masks, vis=True, cache_name=None, vis_path='./data/demo_batch/batch_results',
                                              vis_name='{}_detections'.format(file_name), vis_ext=out_extension)
        # all_detections, all_masks = tester.aggregate(all_detections, all_masks, vis=True, cache_name=None, vis_path='./data/demo/',
        #                                       vis_name='{}_detections'.format(file_name), vis_ext=out_extension)
        #return all_detections, all_masks
        end = time.time()
        print("Time to process {} :{}".format(img,end-start)) 

if __name__ == '__main__':
    main()
