from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import *
import cv2
import torch.utils.data as data
from layers.functions import Detect_test,PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer
from lib.detector import ObjectDetector

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-f', '--file', default=None, help='file to run demo')
parser.add_argument('-c', '--camera_num', default=0, type=int, 
                    help='demo camera number(default is 0)')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='results/', type=str,
                    help='Dir to save results')
parser.add_argument('-th', '--threshold', default=0.45,
                    type=float, help='Detection confidence threshold value')
parser.add_argument('-t', '--type', dest='type', default='image', type=str,
            help='the type of the demo file, could be "image", "video", "camera"')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=True, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()

# Make result file saving folder
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# Label settings
if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
    from data.voc0712 import VOC_CLASSES
    lable_map = VOC_CLASSES
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']
    from data.coco import COCO_CLASSES
    lable_map = COCO_CLASSES

# Version checking
if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c1':
    from models.RFB_Net_mobile_c1 import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c2':
    from models.RFB_Net_mobile_c2 import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
    cfg = (VOC_SSDVGG_300, COCO_SSDVGG_300)[args.dataset == 'COCO']
elif args.version == 'SSD_lite_mobile_v1':
    from models.SSD_lite_mobilenet_v1 import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'SSD_lite_mobile_v2':
    from models.SSD_lite_mobilenet_v2 import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c_leaky':
    print('WARNING::TESTING METHOD')
    from models.RFB_Net_mobile_c_leaky import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c_l_d':
    print('WARNING::TESTING METHOD')
    from models.RFB_Net_mobile_c_l_d import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']

else:
    print('ERROR::UNKNOWN VERSION')
    sys.exit()

# color number book: http://www.n2n.pe.kr/lev-1/color.htm
COLORS = [(255, 0, 0), (153, 255, 0), (0, 0, 255), (102, 0, 0), (153, 102, 51)] # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prior box setting
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def demo_img(object_detector, img, save_dir):
    
    #labels, scores, coords, (total_time, inference_time, misc_time)
    _labels, _scores, _coords, times= object_detector.predict(img)
    FPS = float(1/times[0])
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[1], 2)
        cv2.putText(img, '{label}: {score:.2f}'.format(label=lable_map[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[1], 2)
    
    status = 'FPS_tot: {:.2f} t_inf: {:.2f} t_misc: {:.3f}s \r'.format(FPS, times[1], times[2])
    cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, COLORS[4], 2)
       
    cv2.imwrite(save_dir, img)


def demo_stream(object_detector, video, save_dir):
    index = -1

    FPS = 0.0

    width = int(video.get(3))
    height = int(video.get(4))
    video_dir = os.path.join(save_dir, 'result.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(video_dir, fourcc, 25.0, (width,height))

    while(video.isOpened()):
        index = index + 1
        
        flag, img = video.read()
        if flag == False:
            break
        #labels, scores, coords, (total_time, inference_time, misc_time)
        _labels, _scores, _coords, times= object_detector.predict(img)
        
        FPS = float(1/times[0])    

        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[1], 2)
            cv2.putText(img, '{label}: {score:.2f}'.format(label=lable_map[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[1], 2)
    
        status = 'f_cnt: {:d} FPS_tot: {:.2f} t_inf: {:.2f} t_misc: {:.3f}s \r'.format(index, FPS, times[1], times[2])
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, COLORS[4], 2)

        cv2.imwrite(os.path.join(save_dir, 'frame_{}.jpg'.format(index)), img)
        video_out.write(img)

        sys.stdout.write(status)
        sys.stdout.flush()
    
    video.release()
    video_out.release()
    cv2.destroyAllWindows()   
    
if __name__ == '__main__':
    # Validity check
    print('Validity check...')
    if not args.type == 'camera':
        assert os.path.isfile(args.file), 'ERROR::DEMO FILE DOES NOT EXIST'
    assert os.path.isfile(args.trained_model), 'ERROR::WEIGHT FILE DOES NOT EXIST'

    # Directory setting
    print('Directory setting...')
    if args.type == 'image':
        path, _ = os.path.splitext(args.file)
        filename = args.version + '_' + path.split('/')[-1]
        save_dir = os.path.join(args.save_folder, filename + '.jpg')
    elif args.type == 'video':
        path, _ = os.path.splitext(args.file)
        filename = args.version + '_' + path.split('/')[-1]
        save_dir = os.path.join(args.save_folder, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    elif args.type == 'camera':
        filename = args.version + '_camera_' + str(args.camera_num)
        save_dir = os.path.join(args.save_folder, filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        raise AssertionError('ERROR::TYPE IS NOT CORRECT')

    # Setting network
    print('Network setting...')
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    rgb_means = ((103.94,116.78,123.68), (104, 117, 123))[args.version == 'RFB_vgg' or args.version == 'RFB_E_vgg']
    p = (0.2, 0.6)[args.version == 'RFB_vgg' or args.version == 'RFB_E_vgg']
    print('Loading pretrained model')
    net = build_net('test', 300, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    print('Finished loading model')
    
    detector = Detect_test(num_classes, 0, cfg, 0.6, args.threshold, 100, priors)
    transform = BaseTransform(net.size, rgb_means, (2, 0, 1))
    object_detector = ObjectDetector(net, priorbox, priors, transform, detector)

    # Running demo
    print('Running demo...')
    if args.type == 'image':
        img = cv2.imread(args.file)
        demo_img(object_detector, img, save_dir)
    elif args.type == 'video':
        video = cv2.VideoCapture(args.file)
        demo_stream(object_detector, video, save_dir)
    elif args.type == 'camera':
        video = cv2.VideoCapture(args.camera_num)
        demo_stream(object_detector, video, save_dir)
    else:
        raise AssertionError('ERROR::TYPE IS NOT CORRECT')

