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
from data import VOCroot
from data import AnnotationTransform, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300
import cv2
import torch.utils.data as data
from layers.functions import Detect,PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-i', '--img', default=None, help='Image to demo')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('-t', '--threshold', default=0.4,
                    type=float, help='Detection confidence threshold value')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=True, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
    from data.voc0712 import VOC_CLASSES
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']
    from data.coco import COCO_CLASSES

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_custom':
    from models.RFB_Net_mobile_custom import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c_leaky':
    from models.RFB_Net_mobile_c_leaky import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
elif args.version == 'RFB_mobile_c_l_d':
    from models.RFB_Net_mobile_c_l_d import build_net
    cfg = (VOC_mobile_300, COCO_mobile_300)[args.dataset == 'COCO']
else:
    print('Unkown version!')

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def demo_img(net, detector, transform, save_dir):
    img = cv2.imread(args.img)
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if args.cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)      # forward pass
    boxes, scores = detector.forward(out,priors)
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    for j in range(1, num_classes):
        max_ = max(scores[:, j])
        inds = np.where(scores[:, j] > args.threshold)[0]      #conf > 0.6
        if inds is None:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        #keep = nms_py(c_dets, args.threshold)
        keep = nms(c_dets, args.threshold, force_cpu=args.cpu)
        c_dets = c_dets[keep, :]
        c_bboxes=c_dets[:, :4]
        #print('c_bboxes', c_bboxes)
        for bbox in c_bboxes:
            # Create a Rectangle patch
            label = COCO_CLASSES[j-1]
            score = c_dets[0][4]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
            cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])), FONT, 0.5, COLORS[1], 2)
            cv2.imwrite(save_dir, img)

if __name__ == '__main__':
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    rgb_means = ((103.94,116.78,123.68), (104, 117, 123))[args.version == 'RFB_vgg' or args.version == 'RFB_E_vgg']
    
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
    
    detector = Detect(num_classes,0,cfg)
    transform = BaseTransform(net.size, rgb_means, (2, 0, 1))
    
    demo_img(net, detector, transform, "./result.jpg")

