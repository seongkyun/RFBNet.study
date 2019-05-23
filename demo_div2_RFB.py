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
#from data import VOCroot
#from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, VOC_mobile_300
from data import *
import cv2
import torch.utils.data as data
from layers.functions import Detect,PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_mobile_c2',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO version')
parser.add_argument('-f', '--file', default=None, help='file to run demo')
parser.add_argument('-c', '--camera_num', default=0, type=int, 
                    help='demo camera number(default is 0)')
parser.add_argument('-m', '--trained_model', default=None,
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
    labels = VOC_CLASSES
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']
    from data.coco import COCO_CLASSES
    labels = COCO_CLASSES

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
COLORS = [(255, 0, 0), (153, 255, 0), (0, 0, 255), (102, 0, 0)] # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prior box setting
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def demo_img(net, detector, transform, img, save_dir):
    _t = {'inference': Timer(), 'misc': Timer()}
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if args.cuda:
            x = x.cuda()
            scale = scale.cuda()
    _t['inference'].tic()
    out = net(x)      # forward pass 
    boxes, scores = detector.forward(out,priors)
    inference_time = _t['inference'].toc()
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    _t['misc'].tic()
    for j in range(1, num_classes):
        max_ = max(scores[:, j])
        inds = np.where(scores[:, j] > args.threshold)[0]
        if inds is None:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = nms(c_dets, args.threshold, force_cpu=args.cpu)
        c_dets = c_dets[keep, :]
        c_bboxes=c_dets[:, :4]
        for bbox in c_bboxes:
            # Create a Rectangle patch
            label = labels[j-1]
            score = c_dets[0][4]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
            cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])), FONT, 1, COLORS[1], 2)
    nms_time = _t['misc'].toc()
    #status = ' inference time: {:.3f}s \n nms time: {:.3f}s \n FPS: {:d}'.format(inference_time, nms_time, int(1/(inference_time+nms_time)))
    status = 't_inf: {:.3f} s || t_misc: {:.3f} s  \r'.format(inference_time, nms_time)
    cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (0, 0, 0), 5)
    cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (255, 255, 255), 2)
    cv2.imwrite(save_dir, img)
    print(status)

def is_overlap_area(gt, box):
    #order: [start x, start y, end x, end y]
    if(gt[0]<=int(box[0]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[2]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[0]) and int(box[0])<=gt[2]):
        return True
    else:
        return False

def bigger_box(box_a, box_b):
    #order: [start x, start y, end x, end y]
    bigger_box = [min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    , max(box_a[2], box_b[2]), max(box_a[3], box_b[3])]
    return bigger_box

def is_same_obj(box, r_box, th):
    #order: [start x, start y, end x, end y]
    
    th_y = th // 3
    th_x = (th * 2) // 3

    r_mx = (r_box[0] + r_box[2]) // 2

    sy_dist = abs(r_box[1] - box[1])
    ey_dist = abs(r_box[3] - box[3])
    l_mx = (box[0] + box[2]) // 2
    if sy_dist<th_y and ey_dist<th_y:
        if abs(l_mx - r_mx) < th_x:
            return True
        else:
            return False
    else:
        return False

def get_close_obj(boxes, r_box, th):
    #order: [start x, start y, end x, end y, lable, score]

    # make the same object map
    obj_map = []
    new_obj = 0
    for j in range(len(boxes)):
        obj_map.append(is_same_obj(boxes[j], r_box, th))

    # change the existing object
    for j in range(len(obj_map)):
        new_obj += int(obj_map[j])
        if obj_map[j]:
            label = boxes[j][4]
            score = boxes[j][5]
            boxes[j] = bigger_box(r_box, boxes[j])
            boxes[j].append(label) # label
            boxes[j].append(score) # score

    # add the none existing obj
    if new_obj == 0:
        boxes.append(r_box)

    return None

def demo_stream(net, detector, transform, video, save_dir):
    _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}

    index = -1
    #avgFPS = 0.0

    width = int(video.get(3))
    height = int(video.get(4))

    half = width // 2
    over_area = 80

    scale = torch.Tensor([width, height, width, height])
    scale_half = torch.Tensor([half+over_area, height, half+over_area, height])

    middle_coords = [half-over_area, 0, half+over_area, height]
    l_middle_objs=[]

    while(video.isOpened()):
        _t['total'].tic()
        index = index + 1
        l_middle_objs=[]

        flag, img = video.read()
        
        img_l = img[:, :half+over_area]
        img_r = img[:, half-over_area:]

        with torch.no_grad():
            x_l = transform(img_l).unsqueeze(0)
            x_r = transform(img_r).unsqueeze(0)
            if args.cuda:
                x_l = x_l.cuda()
                x_r = x_r.cuda()
                scale_half = scale_half.cuda()
        _t['inference'].tic()
        out_l = net(x_l)      # forward pass
        out_r = net(x_r)      # forward pass
        inference_time = _t['inference'].toc()
        _t['misc'].tic()
        boxes_l, scores_l = detector.forward(out_l,priors)
        boxes_r, scores_r = detector.forward(out_r,priors)
        boxes_l = boxes_l[0]
        boxes_r = boxes_r[0]
        scores_l = scores_l[0]
        scores_r = scores_r[0]
        boxes_l *= scale_half
        boxes_r *= scale_half
        boxes_l = boxes_l.cpu().numpy()
        boxes_r = boxes_r.cpu().numpy()
        scores_l = scores_l.cpu().numpy()
        scores_r = scores_r.cpu().numpy()
        
        # left objects
        for j in range(1, num_classes):
            max_ = max(scores_l[:, j])
            inds = np.where(scores_l[:, j] > args.threshold)[0]
            if inds is None:
                continue
            c_bboxes = boxes_l[inds]
            c_scores = scores_l[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, args.threshold, force_cpu=False)
            c_dets = c_dets[keep, :]
            c_bboxes=c_dets[:, :4]
            for bbox in c_bboxes:
                # Create a Rectangle patch
                sx = int(bbox[0])
                sy = int(bbox[1])
                ex = int(bbox[2])
                ey = int(bbox[3])
                bbox = [sx, sy, ex, ey]
                if is_overlap_area(middle_coords, bbox):
                    bbox.append(j)
                    bbox.append(float(c_dets[0][4]))
                    l_middle_objs.append(bbox)
                else:
                    label = labels[j-1]
                    score = c_dets[0][4]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
                    cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])), FONT, 0.5, COLORS[1], 2)

        # right objects
        for j in range(1, num_classes):
            max_ = max(scores_r[:, j])
            inds = np.where(scores_r[:, j] > args.threshold)[0]
            if inds is None:
                continue
            c_bboxes = boxes_r[inds]
            c_scores = scores_r[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, args.threshold, force_cpu=False)
            c_dets = c_dets[keep, :]
            c_bboxes=c_dets[:, :4]
            for bbox in c_bboxes:
                # Create a Rectangle patch
                sx = int(half - over_area + bbox[0])
                sy = int(bbox[1])
                ex = int(half - over_area + bbox[2])
                ey = int(bbox[3])
                bbox = [sx, sy, ex, ey]
                if is_overlap_area(middle_coords, bbox):
                    bbox.append(j)
                    bbox.append(float(c_dets[0][4]))
                    get_close_obj(l_middle_objs, bbox, over_area)
                    continue
                else:
                    label = labels[j-1]
                    score = c_dets[0][4]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
                    cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])), FONT, 0.5, COLORS[1], 2)
        
        # middle objects
        for bbox in l_middle_objs:
            label = labels[bbox[4]-1]
            score = bbox[5]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
            cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])), FONT, 0.5, COLORS[1], 2)

        nms_time = _t['misc'].toc()
        total_time = _t['total'].toc()
        status = 'f_cnt: {:d} FPS_inf: {:.2f} FPS_tot: {:.2f} t_misc: {:.3f}s \r'.format(index, float(1/inference_time), float(1/total_time), nms_time)
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (0, 0, 0), 5)
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (255, 255, 255), 2)

        cv2.imshow('result', img)
        cv2.waitKey(33)
        
        cv2.imwrite(os.path.join(save_dir, 'frame_{}.jpg'.format(index)), img)

        sys.stdout.write(status)
        sys.stdout.flush()

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
    
    # Running demo
    print('Running demo...')
    if args.type == 'image':
        img = cv2.imread(args.file)
        demo_img(net, detector, transform, img, save_dir)
    elif args.type == 'video':
        video = cv2.VideoCapture(args.file)
        demo_stream(net, detector, transform, video, save_dir)
    elif args.type == 'camera':
        video = cv2.VideoCapture(args.camera_num)
        demo_stream(net, detector, transform, video, save_dir)
    else:
        raise AssertionError('ERROR::TYPE IS NOT CORRECT')