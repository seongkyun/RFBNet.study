from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils.timer import Timer
import sys

class ObjectDetector:
    def __init__(self, net, priorbox, priors, transform, detector, viz_arch=False):
        self.model = net
        self.priorbox = priorbox
        self.priors = priors
        self.transform = transform
        self.detector = detector

    def predict(self, img, threshold=0.6):
        # make sure the input channel is 3 
        assert img.shape[2] == 3
        #scale = torch.Tensor([img.shape[1::-1], img.shape[1::-1]])
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}
        
        # preprocess image
        _t['total'].tic()
        with torch.no_grad():
            #x = self.transform(img).unsqueeze(0)
            x = self.transform(img).unsqueeze(0).cuda() # for fastening
            #if args.cuda:
            #    x = x.cuda()
            #    scale = scale.cuda()

        # forward
        _t['inference'].tic()
        out = self.model(x)  # forward pass
        inference_time = _t['inference'].toc()

        # detect
        _t['misc'].tic()
        detections = self.detector.forward(out, self.priors)
        
        # output
        labels, scores, coords = [list() for _ in range(3)]
        batch=0
        for classes in range(detections.size(1)):
            num = 0
            while detections[batch,classes,num,0] >= threshold:
                #print(detections[batch,classes,num,0])
                #print(classes-1)
                #box = detections[batch,classes,num,1:]*scale
                #print(box)
                #sys.exit()
                scores.append(detections[batch,classes,num,0])
                labels.append(classes-1)
                coords.append(detections[batch,classes,num,1:]*scale)
                num+=1
        misc_time = _t['misc'].toc()
        total_time = _t['total'].toc()
        
        #if check_time is True:
        #    return labels, scores, coords, (total_time, inference_time, misc_time)
        #return labels, scores, coords
        return labels, scores, coords, (total_time, inference_time, misc_time)

def is_overlap_area(gt, box):
    #order: [start x, start y, end x, end y]
    if(gt[0]<=int(box[0]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[2]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[0]) and int(box[0])<=gt[2])\
    or (int(box[0])<=gt[0] and gt[2]<=int(box[2])):
        return True
    else:
        return False

def lable_selector(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    if box_a[5] > box_b[5]:
        lable = box_a[4]
        score = box_a[5]
    else:
        lable = box_b[4]
        score = box_b[5]
    return lable, score

def bigger_box(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    lable, score = lable_selector(box_a, box_b)
    bigger_box = [min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    , max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
    , lable, score]
    return bigger_box

def is_same_obj(box, r_box, th):
    #order: [start x, start y, end x, end y]
    th_y = th // 3
    th_x = (th * 2) // 3
    r_mx = (r_box[0] + r_box[2]) // 2
    sy_dist = abs(r_box[1] - box[1])
    ey_dist = abs(r_box[3] - box[3])
    l_mx = (box[0] + box[2]) // 2
    if sy_dist<th_y and ey_dist<th_y and r_box[4] == box[4]:
        if abs(l_mx - r_mx) < th_x:
            return True
        else:
            box_size = (box[2] - box[0]) * (box[3] - box[1])
            r_box_size = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
            th_size = th * th * 9
            th_th = int(th*0.2)
            if (box_size >= th_size) and (r_box_size >= th_size)\
            and (abs(box[2] - th*9)<th_th) and (abs(r_box[0] - th*7)<th_th):
                return True
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
            boxes[j] = bigger_box(r_box, boxes[j])
            break

    # add the none existing obj
    if new_obj == 0:
        boxes.append(r_box)

    return None

class ObjectDetector_div:
    def __init__(self, net, priorbox, priors, transform, detector, width, height):
        self.model = net
        self.priorbox = priorbox
        self.priors = priors
        self.transform = transform
        self.detector = detector
        #self.width = width
        #self.height = height
        self.half = width//2
        self.over_area = int(width*0.0625)
        self.scale = torch.Tensor([width, height, width, height])
        self.scale_half = torch.Tensor([self.half+self.over_area, height, self.half+self.over_area, height])
        self.middle_coords = [self.half-self.over_area, 0, self.half+self.over_area, height]

    def predict(self, img, threshold=0.6):
        # make sure the input channel is 3 
        assert img.shape[2] == 3

        _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}

        l_middle_objs = []
        
        img_l = img[:, :self.half+self.over_area]
        img_r = img[:, self.half-self.over_area:]
        
        # preprocess image
        _t['total'].tic()
        with torch.no_grad():
            x_l = self.transform(img_l).unsqueeze(0).cuda() # for fastening
            x_r = self.transform(img_r).unsqueeze(0).cuda() # for fastening
            #if args.cuda:
            #    x = x.cuda()
            #    scale = scale.cuda()

        # forward
        _t['inference'].tic()
        out_l = self.model(x_l)  # forward pass
        out_r = self.model(x_r)
        inference_time = _t['inference'].toc()

        # detect
        _t['misc'].tic()
        detections_l = self.detector.forward(out_l, self.priors)
        detections_r = self.detector.forward(out_r, self.priors)
        
        labels, scores, coords = [list() for _ in range(3)]
        
        # left objects
        batch=0
        for classes in range(detections_l.size(1)):
            num = 0
            while detections_l[batch,classes,num,0] >= threshold:
                
                t_bbox = detections_l[batch,classes,num,1:]*self.scale_half
                sx = int(t_bbox[0])
                sy = int(t_bbox[1])
                ex = int(t_bbox[2])
                ey = int(t_bbox[3])
                bbox = [sx, sy, ex, ey]

                if is_overlap_area(self.middle_coords, bbox):
                    bbox.append(classes-1)
                    bbox.append(detections_l[batch,classes,num,0])
                    l_middle_objs.append(bbox)
                else:
                    scores.append(detections_l[batch,classes,num,0])
                    labels.append(classes-1)
                    coords.append(detections_l[batch,classes,num,1:]*self.scale_half)
                num+=1
        
        # right objects
        batch=0
        for classes in range(detections_r.size(1)):
            num = 0
            while detections_r[batch,classes,num,0] >= threshold:
                t_bbox = detections_r[batch,classes,num,1:]*self.scale_half
                sx = int(self.half - self.over_area + t_bbox[0])
                sy = int(t_bbox[1])
                ex = int(self.half - self.over_area + t_bbox[2])
                ey = int(t_bbox[3])
                bbox = [sx, sy, ex, ey]
                if is_overlap_area(self.middle_coords, bbox):
                    bbox.append(classes-1)
                    bbox.append(detections_r[batch,classes,num,0])
                    get_close_obj(l_middle_objs, bbox, self.over_area)
                else:
                    scores.append(detections_r[batch,classes,num,0])
                    labels.append(classes-1)
                    coords.append(bbox)
                num+=1
        
        # middle objects
        for bbox in l_middle_objs:
            coords.append(bbox[:4])
            labels.append(bbox[4])
            scores.append(bbox[5])

        misc_time = _t['misc'].toc()
        total_time = _t['total'].toc()
        
        return labels, scores, coords, (total_time, inference_time, misc_time)