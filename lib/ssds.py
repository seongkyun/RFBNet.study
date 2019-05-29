from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils.timer import Timer

class ObjectDetector:
    def __init__(self, net, priorbox, priors, transform, detector, viz_arch=False):
        self.model = net
        self.priorbox = priorbox
        self.priors = priors
        self.transform = transform
        self.detector = detector

    def predict(self, img, threshold=0.6, check_time=False):
        # make sure the input channel is 3 
        assert img.shape[2] == 3
        #scale = torch.Tensor([img.shape[1::-1], img.shape[1::-1]])
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        _t = {'preprocess': Timer(), 'net_forward': Timer(), 'detect': Timer(), 'output': Timer()}
        
        # preprocess image
        _t['preprocess'].tic()
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).cuda()
            #if args.cuda:
            #    x = x.cuda()
            #    scale = scale.cuda()
        #if self.use_gpu:
        #    x = x.cuda()
        #if self.half:
        #    x = x.half()
        preprocess_time = _t['preprocess'].toc()

        # forward
        _t['net_forward'].tic()
        out = self.model(x)  # forward pass
        net_forward_time = _t['net_forward'].toc()

        # detect
        _t['detect'].tic()
        detections = self.detector.forward(out, self.priors)
        detect_time = _t['detect'].toc()
        
        # output
        _t['output'].tic()
        labels, scores, coords = [list() for _ in range(3)]
        batch=0
        for classes in range(detections.size(1)):
            num = 0
            while detections[batch,classes,num,0] >= threshold:
                scores.append(detections[batch,classes,num,0])
                labels.append(classes-1)
                coords.append(detections[batch,classes,num,1:]*scale)
                num+=1
        output_time = _t['output'].toc()
        total_time = preprocess_time + net_forward_time + detect_time + output_time
        
        if check_time is True:
            return labels, scores, coords, (total_time, preprocess_time, net_forward_time, detect_time, output_time)
            # total_time = preprocess_time + net_forward_time + detect_time + output_time
            # print('total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
            #     total_time, preprocess_time, net_forward_time, detect_time, output_time
            # ))
        return labels, scores, coords