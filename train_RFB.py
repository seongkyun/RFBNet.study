from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, VOC_mobile_300, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time

global glb_feature_teacher
global glb_feature_student
def get_features4teacher(self, input, output):
    global glb_feature_teacher
    glb_feature_teacher = output
    return None

def get_features4student(self, input, output):
    global glb_feature_student
    glb_feature_student = output
    return None

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_mobile',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/mobilenet_feature.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('-m', '--teacher_net', default='./pretrained/RFBNet300_VOC_80_7.pth',
                    type=str, help='Teacher state_dict file path to open')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = VOC_mobile_300
else:
    print('Unkown version!')

img_dim = (300,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

net = build_net('train', img_dim, num_classes)
print('=============student_net============')
print(net)

#========================================================
#==========Load teacher net and define layers============
#========================================================
cfg_t = (VOC_300, VOC_512)[args.size == '512']
from models.RFB_Net_vgg import build_net
net_t = build_net('train', img_dim, num_classes)
print('=============teacher_net============')
print(net_t)
print('Loading teacher network...')
state_dict = torch.load(args.teacher_net)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net_t.load_state_dict(new_state_dict)

for param in net.parameters():
    param.requires_grad=True
for param in net_t.parameters():
    param.requires_grad=False

t_loc_layers = net_t.loc
s_loc_layers = net.loc

t_conf_layers = net_t.conf
s_conf_layers = net.conf

t_loc_layers_sizes = [(batch_size, 24, 38, 38)
                    , (batch_size, 24, 19, 19)
                    , (batch_size, 24, 10, 10)
                    , (batch_size, 24, 5, 5)
                    , (batch_size, 16, 3, 3)
                    , (batch_size, 16, 1, 1)]
s_loc_layers_sizes = [(batch_size, 24, 19, 19)
                    , (batch_size, 24, 10, 10)
                    , (batch_size, 24, 5, 5)
                    , (batch_size, 24, 3, 3)
                    , (batch_size, 16, 2, 2)
                    , (batch_size, 16, 1, 1)]

t_conf_layers_sizes = [(batch_size, 126, 38, 38)
                    , (batch_size, 126, 19, 19)
                    , (batch_size, 126, 10, 10)
                    , (batch_size, 126, 5, 5)
                    , (batch_size, 84, 3, 3)
                    , (batch_size, 84, 1, 1)]
s_conf_layers_sizes = [(batch_size, 126, 19, 19)
                    , (batch_size, 126, 10, 10)
                    , (batch_size, 126, 5, 5)
                    , (batch_size, 126, 3, 3)
                    , (batch_size, 84, 2, 2)
                    , (batch_size, 84, 1, 1)]

glb_feature_teacher = torch.tensor(torch.zeros(t_loc_layers_sizes[0]), requires_grad=False)
glb_feature_student = torch.tensor(torch.zeros(s_loc_layers_sizes[0]), requires_grad=True)
t_loc_layers[0].register_forward_hook(get_features4teacher)
s_loc_layers[0].register_forward_hook(get_features4student)

criterion_mse = nn.MSELoss()

print('done')
#sys.exit(0)
#========================================================
#========================================================

if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
# load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    #from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net_t = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    net_t.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()



def train():
    global glb_feature_teacher
    global glb_feature_student
    net.train()
    net_t.eval()

    #import sys
    #from torchsummary import summary
    #summary(net, input_size=(3, 300, 300))
    #sys.exit()

    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    guid_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr

    m = torch.nn.MaxPool2d(2,2,0)# Just for last layers

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            guid_loss = 0
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        # load train data
        images, targets = next(batch_iterator)
        
        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        #==============================================
        out_t = net(images)
        emb_teacher = torch.tensor(glb_feature_teacher, requires_grad=False, device=torch.device('cuda'))
        emb_student = torch.tensor(glb_feature_student, requires_grad=True, device=torch.device('cuda'))
        #if iteration%20==0:
        #    print(emb_teacher.size())
        #    print(emb_student.size())
        #    print(emb_teacher)
        #    print(emb_student)
        #==============================================
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)

        #==============================================
        loss_g = criterion_mse(emb_student, m(emb_teacher))
        #==============================================
        loss = loss_l + loss_c + loss_g
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        guid_loss += loss_g.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(),loss_c.item()) + 
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            print('Guide loss: %0.4f'%loss_g.item())

    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
