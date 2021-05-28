from __future__ import print_function

import torch
import numpy as np
import time
import sys
import os

def print_running_time(start_time):
    print()
    print('='*20,end = ' ')
    print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
    using_time = time.time()-start_time
    hours = int(using_time/3600)
    using_time -= hours*3600
    minutes = int(using_time/60)
    using_time -= minutes*60
    print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)),end = ' ')
    print('='*20)
    print()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object): # 旨在把程序中所有print出来的内容都保存到文件中
    def __init__(self, filename="Default.log"):
        path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(path,filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

def get_anchor_pos_neg(supplement_pos_neg_txt_path, dataset, classInstansSet):
    pos_neg_idx = [] # 数据形式：[[[pos_idx], [neg_idx]], [[pos_idx], [neg_idx]], [[pos_idx], [neg_idx]]]
    anchor_num = len(dataset.classes)

    # 首先把原来由邻域定义的正负样本放进去
    for i in range(anchor_num):
        cur_pos = list(classInstansSet[i])
        cur_neg = list(classInstansSet[(i+1) % anchor_num]) + list(classInstansSet[i-1])
        pos_neg_idx.append([cur_pos, cur_neg])
    
    # 读取补充正负样本的文件
    f = open(supplement_pos_neg_txt_path, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line[0] == '#':
            continue
        line = line.split(':')

        anchor_img_idx_str = line[0].rjust(10,'0')
        anchor_torch_idx = dataset.class_to_idx[anchor_img_idx_str]

        # 补充的正样本锚点在pytorch中的索引
        pos_anchor_img_idx_str = line[1].split()
        pos_anchor_torch_idx = []
        if len(pos_anchor_img_idx_str):
            pos_anchor_img_idx_str = [i.rjust(10,'0') for i in pos_anchor_img_idx_str]
            pos_anchor_torch_idx = [dataset.class_to_idx[i] for i in pos_anchor_img_idx_str]

        # 补充的负样本锚点在pytorch中的索引
        neg_anchor_img_idx_str = line[2].split()
        neg_anchor_torch_idx = []
        if len(neg_anchor_img_idx_str):
            neg_anchor_img_idx_str = [i.rjust(10,'0') for i in neg_anchor_img_idx_str]
            neg_anchor_torch_idx = [dataset.class_to_idx[i] for i in neg_anchor_img_idx_str]
        
        pos_sample_torch_idx = []
        for i in pos_anchor_torch_idx:
            pos_sample_torch_idx += classInstansSet[i]
        
        neg_sample_torch_idx = []
        for i in neg_anchor_torch_idx:
           neg_sample_torch_idx += classInstansSet[i]

        pos_neg_idx[anchor_torch_idx][0] += pos_sample_torch_idx
        pos_neg_idx[anchor_torch_idx][0] = list(set(pos_neg_idx[anchor_torch_idx][0]))

        pos_neg_idx[anchor_torch_idx][1] += neg_sample_torch_idx
        pos_neg_idx[anchor_torch_idx][1] = list(set(pos_neg_idx[anchor_torch_idx][1]))

        pass
    
    return pos_neg_idx



def check_pytorch_idx_validation(class_to_idx):
    keys = []
    values = []
    for key in class_to_idx.keys():
        keys.append(int(key))
        values.append(class_to_idx[key])
    origin_keys = keys.copy()
    origin_values = values.copy()
    keys.sort()
    values.sort()
    for i, key in enumerate(keys):
        cur_value = values[i]
        idx = origin_keys.index(key)
        origin_value = origin_values[idx]
        if cur_value != origin_value:
            print('Error! Pytorch file name sort error! Program exit.')
            exit()

    
if __name__ == '__main__':
    meter = AverageMeter()
