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
