"""
Train CLSL with AlexNet

This code refers to CMC:https://github.com/HobbitLong/CMC/#contrastive-multiview-coding

Author: Shaochi Hu
"""
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

from torchvision import transforms, datasets
import torchvision

import tensorboard_logger as tb_logger

from dataset import ImageFolderInstance
from models.alexnet import MyAlexNetCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from util import adjust_learning_rate, AverageMeter,print_running_time


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print every print_freq batchs')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save model checkpoint every save_freq epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=4096) # negative sample number
    parser.add_argument('--nce_t', type=float, default=0.07) # temperature parameter
    parser.add_argument('--nce_m', type=float, default=0.5) # memory update rate
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product') # dimension of network's output

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.8, help='low area in crop')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'

    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    print('start program at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    opt.model_name = '{}_lossMethod_{}_NegNum_{}_Model_{}_lr_{}_decay_{}_bsz_{}'.format(curTime, opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size)

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    data_folder = os.path.join(args.data_folder, 'train')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, num_workers=args.num_workers, pin_memory=True)

    # 获得某类物体的图片索引，图片索引以set的形式保存
    classInstansSet = []
    tmpSet = set()
    last_t = 0
    for i,t in enumerate(train_dataset.targets):
        if last_t != t:
            classInstansSet.append(tmpSet.copy())
            tmpSet.clear()
        last_t = t
        tmpSet.add(i)
    classInstansSet.append(tmpSet)

    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data, classInstansSet

def set_model(args, n_data, classInstansSet):

    model = MyAlexNetCMC(args.feat_dim)
    contrast = NCEAverage(args.feat_dim, n_data,classInstansSet,args.nce_k, args.nce_t, args.nce_m, args.softmax)

    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion

def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def train(epoch,train_loader, model, contrast, criterion, optimizer, opt):
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    probs = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = img.size(0)
        img = img.float()
        if torch.cuda.is_available():
            index = index.cuda()
            img = img.cuda()

        # ===================forward=====================
        feat = model(img)
        mutualInfo = contrast(feat, target, index)
        loss = criterion(mutualInfo)
        prob = mutualInfo[:,0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        probs.update(prob.item(), bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'p {probs.val:.3f} ({probs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, probs=probs,))
            sys.stdout.flush()

    return losses.avg, probs.avg


def main():
    # parse the args
    args = parse_option()
    args.start_epoch = 1

    # set the loader
    train_loader, n_data, classInstansSet= get_train_loader(args)

    # set the model
    model, contrast, criterion = set_model(args,n_data,classInstansSet)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # train by epoch
    print('start training at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)

        loss, prob = train(epoch, train_loader, model, contrast, criterion, optimizer, args)

        print_running_time(start_time)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # if args.amp:
            #     state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state


if __name__ == '__main__':
    main()


