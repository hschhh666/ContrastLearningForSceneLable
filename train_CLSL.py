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

import numpy as np

from torchvision import transforms, datasets
import torchvision

import tensorboard_logger as tb_logger

from dataset import ImageFolderInstance, ImageFolderInstance_LoadAllImgToMemory
from models.alexnet import MyAlexNetCMC
from NCE.NCEAverage import NCEAverage, E2EAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from util import adjust_learning_rate, AverageMeter,print_running_time, Logger, check_pytorch_idx_validation, get_anchor_pos_neg
from sampleIdx import SampleIndex, RandomBatchSamplerWithPosAndNeg, RandomBatchSamplerWithSupplementPosAndNeg

from calculateSampleDis import calSampleDisAndImgCaseStudy
from onlyCalculdateFeat import onlyCalFeat


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print every print_freq batchs')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save model checkpoint every save_freq epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--contrastMethod', type=str, default='e2e',choices=['e2e', 'membank'], help='method of contrast, e2e or membank')

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
    parser.add_argument('--data_folder', type=str, default=None, help='path to training data') # 训练数据文件夹，即锚点/正负样本文件夹
    parser.add_argument('--test_data_folder', type=str, default=None, help='path to testing data') # 测试数据文件夹，即所有视频帧的文件夹
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')
    parser.add_argument('--log_txt_path', type=str, default=None, help='path to log file')
    parser.add_argument('--result_path', type=str, default=None, help='path to sample dis and img case study') # 训练结束后，图像间距离的case study保存在这个路径下

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.8, help='low area in crop')

    parser.add_argument('--comment_info', type=str, default='', help='Comment message, donot influence program')

    parser.add_argument('--supplement_pos_neg_txt_path', type=str, default='')
    parser.add_argument('--training_data_cache_method', type=str, default='default', choices=['default', 'memory', 'GPU'], help='where to save training data. \'memory\' or \'GPU\' will load all training data into memory or GPU at begining to speed up data reading at training stage.')
    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'

    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    
    opt.model_name = '{}_lossMethod_{}_NegNum_{}_Model_{}_lr_{}_decay_{}_bsz_{}_featDim_{}_contrasMethod_{}_{}'.format(curTime, opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size, opt.feat_dim, opt.contrastMethod, opt.comment_info)

    

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None) or (opt.log_txt_path is None) or (opt.result_path is None) or (opt.test_data_folder is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path | log_txt_path | result_path | test_data_folder')

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.log_txt_path):
        os.makedirs(opt.log_txt_path)

    opt.result_path = os.path.join(opt.result_path, opt.model_name)
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    
    log_file_name = os.path.join(opt.log_txt_path, 'log_'+opt.model_name+'.txt') 
    sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中

    if opt.comment_info != '':
        print('comment message : ', opt.comment_info)

    print('start program at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    print('Dataset :', opt.data_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    if opt.contrastMethod != 'e2e' and opt.contrastMethod != 'membank':
        raise ValueError('contrast method must be e2e or memory bank.')

    return opt


def get_train_loader(args):
    data_folder = os.path.join(args.data_folder, 'train')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform_withRandom = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.GaussianBlur(9, (0.1,3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    ])

    train_transform_withoutRandom = transforms.Compose([  
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # 选择训练数据读取方式
    # default：传统方式，即每次仅读取一个batch的数据到内存中；
    # memory 或 GPU：在程序开始阶段就将所有数据读到内存或显存中，以加快训练过程中的数据读取速度，但面临内存或显存不够用的问题
    # ！！！！！！！！！注意！！！！！！！！！如果数据被预先加载到显存中，则不会有数据增强，因为一旦进入显存后就无法再进行CPU运算了
    # 总结：如果选择将数据一次性加载到显存中，则不会有随机数据增强
    if args.training_data_cache_method == 'default':
        train_dataset = ImageFolderInstance(data_folder, transform=train_transform_withRandom)
    else:
        if torch.cuda.is_available() and args.training_data_cache_method == 'GPU':
            train_dataset = ImageFolderInstance_LoadAllImgToMemory(data_folder, transform=train_transform_withoutRandom, training_data_cache_method='GPU')
        else:
            train_dataset = ImageFolderInstance_LoadAllImgToMemory(data_folder, transform=train_transform_withRandom, training_data_cache_method='memory')
            if (not torch.cuda.is_available()) and args.training_data_cache_method == 'GPU':
                print('CUDA is not is_available, load all training data into memory instead of GPU')

    check_pytorch_idx_validation(train_dataset.class_to_idx)

    # 获得某类物体的图片索引，图片索引以set的形式保存
    classInstansSet = []
    tmpSet = set()
    last_t = 0
    for i,t in enumerate(train_dataset.targets):# targets的总数就是图片的总个数，targets里的每个值表示是该照片是第几类。targets是按顺序排列的
        if last_t != t:
            classInstansSet.append(tmpSet.copy())
            tmpSet.clear()
        last_t = t
        tmpSet.add(i)
    classInstansSet.append(tmpSet)# 把最后一个添加进去

    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    if args.supplement_pos_neg_txt_path == '': # 如果没有补充正负锚点文件的话，还按照原来的邻域正负样本采样方式
        batch_sampler = RandomBatchSamplerWithPosAndNeg(train_dataset, batch_size=args.batch_size, classInstansSet=classInstansSet, nce_k = args.nce_k)
    else:
        pos_neg_idx = get_anchor_pos_neg(args.supplement_pos_neg_txt_path, train_dataset, classInstansSet)
        batch_sampler = RandomBatchSamplerWithSupplementPosAndNeg(train_dataset, batch_size=args.batch_size, all_pos_neg_idx=pos_neg_idx, nce_k = args.nce_k)
        print('Training with supplement pos and neg. %s'%args.supplement_pos_neg_txt_path)

    pin_memory = True
    num_workers = args.num_workers
    if args.training_data_cache_method == 'GPU' and torch.cuda.is_available():
        pin_memory = False # 如果所有训练数据已经放在显存里了，那就不能再设置这个选项了。这个选项的含义是锁页内存，锁页内存中的数据永远不会与虚拟内存交换，即放在这里面的数据可以有很高的IO
        num_workers = 0 # 如果所有训练数据已经放在显存里了，那么就不能设置多线程transform了，因为这一步必须要CPU完成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, n_data, classInstansSet, train_dataset

def set_model(args, n_data, classInstansSet):

    model = MyAlexNetCMC(args.feat_dim)

    if args.resume:
        if torch.cuda.is_available():
            ckpt = torch.load(args.resume)
        else:
            ckpt = torch.load(args.resume,map_location=torch.device('cpu'))
        print("==> loaded pre-trained checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
        model.load_state_dict(ckpt['model'])
        print('==> done')

    contrast = 'placeholder'
    if args.contrastMethod == 'membank':
        contrast = NCEAverage(args.feat_dim, n_data,classInstansSet,args.nce_k, args.nce_t, args.nce_m, args.softmax)
    elif args.contrastMethod == 'e2e':
        contrast = E2EAverage(args.nce_k, n_data, args.nce_t, args.softmax)


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

def train_e2e(epoch,train_loader,train_dataset, model, contrast, sampleIndex, criterion, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    probs = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # img = sampleIndex.getAndCatAnchorPosNeg(target,opt.nce_k,img,train_dataset) # img的size是batchSize*(1+1+N),分别是anchor，pos，neg

        bsz = img.size(0)
        if torch.cuda.is_available():
            index = index.cuda()
            img = img.cuda()

        # ===================forward=====================
        feat = model(img)
        mutualInfo = contrast(feat)
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



def train_mem_bank(epoch,train_loader, model, contrast, criterion, optimizer, opt):
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
    train_loader, n_data, classInstansSet, train_dataset= get_train_loader(args)

    # set the model
    model, contrast, criterion = set_model(args,n_data,classInstansSet)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # train by epoch
    print('start training at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    sampleIdx = SampleIndex(classInstansSet)
    min_loss = np.inf
    best_model_path = ''
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)

        if args.contrastMethod == 'e2e':
            loss, prob = train_e2e(epoch, train_loader,train_dataset, model, contrast, sampleIdx, criterion, optimizer, args)
        else:
            loss, prob = train_mem_bank(epoch, train_loader, model, contrast, criterion, optimizer, args)

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

        if loss < min_loss:
            if min_loss != np.inf:
                os.remove(best_model_path)
            min_loss = loss
            best_model_path = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}_Best.pth'.format(epoch=epoch))
            print('==> Saving best model...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # if args.amp:
            #     state['amp'] = amp.state_dict()
            torch.save(state, best_model_path)
            # help release GPU memory
            del state
    
    print("==================== Training finished. Start testing ====================")
    print('==> loading best model')
    print('min loss = %.3f'%min_loss)
    if torch.cuda.is_available():
        ckpt = torch.load(best_model_path)
    else:
        ckpt = torch.load(best_model_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(best_model_path, ckpt['epoch']))
    print('==> done')
    model.eval()
    
    print('Calculating features...')
    onlyCalFeat(model, args)
    print('Done.\n\n')
    print('Calculating anchor-positive anchor-negative average distance......')
    calSampleDisAndImgCaseStudy(model, args) # 训练结束后，计算锚点-正样本和锚点-负样本间的平均距离，以及进行图像间距离的case study
    print('Done.\n\n')
    print('Program exit normally.')

if __name__ == '__main__':
    main()


