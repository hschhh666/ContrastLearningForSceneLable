import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse

import numpy as np

from torchvision import transforms, datasets

from models.alexnet import MyAlexNetCMC
from dataset import ImageFolderInstance
from sampleIdx import SampleIndex
from util import AverageMeter,print_running_time
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--result_path', type=str, default=None, help='path to save result')

    parser.add_argument('--crop_low', type=float, default=0.8, help='low area in crop')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')



    opt = parser.parse_args()

    # opt.data_folder = 'C:\\Users\\HuShaochi\\Desktop\\FewAnchorBasedSceneRecognition\\dataset'


    if (opt.data_folder is None) or (opt.model_path is None) or (opt.result_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | result_path')

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    
    return opt


def get_train_val_loader(args):
    train_folder = os.path.join(args.data_folder, 'train')
    val_folder = os.path.join(args.data_folder, 'val')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.Resize((224,224)),
        transforms.RandomGrayscale(p=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.Resize((224,224)),
        transforms.RandomGrayscale(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = ImageFolderInstance(args, train_folder, transform=train_transform)
    val_dataset = ImageFolderInstance(args, val_folder, transform=val_transform)

    train_n_data = len(train_dataset)
    val_n_data = len(val_dataset)
    print('number of train: {}'.format(len(train_dataset)))
    print('number of val: {}'.format(len(val_dataset)))

    # 获得某类物体的图片索引，图片索引以set的形式保存
    train_classInstansSet = []
    tmpSet = set()
    last_t = 0
    for i,t in enumerate(train_dataset.targets):
        if last_t != t:
            train_classInstansSet.append(tmpSet.copy())
            tmpSet.clear()
        last_t = t
        tmpSet.add(i)
    train_classInstansSet.append(tmpSet)

    # 获得某类物体的图片索引，图片索引以set的形式保存
    val_classInstansSet = []
    tmpSet = set()
    last_t = 0
    for i,t in enumerate(val_dataset.targets):
        if last_t != t:
            val_classInstansSet.append(tmpSet.copy())
            tmpSet.clear()
        last_t = t
        tmpSet.add(i)
    val_classInstansSet.append(tmpSet)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle = False)#这里没必要随机打乱
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,shuffle = False)


    return train_classInstansSet, val_classInstansSet, train_loader, val_loader, train_dataset, val_dataset

def set_model(args):
    print('==> loading pre-trained model')
    if torch.cuda.is_available():
        ckpt = torch.load(args.model_path)
    else:
        ckpt = torch.load(args.model_path,map_location=torch.device('cpu'))
    featDim = ckpt['opt'].feat_dim
    model = MyAlexNetCMC(featDim)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')
    
    model.eval()
    return model


def getFeatMem(model, data_loader,n_data): # 计算所有数据的feature，并保存入memory中
    model_param = list(model.named_parameters())
    featDim = model_param[-1][1].size(0)
    memory = torch.ones(n_data, featDim) 
    if torch.cuda.is_available():
        memory = memory.cuda()
    for idx,(img, target, index) in enumerate(data_loader):
        if torch.cuda.is_available():
            img = img.cuda()
            index = index.cuda()
        feat = model(img)
        with torch.no_grad():#得加这句话，要不显存跑着跑着就不够了
            memory.index_copy_(0,index,feat) 
        # print('calculating feature {}/{}'.format(idx+1, len(data_loader)))

    if torch.cuda.is_available():
        memory = memory.cpu()
    return memory

def calculateAvgSampleDis(memory, classInstansSet):
    # 设总共有n个样本，那么这里做的工作就是，对于每个锚点，都给它随机找一个正样本，然后计算它和正样本的距离；同样也给它随机找一个负样本，然后计算它和负样本间的距离。注意，是每个锚点，都只找一个正样本，和一个负样本。

    sampleIdx = SampleIndex(classInstansSet)
    n_data = memory.size(0)
    target = []
    for i,s in enumerate(classInstansSet):
        target = target + ([i]*len(s))
    target = torch.Tensor(target).int() # 每个图片所对应的类

    posDises = AverageMeter()
    negDises = AverageMeter()

    with torch.no_grad():
        loop = 30 # 因为每个锚点都只随机取一个正负样本，样本量可能有点少，所以这整个过程重复算几次
        for i in range(loop):
            # print('calculating mutual information {}/{}'.format(i+1, loop))
            # 首先计算锚点与正样本间的平均距离
            posIdx = sampleIdx.getNPosIdx(target).view(-1)# 每个锚点的正样本的索引
            posMem = torch.index_select(memory,0,posIdx) # 取出正样本的特征
            posMem = posMem.transpose(1,0).contiguous()  # 转置一下，方便矩阵乘法
            posDis = torch.mm(memory, posMem) # 矩阵乘法 
            posDis = torch.exp(posDis) # 这里要取一下指数，那么结果的取值范围就为[0.36,  2.71]
            posDis = posDis.trace()/n_data #取对角线，才是锚点与它正样本的距离
            posDises.update(posDis)

            # 再计算锚点与负样本间的平均距离
            negIdx = sampleIdx.getNNegIdx(target).view(-1)
            negMem = torch.index_select(memory,0,negIdx)
            negMem = negMem.transpose(1,0).contiguous()
            negDis = torch.mm(memory, negMem)
            negDis = torch.exp(negDis)
            negDis = negDis.trace()/n_data
            negDises.update(negDis)
    
    return posDises.avg, negDises.avg


def imageCaseStudy(args, memory, classInstansSet, my_dataset, name = ''):# 就是画图，画图做case study
    sampleIndex = SampleIndex(classInstansSet)
    n_data = len(my_dataset)

    anchor_num = 5
    samplePerAnchor = 4 #每个anchor找samplePerAnchor个样本，其中一半正样本一半负样本
    
    anchor = np.random.randint(0,n_data,anchor_num) # 随机采样n个anchor，得到它们的index
    InstanceClassDic = {}
    for i , s in enumerate(classInstansSet):
        for t in s:
            InstanceClassDic[t] = i
    target = []
    for i in anchor:
        target.append(InstanceClassDic[i]) # 获取anchor所在的类
    
    anchor = torch.Tensor(anchor).long()
    target = torch.Tensor(target).long()
    posIdxes = sampleIndex.getNPosIdx(target,samplePerAnchor//2) # 找锚点的正样本
    negIdxes = sampleIndex.getNNegIdx(target,samplePerAnchor//2) # 找锚点的负样本
    sample = torch.cat((posIdxes, negIdxes),1) # concatenate 正负样本的索引

    dises = torch.Tensor()
    for i, _ in enumerate(anchor):
        anchor_mem = torch.index_select(memory,0, anchor[i])
        sample_mem = torch.index_select(memory,0, sample[i])
        sample_mem = sample_mem.transpose(1,0).contiguous()
        dis = torch.mm(anchor_mem, sample_mem) # 计算anchor与它的正负样本间的距离
        dises = torch.cat((dises,dis),0)

    dises = torch.exp(dises)
    dises = dises.detach().numpy()
    
    def imgConvent(img):
        img = img[0] # 别问为啥，问就是试出来的
        mean=[0.485, 0.456, 0.406] 
        std=[0.229, 0.224, 0.225]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        for i in range(3):
            img[:,:,i] = img[:,:,i]*std[i]+mean[i] # 把原本normalize的图像变回来
        return img

    m = anchor_num
    n = 1 + samplePerAnchor
    
    plt.figure(figsize=(6*1.5,4.8*1.5 ))
    for i in range(m):
        anchor_img = imgConvent(my_dataset[anchor[i]])
        plt.subplot(m,n, n * i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(anchor_img)
        for j in range(samplePerAnchor):
            sample_img = imgConvent(my_dataset[sample[i][j]])
            plt.subplot(m,n, n * i + j + 2)
            plt.xlabel(dises[i][j])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sample_img)
    plt.savefig(os.path.join(args.result_path, 'case_%s.png'%(name)))
    plt.close()
    
def main():# 供直接运行本脚本

    # parse argument
    args = parse_option()

    # get data
    train_classInstansSet, val_classInstansSet, train_loader, val_loader, train_dataset, val_dataset = get_train_val_loader(args)

    # get data size
    train_n_data = len(train_dataset)
    val_n_data = len(val_dataset)

    # load model
    model = set_model(args)

    # calculate positive distance
    train_memory = getFeatMem(model, train_loader, train_n_data)
    posDis, negDis = calculateAvgSampleDis(train_memory, train_classInstansSet)

    f = open(os.path.join(args.result_path,'testingLog.txt'),'w')

    line = 'train dataset : positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis)
    print(line)
    f.write(line+'\n\n')

    
    # calculate negative distance
    val_memory = getFeatMem(model, val_loader, val_n_data)    
    posDis, negDis = calculateAvgSampleDis(val_memory, val_classInstansSet)
    line = 'val dataset : positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis)
    print(line)
    f.write(line+'\n')
    f.close()

    np.save(os.path.join(args.result_path, 'memory.npy'),val_memory.numpy())
    print('Save memory to file ',os.path.join(args.result_path, 'memory.npy'))

    # image case study
    if torch.cuda.is_available():
        torch.cuda.empty_cache() #释放部分显存
    caseNum = 50
    for i in range(caseNum):
        print('ploting case study images {}/{}'.format(i*2+1, caseNum*2))
        imageCaseStudy(args, train_memory, train_classInstansSet, train_dataset, name= 'train'+str(i))
        print('ploting case study images {}/{}'.format(i*2+2, caseNum*2))
        imageCaseStudy(args, val_memory, val_classInstansSet, val_dataset,name= 'val' + str(i))


def calSampleDisAndImgCaseStudy(model, args): # 供外部调用
    # get data
    train_classInstansSet, val_classInstansSet, train_loader, val_loader, train_dataset, val_dataset = get_train_val_loader(args)

    # get data size
    train_n_data = len(train_dataset)
    val_n_data = len(val_dataset)

    # calculate positive distance
    train_memory = getFeatMem(model, train_loader, train_n_data)
    posDis, negDis = calculateAvgSampleDis(train_memory, train_classInstansSet)

    f = open(os.path.join(args.result_path,'testingLog.txt'),'w')

    line = 'train dataset : positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis)
    print(line)
    f.write(line+'\n\n')

    
    # calculate negative distance
    val_memory = getFeatMem(model, val_loader, val_n_data)    
    posDis, negDis = calculateAvgSampleDis(val_memory, val_classInstansSet)
    line = 'val dataset : positive sample average distance = {}, negative sample average distance = {}'.format(posDis, negDis)
    print(line)
    f.write(line+'\n')
    f.close()

    # # image case study
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache() #释放部分显存
    # caseNum = 5
    # for i in range(caseNum):
    #     # print('ploting case study images {}/{}'.format(i*2+1, caseNum*2))
    #     imageCaseStudy(args, train_memory, train_classInstansSet, train_dataset, name= 'train'+str(i))
    #     # print('ploting case study images {}/{}'.format(i*2+2, caseNum*2))
    #     imageCaseStudy(args, val_memory, val_classInstansSet, val_dataset,name= 'val' + str(i))

if __name__ == '__main__':
    main()

