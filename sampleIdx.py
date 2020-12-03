import numpy as np
import torch
import random

class SampleIndex(object):
    def __init__(self, classInstansSet):
        self.classInstansSet = classInstansSet
        self.n_data = 0
        self.class_num = 0
        for item in classInstansSet:
            self.n_data += len(item)
            self.class_num += 1
        self.datasetIdx = set([i for i in range(self.n_data)])
    
    def getRandomIdx(self, targets, N):# targets的尺寸是batch_size*1, 表明该batch中每个图片所述类别。对每个图片，采样N个它所对应的负样本
        index = []
        for t in targets:
            posIdx = random.sample(self.classInstansSet[t.item()],1)
            negIdx = []
            if t == 0:
                negIdx += self.classInstansSet[t.item()+1]
                negIdx += self.classInstansSet[self.class_num - 1]
            elif t == self.class_num - 1:
                negIdx += self.classInstansSet[t.item()-1]
                negIdx += self.classInstansSet[0]
            else:
                negIdx += self.classInstansSet[t.item()-1]
                negIdx += self.classInstansSet[t.item()+1]

            # if t != 0:
            #     negIdx += self.classInstansSet[t.item()-1]
            # if t != self.class_num - 1:
            #     negIdx += self.classInstansSet[t.item()+1]
            
            negIdx = random.sample(negIdx, N)
            # negIdx = random.sample(self.datasetIdx - self.classInstansSet[t.item()], N)
            index.append(posIdx + negIdx)

        index = torch.Tensor(index).long()
        if torch.cuda.is_available():
            index = index.cuda()
        return index # 这里的size为batch_size * (N + 1)，其中N为负样本的个数。每行第0个数是正样本索引
    
    def getAndCatAnchorPosNeg(self, targets, N, anchor, dataset): # 输入batchSize张anchor image, 输出 batchSize + batchSize*(1+N)张照片，batchSize是anchor image，batchSize*(1+N)中，对于每个batch，第一个是pos，剩下N个是neg
        index = self.getRandomIdx(targets,N)
        bsz = anchor.size(0)
        img = torch.ones(bsz*(N+2),anchor.size(1),anchor.size(2),anchor.size(3))
        img[0:bsz] = anchor
        count = bsz
        for idx in index:
            for i in idx:
                img[count] = dataset[i][0]
                count += 1
        return img


    
    def getNPosIdx(self, targets,n=1):# targets的尺寸是batch_size*1, 表明该batch中每个图片所述类别。对每个图片，采样n个它所对应的正样本
        index = []
        for t in targets:
            posIdx = random.sample(self.classInstansSet[t.item()],n)
            index.append(posIdx)
        
        return torch.Tensor(index).long()

    def getNNegIdx(self, targets,n=1):# targets的尺寸是batch_size*1, 表明该batch中每个图片所述类别。对每个图片，采样n个它所对应的负样本
        index = []
        for t in targets:
            negIdx = []
            if t == 0:
                negIdx += self.classInstansSet[t.item()+1]
                negIdx += self.classInstansSet[self.class_num - 1]
            elif t == self.class_num - 1:
                negIdx += self.classInstansSet[t.item()-1]
                negIdx += self.classInstansSet[0]
            else:
                negIdx += self.classInstansSet[t.item()-1]
                negIdx += self.classInstansSet[t.item()+1]
            negIdx = random.sample(negIdx, n)
            index.append(negIdx)
        return torch.Tensor(index).long()