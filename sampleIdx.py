import numpy as np
import torch
import random

class SampleIndex(object):
    def __init__(self, classInstansSet):
        self.classInstansSet = classInstansSet
        self.n_data = 0
        for item in classInstansSet:
            self.n_data += len(item)
        self.datasetIdx = set([i for i in range(self.n_data)])
    
    def getRandomIdx(self, targets, N):
        index = []
        for t in targets:
            posIdx = random.sample(self.classInstansSet[t.item()],1)
            negIdx = random.sample(self.datasetIdx - self.classInstansSet[t.item()], N)
            index.append(posIdx + negIdx)
        
        index = torch.Tensor(index).long()
        if torch.cuda.is_available():
            index = index.cuda()
        return index # 这里的size为batch_size * (N + 1)，其中N为负样本的个数。每行第0个数是正样本索引
        