import torch
from torch import nn
from sampleIdx import SampleIndex
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, classInstansSet, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)) # outputSize指的是样本总数，inputSize指的是特征维度
        self.sampleIdx = SampleIndex(classInstansSet)

    def forward(self, anchor, target, index):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()
        batchSize = anchor.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)
        
        idx = self.sampleIdx.getRandomIdx(target,K)
        memory_feat = torch.index_select(self.memory,0,idx.view(-1)).detach()
        memory_feat = memory_feat.view(batchSize, K+1, inputSize)
        mutualInfo = torch.bmm(memory_feat, anchor.view(batchSize, inputSize,1))


        if self.use_softmax:
            mutualInfo = torch.div(mutualInfo, T)
            mutualInfo = mutualInfo.contiguous()
        else:
            mutualInfo = torch.exp(torch.div(mutualInfo, T))
            # Z是归一化因子，这里取的是常数。计算方法是第一次算出来的互信息的均值
            if Z < 0:
                self.params[2] = mutualInfo.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            mutualInfo = torch.div(mutualInfo, Z).contiguous()

        # # update memory
        with torch.no_grad():
            feat = torch.index_select(self.memory, 0, index.view(-1))
            feat.mul_(momentum)
            feat.add_(torch.mul(anchor, 1 - momentum))
            norm = feat.pow(2).sum(1, keepdim=True).pow(0.5)
            updated = feat.div(norm)
            self.memory.index_copy_(0, index, updated)

        return mutualInfo

# =========================
# End to End method
# =========================
class E2EAverage(nn.Module):
    def __init__(self, K, outputSize, T=0.07, use_softmax=False):
        super(E2EAverage, self).__init__()
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T, -1,outputSize]))
    
    def forward(self, feat):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        outputSize = self.params[3].item()

        featNum = feat.size(0)
        bsz = featNum // (K+2) #  每个anchor对应一个pos和K个neg，即(K+2)个是一组。
        featSize = feat.size(1) # 获取特征的维度
        mutualInfo = torch.ones(bsz,K+1,1) # 这个size和基于memory bank的方法是一致的，也别问为啥了，反正一样就能用。
        if torch.cuda.is_available():
            mutualInfo = mutualInfo.cuda()
        for i in range(bsz):#逐个计算每张照片与其正负样本的距离
            anchor = feat[i].view(featSize,-1) # 这里相当于扩维+矩阵转置
            pos = feat[bsz + i*(K+1)].view(1,-1)# 扩充维度
            neg = feat[(bsz + i*(K+1) + 1):(bsz + i*(K+1) + 1 + K)] #选择
            sampleFeat = torch.cat((pos,neg),0)
            mi = torch.mm(sampleFeat, anchor)
            mutualInfo[i] = mi
        
        if self.use_softmax:
            mutualInfo = torch.div(mutualInfo, T)
            mutualInfo = mutualInfo.contiguous()
        else:
            mutualInfo = torch.exp(torch.div(mutualInfo, T))
            # Z是归一化因子，这里取的是常数。计算方法是第一次算出来的互信息的均值
            if Z < 0:
                self.params[2] = mutualInfo.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            mutualInfo = torch.div(mutualInfo, Z).contiguous()
            
        return mutualInfo


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
