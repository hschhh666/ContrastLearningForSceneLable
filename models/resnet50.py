import torch
import torch.nn as nn
from torchvision import models
from models.alexnet import Normalize

class MyResnet50(nn.Module):
    def __init__(self, feat_dim=128, pretrained=False):
        super(MyResnet50, self).__init__()
        self.pretrained = pretrained
        self.feat_dim = feat_dim
        self.encoder = models.resnet18(pretrained=self.pretrained)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, self.feat_dim),
            Normalize(2),
        )
        
    def forward(self, sample):
        return self.encoder(sample)

if __name__ == '__main__':
    model = MyResnet50(feat_dim=256,pretrained=True)
    model = model.cuda()
    data = torch.rand((32,3,224,224))
    data = data.cuda()
    res = model(data)
    print(res.shape)
    pass

