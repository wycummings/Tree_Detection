import torch
import torch.nn as nn
import torch.nn.functional
from torchvision import models


class vgg_16(nn.Module):
    def __init__(self, input_size, num_class):
        super(vgg_16, self).__init__()
        vgg_base=models.vgg16(pretrained=True)
        self.features=vgg_base.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, num_class)
        )

       
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
