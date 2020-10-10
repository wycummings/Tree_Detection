import torch
import torch.nn as nn
import torch.nn.functional
from torchvision import models


class resnet18(nn.Module):
    def __init__(self, input_size, num_class):
        super(resnet18, self).__init__()
        resnet18_base = models.resnet18(pretrained=True)
        num_ftrs = resnet18_base.fc.in_features
        resnet18_base.fc = nn.Linear(num_ftrs, num_class)
        self.model = resnet18_base
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class resnet34(nn.Module):
    def __init__(self, input_size, num_class):
        super(resnet34, self).__init__()
        resnet34_base = models.resnet34(pretrained=True)
        num_ftrs = resnet34_base.fc.in_features
        resnet34_base.fc = nn.Linear(num_ftrs, num_class)
        self.model = resnet34_base
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet50(nn.Module):
    def __init__(self, input_size, num_class):
        super(resnet50, self).__init__()
        resnet50_base = models.resnet50(pretrained=True)
        num_ftrs = resnet50_base.fc.in_features
        resnet50_base.fc = nn.Linear(num_ftrs, num_class)
        self.model = resnet50_base
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet101(nn.Module):
    def __init__(self, input_size, num_class):
        super(resnet101, self).__init__()
        resnet101_base = models.resnet101(pretrained=True)
        num_ftrs = resnet101_base.fc.in_features
        resnet101_base.fc = nn.Linear(num_ftrs, num_class)
        self.model = resnet101_base
        
    def forward(self, x):
        x = self.model(x)
        return x

