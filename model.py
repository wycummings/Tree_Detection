import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleNet(nn.Module):
    def __init__(self, input_size, num_class):
        super(SampleNet, self).__init__()
        self.feature = nn.Sequential(
                           ConvWithAct(3,   64,  downsample=True),
                           ConvWithAct(64,  64,  downsample=True),
                           ConvWithAct(64,  128, downsample=True),
                           ConvWithAct(128, 128, downsample=True),
                           ConvWithAct(128, 64,  downsample=True)
                       )
        self.fc = nn.Linear(int((input_size / 2 ** 5)) ** 2 * 64, num_class)
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvWithAct(nn.Module):
    def __init__(self, in_plane, plane, downsample=False):
        super(ConvWithAct, self).__init__()
        self.downsample = downsample
        self.conv = nn.Conv2d(in_plane, plane, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(num_features=plane, momentum=0.9)
        self.act  = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.downsample:
            x = F.max_pool2d(x, kernel_size=(2,2))
        return x
