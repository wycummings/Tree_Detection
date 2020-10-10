import os
import pdb
import random
import glob

import torch
from   torch.utils.data       import DataLoader
from   torchvision.datasets   import ImageFolder
import torchvision.transforms as     transforms
import numpy                  as     np
import PIL.Image              as     Image


class SamplePairing(object):

    def __init__(self, img_path):
        self.img_path  = img_path
        self.img_list  = glob.glob('./new_dataset/train/*/*.png')
        np.random.seed(10)

    def __call__(self, img):
        toss = np.random.choice([1, 0])

        if toss:
            img_a_array = np.asarray(img)

            # pick one image from the pool
            img_b       = np.random.choice(self.img_list)
            img_b_array = np.asarray(Image.open(img_b))

            # mix two images
            mean_img    = np.mean([img_a_array, img_b_array], axis=0)
            img         = Image.fromarray(np.uint8(mean_img))

        return img

    
def make_dataloader(dir_path, batchsize, patchsize, SP=False, val=False):

    #SP = SamplePairing(dir_path)
    
    # dir_path = './new_dataset/train' 
    
    # normalizition and data augmentation setting
    if val:
        transform = transforms.Compose([
                      transforms.CenterCrop(patchsize),
                      transforms.ToTensor()
                    ])
    elif SP:
        transform = transforms.Compose([
                      SamplePairing(dir_path),
                      transforms.RandomCrop(patchsize),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor()
                    ])
    else:
        transform = transforms.Compose([
                      transforms.RandomCrop(patchsize),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor()
                    ])

    # make dataloader
    dataset    = ImageFolder(dir_path, transform=transform)     
    dataloader = DataLoader(
                   dataset,
                   batch_size  = batchsize,
                   shuffle     = not val,
                   num_workers = 4
                 )

    return dataloader