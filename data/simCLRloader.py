from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
from tqdm import tqdm


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

class SimCLRDataset(data.Dataset):
    def __init__(self, dataset_name, split, dataset_root=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        

        if self.dataset_name=='cifar10':

            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

        

            if split != 'test':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomRotation(5),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])
            
            self.transform = ContrastiveTransformations(self.transform, n_views=2)

            self.data = datasets.__dict__[self.dataset_name.upper()](
                root = dataset_root, train=self.split=='train',
                download=True, transform=self.transform)
            
        elif self.dataset_name=='cifar100':
            self.mean_pix = [x/255.0 for x in [129.3, 124.1, 112.4]]
            self.std_pix = [x/255.0 for x in [68.2, 65.4, 70.4]]
            
            if split != 'test':
                self.transform = transforms.Compose([ 
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomRotation(5),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])

            self.transform = ContrastiveTransformations(self.transform, n_views=2)

            self.data = datasets.__dict__[self.dataset_name.upper()](
                root = dataset_root, train=self.split=='train',
                download=True, transform=self.transform)

            
        # elif self.dataset_name=='svhn':
        #     self.mean_pix = [0.485, 0.456, 0.406]
        #     self.std_pix = [0.229, 0.224, 0.225]

        #     if self.random_sized_crop:
        #         raise ValueError('The random size crop option is not supported for the SVHN dataset')

        #     transform = []
        #     if (split != 'test'):
        #         transform.append(transforms.RandomCrop(32, padding=4))
        #     transform.append(lambda x: np.asarray(x))
        #     self.transform = transforms.Compose(transform)
        #     self.data = datasets.__dict__[self.dataset_name.upper()](
        #         dataset_root, split=self.split,
        #         download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dataset_name))

    def __getitem__(self, index):
        imgs, label = self.data[index] # n views of images and label
        return imgs, int(label)

    def __len__(self):
        return len(self.data)



