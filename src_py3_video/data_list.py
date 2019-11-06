#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os, sys
import os.path



def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        elif len(image_list[0].split()) == 2 :
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
        else:
            images = [(val.split()[0], "xx") for val in image_list]
    return images

def pil_spectloader(path):
    # img_list = []
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_list = img.convert('RGB')
    return img_list

def pil_loader(path):
    img_fold = os.listdir(path)
    img_list = []
    for im in img_fold:
        with open(os.path.join(path, im), 'rb') as f:
            with Image.open(f) as img:
                img_list.append(img.convert('RGB'))
    return img_list

def newpil_loader(path):
    img_fold = os.listdir(path)
    img_list = []
    for im in img_fold:
        with open(os.path.join(path, im), 'rb') as f:
            with Image.open(f) as img:
                img_list.append(img.convert('RGB'))
    return img_list[random.randint(0, len(img_list)-1)]


def default_loader(path):
    return newpil_loader(path)


class ImageList(object):

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,loader=default_loader):

        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_spectloader

    def __getitem__(self, index):
        
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            video = self.transform(img)
        if self.target_transform is not None:
                target = self.target_transform(target)
        return video, target

    def __len__(self):
        return len(self.imgs)


