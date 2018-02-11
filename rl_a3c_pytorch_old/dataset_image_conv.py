"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from __future__ import print_function
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
import re
from pdb import set_trace as st
from skimage.color import rgb2gray

class dataset_image(data.Dataset):
    def __init__(self, dataset_file='demon_labels.txt.bak', is_test=False, root='/media/data3/doronsobol/pretrain_data/'):
        self.labels_file = dataset_file

        with open(self.labels_file) as f:
            labels_lines = f.readlines()

        self.dataset_size = len(labels_lines)

        if is_test:
            labels_lines = labels_lines[len(labels_lines)*80//100:]
        else:
            labels_lines = labels_lines[:len(labels_lines)*80//100]

        #self.imgs_labels = [{'img': x.split(':')[0], 'label': torch.FloatTensor(map(float ,x.split(':')[1].split()))} for x in labels_lines]
        self.imgs_labels = [{'img': os.path.join(root,x.split(':')[0]), 'label': list(map(lambda y: os.path.join(root,y.strip()) ,x.split(':')[1:]))} for x in labels_lines]
        self.permute = np.arange(self.dataset_size)
        np.random.shuffle(self.permute)
        self.dataset_size = len(self.imgs_labels)
        self.num_steps = 0

    def __getitem__(self, index):
        data = torch.FloatTensor(self._load_one_image(self.imgs_labels[index]['img']))
        labels_list = list(map(lambda x: torch.from_numpy(np.load(x)).float(), self.imgs_labels[index]['label']))

        labels = torch.cat(labels_list)

        return data, labels

    def __len__(self):
        return self.dataset_size

    def _load_one_image(self, img_name, test=False):
        try:
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(img_name)
            print(e)
        img = rgb2gray(img)
        #img = cv2.resize(rgb2gray(img), (80, conf["dimension2"]))
        img = cv2.resize(img, (80, 80))
        img = np.reshape(img, [1, 80, 80])
        return self.notmalize_image(img)

    
    def non_notmalize_image(self, observation):
        return observation

    def notmalize_image(self, observation):
        state_mean = 0
        state_std = 0
        alpha = 0.9999
    
        self.num_steps += 1
        state_mean = state_mean * alpha + \
            observation.mean() * (1 - alpha)
        state_std = state_std * alpha + \
            observation.std() * (1 - alpha)

        unbiased_mean = state_mean / (1 - pow(alpha, self.num_steps))
        unbiased_std = state_std / (1 - pow(alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
