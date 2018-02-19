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
    def __init__(self, dataset_file='demon_labels.txt.bak',is_max=True, is_test=False, sec_len=5, use_translation=True, root='/media/data2/doronsobol/'):
        self.labels_file = os.path.join(root, dataset_file)
        if 'tennis' in self.labels_file:
            self.translator = np.array([0 ,1 , 0, 2, 3, 0, 2, 3, 2, 3, 1, 4, 5, 1 , 4, 5, 4, 5])
        else:
            self.translator = np.array([0, 1, 4, 5])

        self.use_translation = use_translation
        self.sec_len = sec_len
        with open(self.labels_file) as f:
            labels_lines = f.readlines()

        self.dataset_size = len(labels_lines)/self.sec_len
        labels_lines = labels_lines[:self.dataset_size*self.sec_len]

        if is_test:
            labels_lines = labels_lines[len(labels_lines)*80/100:]
        else:
            labels_lines = labels_lines[:len(labels_lines)*80/100]

        self.imgs_labels = [{'img': x.split(':')[0], 'label': torch.FloatTensor(map(float ,x.split(':')[1].split()))} for x in labels_lines]
        #self.imgs_labels = [{'img': x.split(':')[0], 'label': [torch.FloatTensor(map(float ,c.split())), for c in x.split(':')]} for x in labels_lines]
        self.permute = np.arange(self.dataset_size)
        np.random.shuffle(self.permute)
        self.dataset_size = len(self.imgs_labels)/self.sec_len
        self.num_steps = 0
        self.is_max = is_max

    def __getitem__(self, index):
        crop_imgs = [torch.FloatTensor(self._load_one_image(self.imgs_labels[self.sec_len*index+i]['img'])) for i in range(self.sec_len)]
        data = torch.stack(crop_imgs)

        labels = [self.imgs_labels[self.sec_len*index+i]['label'] for i in range(self.sec_len)]

        labels = torch.stack(labels)
        if self.is_max:
            labels = torch.max(labels, 1)[1]
            if self.use_translation:
                labels = self.translator[labels.numpy()]
                labels = torch.from_numpy(labels)

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
