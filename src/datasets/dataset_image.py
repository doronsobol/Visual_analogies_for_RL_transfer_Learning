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
import pdb



class dataset_image(data.Dataset):
  def __init__(self, specs):
    self.orig_gray = (1 == specs['orig_gray'])
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.scale = specs['scale']
    self.rotation = specs['rotation']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    self.random_crop = 1 == specs['random_crop']
    self.random_flip = 1 ==  specs['random_flip']
    self.use_first_channel = specs['use_first_channel'] if 'use_first_channel' in specs else False

    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def __getitem__(self, index):
    crop_img = self._load_one_image(self.images[index])
    raw_data = crop_img.transpose((2, 0, 1))  # convert to HWC
    data = ((torch.FloatTensor(raw_data)/255.0)-0.5)*2
    return data

  def __len__(self):
    return self.dataset_size

  def _load_one_image(self, img_name, test=False):
    if self.orig_gray:
      img = cv2.imread(img_name)
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:  
      try:
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        if self.use_first_channel:
          img = img[:,:,:1]
      except Exception as e:
        print(img_name)
        print(e)
    if self.scale > 0:
      img = cv2.resize(img,None,fx=self.scale,fy=self.scale)

    img = np.float32(img)
    if 2 == len(img.shape):
      h, w = img.shape
      img = np.reshape(img, (h, w, 1))
    h, w, c = img.shape
    if test==True:
      if self.random_crop:
        x_offset = np.int( (w - self.crop_image_width)/2 )
        y_offset = np.int( (h - self.crop_image_height)/2 )
      else:
        x_offset, y_offset = 0, 0
    else:
      if self.random_flip and np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      if self.random_crop:
        x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
        y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
      else:
        x_offset, y_offset = 0, 0
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    if 0 != self.rotation:
        rows, cols = crop_img.shape[:2]
        m = cv2.getRotationMatrix2D((cols/2,rows/2),self.rotation,1)
        crop_img = cv2.warpAffine(crop_img, m, (cols,rows))
    if 2 == len(crop_img.shape):
      h, w = crop_img.shape
      crop_img = np.reshape(crop_img, (h, w, 1))
    return crop_img

class dataset_blur_image(dataset_image):
  def _load_one_image(self, img_name, test=False):
    if self.orig_gray:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_GRAY2RGB)
    else:  
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3,3), 0)
    if self.scale > 0:
      img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      if self.random_crop:
        x_offset = np.int( (w - self.crop_image_width)/2 )
        y_offset = np.int( (h - self.crop_image_height)/2 )
      else:
        x_offset, y_offset = 0, 0
    else:
      if self.random_flip and np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      if self.random_crop:
        x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
        y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
      else:
        x_offset, y_offset = 0, 0

    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

class dataset_image_pairs(dataset_image):
  def __init__(self, specs):
    self.orig_gray = (1 == specs['orig_gray'])
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    self.scale = specs['scale']
    self.random_crop = 1 == specs['random_crop']
    self.random_flip = 1 ==  specs['random_flip']
    self.rotation = specs['rotation']
    self.labels_conversion = {}
    exec ('self.labels_conversion = {}'.format(specs['labels_conversion']))
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def __getitem__(self, index):
    crop_img_0, crop_img_1 = self._load_one_image(self.images[index])
    raw_data_1 = crop_img_1.transpose((2, 0, 1))  # convert to HWC
    raw_data_0 = crop_img_0.transpose((2, 0, 1))  # convert to HWC
    data_0 = ((torch.FloatTensor(raw_data_0)/255.0)-0.5)*2
    data_1 = ((torch.FloatTensor(raw_data_1)/255.0)-0.5)*2
    return (data_0, data_1)

  def _load_one_image(self, img_name, test=False):
    if self.orig_gray:
      img_1 = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_GRAY2RGB)
      img_2 = cv2.cvtColor(cv2.imread(img_name[:-5]+'2.jpg'), cv2.COLOR_GRAY2RGB)
    else:  
      try:
        img_0 = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(cv2.imread(img_name[:-5]+'2.jpg'), cv2.COLOR_BGR2RGB)
      except Exception as e:
        print(img_name)
        print(e)
        raise e
    if self.scale > 0:
      img_0 = cv2.resize(img_0,None,fx=self.scale,fy=self.scale)
      img_1 = cv2.resize(img_1,None,fx=self.scale,fy=self.scale)

    img_0 = np.float32(img_0)
    img_1 = np.float32(img_1)
    h, w, c = img_0.shape
    if test==True:
      if self.random_crop:
        x_offset = np.int( (w - self.crop_image_width)/2 )
        y_offset = np.int( (h - self.crop_image_height)/2 )
      else:
        x_offset, y_offset = 0, 0
    else:
      if self.random_flip and np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      if self.random_crop:
        x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
        y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
      else:
        x_offset, y_offset = 0, 0
    crop_img_0 = img_0[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    crop_img_1 = img_1[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    if 0 != self.rotation:
        rows, cols = crop_img_0.shape[:2]
        m = cv2.getRotationMatrix2D((cols/2,rows/2),self.rotation,1)
        crop_img_0 = cv2.warpAffine(crop_img_0, m, (cols,rows))
        crop_img_1 = cv2.warpAffine(crop_img_1, m, (cols,rows))
    return crop_img_0, crop_img_1

class dataset_image_pairs_and_actions(dataset_image_pairs):
  def get_action(self, index):
    path = self.images[index]
    label = self.labels_conversion[int(re.match(r'.*_[0-9]+_([0-9]+)_[0-3].jpg',path).group(1))]
    return label

  def __getitem__(self, index):
    action = self.get_action(index)
    crop_img_0, crop_img_1 = self._load_one_image(self.images[index])
    raw_data_1 = crop_img_1.transpose((2, 0, 1))  # convert to HWC
    raw_data_0 = crop_img_0.transpose((2, 0, 1))  # convert to HWC
    data_0 = ((torch.FloatTensor(raw_data_0)/255.0)-0.5)*2
    data_1 = ((torch.FloatTensor(raw_data_1)/255.0)-0.5)*2
    return (data_0, data_1), action

class dataset_image_pairs_and_actions_distance(dataset_image_pairs_and_actions):
  def get_numbers(self, index):
    path = self.images[index]
    return torch.FloatTensor([float(re.match(r'.*_([0-9]+)_[0-9]+_[0-3].jpg',path).group(1))])

  def __getitem__(self, index):
    action = self.get_action(index)
    distance = self.get_numbers(index)
    crop_img_0, crop_img_1 = self._load_one_image(self.images[index])
    raw_data_1 = crop_img_1.transpose((2, 0, 1))  # convert to HWC
    raw_data_0 = crop_img_0.transpose((2, 0, 1))  # convert to HWC
    data_0 = ((torch.FloatTensor(raw_data_0)/255.0)-0.5)*2
    data_1 = ((torch.FloatTensor(raw_data_1)/255.0)-0.5)*2

    return (data_0, data_1), (action, distance)


class dataset_imagenet_image(dataset_image):
  def __init__(self, specs):
    self.orig_gray = (1 == specs['orig_gray'])
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    self.scale = specs['scale']
    self.random_crop = 1 == specs['random_crop']
    self.random_flip = 1 ==  specs['random_flip']
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def _load_one_image(self, img_name, test=False):
    if self.orig_gray:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_GRAY2RGB)
    else:  
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    if h > w:
      scale = self.crop_image_width * 1.0 / w
    else:
      scale = self.crop_image_height * 1.0 / h
    scale *= self.scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      x_offset = np.int((w - self.crop_image_width) / 2)
      y_offset = np.int((h - self.crop_image_height) / 2)
    else:
      if self.random_flip and np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      if self.random_crop:
        x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
        y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
      else:
        x_offset, y_offset = 0, 0
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

class dataset_dvd_image(dataset_image):
  def __init__(self, specs):
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    self.orig_gray = (1 == specs['orig_gray'])
    self.random_crop = 1 == specs['random_crop']
    self.random_flip = 1 ==  specs['random_flip']
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def _load_one_image(self, img_name, test=False):
    if self.orig_gray:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_GRAY2RGB)
    else:  
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    # if h > w:
    #   scale = self.crop_image_width * 1.0 / w
    # else:
    #   scale = self.crop_image_height * 1.0 / h
    # img = cv2.resize(img, None, fx=scale, fy=scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      x_offset = np.int((w - self.crop_image_width) / 2)[0]
      y_offset = np.int((h - self.crop_image_height) / 2)[0]
    else:
      if self.random_flip and np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      if self.random_crop:
        x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
        y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
      else:
        x_offset, y_offset = 0, 0
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

