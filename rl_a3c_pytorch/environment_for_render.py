from __future__ import division, print_function
import gym
import os
import numpy as np
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize
from skimage.color import rgb2gray
import json
import cv2
import logging
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pdb import set_trace as st
from PIL import Image, ImageTk
from tools import *
import itertools
from torch.autograd import Variable
import random
import cv2
from collections import deque
tennis_median_im = cv2.imread('/tmp/median.jpg', 0)

def create_xy_image(width=256):
  coordinates = list(itertools.product(range(width), range(width)))
  arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / float((width/2))
  new_map = np.transpose(np.float32(arr), [2, 0, 1])
  #xy = Variable(torch.from_numpy(new_map), requires_grad=False)
  xy = torch.from_numpy(new_map).clone()
  return xy

#xy = create_xy_image().cuda()
xy = create_xy_image()
xy = xy.unsqueeze(0).expand(1, xy.size(0), xy.size(1), xy.size(2))
state_to_save = np.zeros((80,80,3))

def  init_render():
  pass
  #global canvas, data, state_to_save, theimage, tk, tk,master
import tkinter as tk
canvas_width = 256*2
canvas_height = 256
master = tk.Tk()
#canvas = Canvas(master, width=canvas_width,height=canvas_height)
frame = tk.Frame(master, width=256*2, height=256)
frame.pack()
canvas = tk.Canvas(frame, width=256*2,height=256)
canvas.place(x=-2,y=-2)
data=np.array(np.random.random((256,256*2))*100,dtype=int)
theimage = Image.frombytes('L', (data.shape[1],data.shape[0]), data.astype('b').tostring())

def dist_exp(im):
    e_im = np.exp(-im/6.)
    e_im = 1.0-e_im
    new_im = e_im*255.0
    return new_im.astype(np.uint8)

transformations = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
def atari_env(env_id, env_conf, convertor=None, flann=None, config=None, opts=None, test=True):
    args = opts
    if opts is None or opts.render:
        init_render()
    if opts.deterministic:
        env_id = env_id.split('-')[0] + 'NoFrameskip-'+ env_id.split('-')[1]
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.skip_rate)

    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        transform = transforms.Compose(transformations)
        env = AtariRescale(env, env_conf, convertor, transform, flann, config, opts)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    return env

p_frame = np.zeros((1, 80, 80))
pp_frame = np.zeros((1, 80, 80))
im_num = 0
def _orig_process_frame(frame, conf, convertor, transform, obj, args):
    global p_frame, pp_frame, im_num
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = cv2.resize(frame, (256, 256))
    frame = rgb2gray(frame)*255.
    frame = cv2.resize(frame, (256, 256))

    if args.render:
        #st()
        orig_ata = frame.astype(np.uint8)
    frame = transform_images(frame, 'B', convertor.args.image_size, gray=args.gray, rotation=args.rotate_A)


    attention, frame = convertor.test(frame)

    cols = frame.shape[1]
    rows = frame.shape[0]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),360-args.rotate_B,1)
    #cv2.imwrite(im_path_E, frame)
    frame = cv2.warpAffine(frame, M, (cols,rows))
    if args.render:
        cols = frame.shape[1]
        rows = frame.shape[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
        #cv2.imwrite(im_path_E, frame)
        frame = cv2.warpAffine(frame, M, (cols,rows))

        conv_frame = cv2.resize(frame, (256,256)) 
        data = np.concatenate((orig_ata, conv_frame), axis=1)
        data = cv2.resize(data, (256*2, 256)).astype(int)
        #cv2.imwrite('im_log/{}.jpg'.format(im_num), data)
        #st()
        im=Image.frombytes('L', (256*2,256), data.astype('b').tostring())
        photo = ImageTk.PhotoImage(image=im)
        canvas.create_image(0,0,image=photo,anchor=tk.NW)
        master.update()
        data=np.roll(data,-1,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    frame = rgb2gray(frame)
    frame = cv2.resize(frame, (80, conf["dimension2"]))
    #frame = cv2.resize(rgb2gray(frame), (80, conf["dimension2"]))
    frame = cv2.resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    obj.im_num += 1
    im_num += 1
    #eturn framei

    all_frame = np.concatenate((pp_frame, p_frame, frame), axis=0)
    pp_frame = p_frame
    p_frame = frame
    return all_frame

def frame2attention(frame, conf, trainer, obj, config, opts):
    global tennis_median_im
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    old_frame = frame
    orig_ata = frame
    dilation = frame
    #frame = np.transpose(frame, (2, 0, 1))
    #print("before: {}".format(str(np.array(frame).shape)))
    #print("before: {}".format(str(type(np.array(frame)[0][0][0]))))
    #print(frame.shape)
    #attention = frame
    #convertor = None
    if opts.blurr:
        #im_path_A = os.path.join("im_log", str(obj.im_num) + '_A.png')
        #im_path_D = os.path.join("im_log", str(obj.im_num) + '_D.png')
        #im_path_B = os.path.join("im_log", str(obj.im_num) + '_B.png')
        #im_path_E = os.path.join("im_log", str(obj.im_num) + '_E.png')
        #im_path_F = os.path.join("im_log", str(obj.im_num) + '_F.png')
        #st()
        frame = cv2.resize(frame, (256, 256))
        #cv2.imwrite(im_path_A, frame[:,:,::-1])
        frame = rgb2gray(frame)*255.
        orig_ata = frame
        #st()
        #median_im = np.zeros((256,256))o
        if 'Pong' in opts.env:
            median_im = 87.
            frame = np.absolute((frame - median_im)).astype(np.uint8)
        elif 'Tennis' in opts.env:
            #median_im = 95.
            frame = np.absolute((frame - tennis_median_im)).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        #imgplot = plt.imshow(frame)
        #frame = frame.astype(np.uint8)
        #cv2.imwrite(im_path_B, cv2.resize(frame, (64,64)))
        if False:
            _, filtered_bin_image = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)
        else:
            _, filtered_bin_image = cv2.threshold(frame,20,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation = cv2.dilate(filtered_bin_image ,kernel,iterations = 1)
        #imgplot = plt.imshow(dilation)
        #plt.show()
        #cv2.imwrite(im_path_C, cv2.resize(dilation, (64, 64)))
        if opts.blurr:
            inv_im = 255-dilation
            inv_im_dist = cv2.distanceTransform(inv_im, cv2.DIST_L2, 3)
            exp_inv_im_dist = dist_exp(inv_im_dist)
            exp_im_dist = 255-exp_inv_im_dist

            exp_inv_im_dist = dist_exp(inv_im_dist/2)
            exp_im_dist_1 = 255-exp_inv_im_dist

            if False:
                frame = exp_im_dist
            else:
                #frame = np.stack((exp_im_dist, exp_im_dist_1, dilation), axis=-1)
                frame = np.stack((dilation, exp_im_dist_1, exp_im_dist), axis=-1)
        else:
            frame = np.stack((dilation, dilation,dilation), axis=-1)
            #cv2.imwrite(im_path_B, frame[:,:,::-1])
    return frame, orig_ata, old_frame, dilation

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def _atten_process_frame(frame, conf, trainer, obj, config, opts):
    global p_frame, pp_frame, im_num, state_to_save  
    global theimage, data, orig_ata, orig_heimage, canvas
    frame, orig_ata, old_frame, dilation = frame2attention(frame, conf, trainer, obj, config, opts)
    #im_path_C = os.path.join("im_log", str(obj.im_num) + '_C.png')
    blurred_frame = frame
    #orig_ata = frame[:,:,0]
    frame_to_render = frame
    if opts.convertor:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = rgb2gray(frame)*255.

        #imgplot = plt.imshow(frame)
        #plt.show()
        if 1==opts.a2b:
            rotate_A = config.datasets['train_a']['rotation']
            rotate_B = config.datasets['train_b']['rotation']
        else:
            rotate_A = config.datasets['train_b']['rotation']
            rotate_B = config.datasets['train_a']['rotation']

        cols = frame.shape[1]
        rows = frame.shape[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_A,1)
        frame = cv2.warpAffine(frame, M, (cols,rows))


        frame = frame.transpose(2,0,1) 
        if not opts.blurr:
            frame = frame[-1:,:,:]
        final_data = torch.from_numpy((frame / 255.0 - 0.5) * 2).float().clone()
        #final_data = torch.FloatTensor((frame / 255.0 - 0.5) * 2)
        final_data = final_data.contiguous()
        final_data = final_data.resize_(1,final_data.size(0),final_data.size(1),final_data.size(2))


        use_xy = 1 == config.hyperparameters['gen']['use_xy']

        if use_xy:
            final_data = torch.cat([final_data, xy], 1)
        #final_data = final_data

        #xy = Variable(torch.from_numpy(new_map), requires_grad=False)
        if opts.cuda:
            #final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2))).cuda(opts.gpu)
            final_data_in = Variable(final_data).cuda(trainer.gpu)
        else:
            final_data_in = Variable(final_data)

        orig_ata = frame[0]


        final_data_in = final_data_in.contiguous()
        #ForkedPdb().set_trace()
        if 1==opts.a2b:
            output_data = trainer.gen.forward_a2b(final_data_in)
            #output_data = (final_data,)
        else:
            output_data = trainer.gen.forward_b2a(final_data_in)
            #output_data = (final_data,)

        output_img = output_data[0].data.cpu().numpy()
        #print(output_img.size())
        new_output_img = np.transpose(output_img, [2, 3, 1, 0])
        new_output_img = new_output_img[:, :, :, 0]
        out_img = np.uint8(255 * (new_output_img / 2 + 0.5))
        #frame = 255-out_img
        frame = out_img


        cols = frame.shape[1]
        rows = frame.shape[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),360-rotate_B,1)
        #cv2.imwrite(im_path_E, frame)
        frame = cv2.warpAffine(frame, M, (cols,rows))
        frame_to_render = frame
        """
        if not args.gray:
            frame = frame[:,:,2]
            gray_im = cv2.resize(frame, (256,256))
            _, filtered_bin_image = cv2.threshold(gray_im,70,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            eroded = cv2.erode(filtered_bin_image ,kernel,iterations = 1)
            dilation = cv2.dilate(eroded ,kernel,iterations = 1)

            inv_im = 255-dilation
            inv_im_dist = cv2.distanceTransform(inv_im, cv2.DIST_L2, 3)
            exp_inv_im_dist = dist_exp(inv_im_dist)
            exp_im_dist = 255-exp_inv_im_dist
            frame = np.stack((exp_im_dist, exp_im_dist, exp_im_dist), axis=-1)
        """
    if opts.save_blurr_and_converted:
        cv2.imwrite('im_log/converted_{}.jpg'.format(obj.im_num), cv2.resize(frame, (256,256)))
        cv2.imwrite('im_log/blurred_{}.jpg'.format(obj.im_num), cv2.resize(blurred_frame, (256,256)))

    if opts.render:
        conv_frame = cv2.resize(frame, (256,256)) 
        orig_ata = cv2.resize(orig_ata, (256,256)) 
        if opts.blurr:
            conv_frame = rgb2gray(conv_frame)*255.

        data = np.concatenate((orig_ata, conv_frame), axis=1)
        data = cv2.resize(data, (256*2, 256)).astype(int)
        #cv2.imwrite('im_log/{}.jpg'.format(im_num), data)
        #st()
        im=Image.frombytes('L', (256*2,256), data.astype('b').tostring())
        photo = ImageTk.PhotoImage(image=im)
        canvas.create_image(0,0,image=photo,anchor=tk.NW)
        master.update()
        data=np.roll(data,-1,1)

    if frame.shape[-1] != 3:
        dilation = np.stack((dilation, dilation,dilation), axis=-1)
        frame = np.stack((frame, frame, frame), axis=-1)
        
    dilation = rgb2gray(dilation)
    dilation = cv2.resize(dilation, (80, conf["dimension2"]))
    #dilation = cv2.resize(rgb2gray(dilation), (80, conf["dimension2"]))
    dilation = cv2.resize(dilation, (80, 80))
    dilation = np.reshape(dilation, [1, 80, 80])

    frame = rgb2gray(frame)
    frame = cv2.resize(frame, (80, conf["dimension2"]))
    #frame = cv2.resize(rgb2gray(frame), (80, conf["dimension2"]))
    frame = cv2.resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])

    old_frame = cv2.resize(old_frame, (80, 80))
    state_to_save = old_frame

    old_state = rgb2gray(blurred_frame)
    old_state = cv2.resize(old_state, (80, conf["dimension2"]))
    old_state = cv2.resize(old_state, (80, 80))
    old_state = np.reshape(old_state, [1, 80, 80])
    obj.im_num += 1

    #frame = np.stack((frame, dilation))
    frame = np.stack((frame, old_state))
    return frame

_process_frame = _atten_process_frame
#_process_frame = _orig_process_frame

class AtariRescale(vectorized.ObservationWrapper):
    def __init__(self, env, env_conf, convertor, transformation, flann , config, opts):
        super(AtariRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [2, 1, 80, 80])
        self.conf = env_conf
        self.convertor = convertor
        self.transformation = transformation
        self.im_num = 0
        self.flann = flann
        self.config = config
        self.opts = opts

    def _observation(self, observation_n):
        return [
                _process_frame(observation, self.conf, self.convertor, self, self.config, self.opts)
                for observation in observation_n
                ]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = random.randrange(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
        return obs


class NormalizedEnv(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = np.array([0, 0]).reshape((2,1,1,1))
        self.state_std = np.array([0, 0]).reshape((2,1,1,1))
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                    observation.mean(axis=(1,2,3), keepdims=True) * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                    observation.std(axis=(1,2,3), keepdims=True) * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8)
                for observation in observation_n]
