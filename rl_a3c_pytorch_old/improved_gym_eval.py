from __future__ import division
import time
import os
import sys
import cv2
#from sklearn.neighbors import NearestNeighbors
#from pyflann import *
sys.path.append('..')
from import torch
import torch.nn.functional as F
from environment import atari_env
from utils import read_config, setup_logger
from model import A3Clstm
from torch.autograd import Variable
import gym
import logging

cyclegan_based_options.test_options import TestOptions
#opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
import argparse
import numpy as np
opt = TestOptions()

parser = opt.parser
parser.add_argument(
    '--model_env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to use for model (default: Pong-v0)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', 
    default='logs/', 
    metavar='LG', 
    help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--convertor',
    type=int,
    default=1,
    metavar='M',
    help='If should use convertor')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
args = opt.parse()

from data.data_loader import CreateDataLoader
from cyclegan_based_models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

torch.set_default_tensor_type('torch.FloatTensor')

args.nThreads = 1   # test code only supports nThreads=1
args.batchSize = 1  #test code only supports batchSize=1
args.serial_batches = True # no shuffle



#data_loader = CreateDataLoader(args)
#dataset = data_loader.load_data()
#r_p = data_loader.dataset_B.return_paths
#data_loader.dataset_B.return_paths = False
#b_images = torch.cat([torch.unsqueeze(e, 0) for e in data_loader.dataset_B]).numpy().astype(np.float32)
#data_loader.dataset_B.return_paths = r_p

#print("b_images.shape: {}".format(b_images.shape))
#print(type(b_images[0]))
#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#print "before:"
#flann = cv2.flann.Index(b_images.reshape(b_images.shape[0], -1), index_params)
#print "after:"
#flann = FLANN()
#params = flann.build_index(b_images.reshape(b_images.shape[0], -1), target_precision=0.9, log_level="info")
#nn = NearestNeighbors(n_neighbors=1)
#nn.fit(b_images.reshape(b_images.shape[0], -1))
#params = None 
#print ("fits")
model = create_model(args)
#visualizer = Visualizer(args)

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.model_env),
    map_location=lambda storage, loc: storage)

done = True

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

print("creating enviroment")
env = atari_env("{}".format(args.env), env_conf, model)#, nn, b_images)
print("created enviroment")
model = model if args.convertor == 1 else None
model_env = atari_env("{}".format(args.model_env), env_conf, model)
print("actoins are: {}".format(str(env.action_space)))
print("actoin space: {}".format(str(model_env.action_space)))
model = A3Clstm(env.observation_space.shape[0], model_env.action_space)
model.eval()

env = gym.wrappers.Monitor(env, "{}_monitor".format(args.env), force=True)
num_tests = 0
reward_total_sum = 0

for i_episode in range(args.num_episodes):
    state = env.reset()
    episode_length = 0
    reward_sum = 0
    first = True
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                env.render()
        if done:
            model.load_state_dict(saved_state)
            cx = Variable(torch.zeros(1, 512), volatile=True)
            hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        state = torch.from_numpy(state).float()
        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        corrected_action = action[0, 0] if action[0, 0] <= 4 else action[0, 0] - 2
        if first:
            corrected_action = 1
            first = False
        state, reward, done, _ = env.step(corrected_action)
        episode_length += 1
        reward_sum += reward
        done = done or episode_length >= args.max_episode_length
        if done:
            first = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(
                    reward_sum, reward_mean))

            break
