from __future__ import division
import time
import os
import sys
sys.path.append('..')
#from cyclegan_based_options.test_options import TestOptions
from discogan_based_models.discogan_based_options.options import Options as TestOptions

from environment import atari_env
import argparse
import torch
import torch.nn.functional as F
from utils import read_config, setup_logger
from model import A3Clstm
from torch.autograd import Variable
import gym
from tqdm import tqdm
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pdb
opt = TestOptions()
opt.initialize()
parser = opt.parser
#parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--model_env',
    default='Pong-v0',
    metavar='MENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--use_convertor',
    default=False,
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
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
    metavar='CON',
    help='If should use convertor')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
#args = opt.parse()
args = opt.parser.parse_args()


from discogan_based_models.distance_gan_model import DistanceGAN
from pdb import set_trace as st
from util import html


args.nThreads = 1   # test code only supports nThreads=1
args.batchSize = 1  #test code only supports batchSize=1
args.serial_batches = True # no shuffle

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]
model_env = atari_env("{}".format(args.model_env), env_conf, None)
#st()
if 'DemonAttack' in args.model_env:
    num_of_model_actions = 4
elif 'Assault' in args.model_env:
    num_of_model_actions = 5
elif 'Pong' in args.model_env:
    num_of_model_actions = 6
elif 'Breakout' in args.model_env:
    num_of_model_actions = 4
else:
    raise Exception("Cant translate model_enviroment: {}".format(args.model_env))

model = A3Clstm(model_env.observation_space.shape[0], num_of_model_actions)#model_env.action_space)
model.eval()
if args.use_convertor:
    distance_gan = DistanceGAN(args)
    distance_gan.initialize()
else:
    distance_gan = None
#data_loader = CreateDataLoader(args)
#dataset = data_loader.load_data()
#r_p = data_loader.dataset_B.return_paths
#data_loader.dataset_B.return_paths = False
#b_images =  np.array([np.array(e) for e in data_loader.dataset_B])#.astype(np.float32)
#b_images = np.transpose(b_images, (0,2,3,1))
#data_loader.dataset_B.return_paths = r_p
#print ("max: {}".format(np.amax(b_images)))


#print("b_images.shape: {}".format(b_images.shape))
#print(type(b_images[0]))
#nn = NearestNeighbors(n_neighbors=1)
#nn.fit(b_images.reshape(b_images.shape[0], -1))
#print ("fits")


torch.set_default_tensor_type('torch.FloatTensor')
print 'loading from {0}{1}.dat'.format(args.load_model_dir, args.model_env)
saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.model_env),
    map_location=lambda storage, loc: storage)

done = True

convertor = distance_gan



log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

#env = atari_env("{}".format(args.env), env_conf)
env = atari_env("{}".format(args.env), env_conf, convertor, None, None, args)
convertor = convertor if args.convertor == 1 else None

env = gym.wrappers.Monitor(env, "{}_monitor".format(args.env), force=True, video_callable=lambda x: True)
num_tests = 0
reward_total_sum = 0
print env.unwrapped.get_action_meanings()
#st()
user_play = False



class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()










for i_episode in tqdm(range(args.num_episodes)):
    state = env.reset()
    episode_length = 0
    reward_sum = 0
    first = True
    restart = True
    num_life = 5
    while True:
        #if args.render:
        #    if i_episode % args.render_freq == 0:
        #        env.render()
        if done:
            model.load_state_dict(saved_state)
            cx = Variable(torch.zeros(1, 512), volatile=True)
            hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        state = torch.from_numpy(state).float()
        value, logit = model(Variable(
            state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)
        
        action = prob.max(1)[1].data.numpy()
        #corrected_action = action[0] if action[0] < 4 else action[0] - 2
        if 'DemonAttack' in args.env:
            translate = [0, 1, 1,2,3]
        elif 'Assault' in args.env:
            translate = [0, 2, 3,4]
        elif 'Pong' in args.env:
            translate = [0,1,4,5]
        elif 'Breakout' in args.env:
            translate = [0,1,2,3, 2, 3]
        elif 'Tennis' in args.env:
            translate = [1,1,11,12, 11, 12]
        else:
            raise Exception("Cant translate enviroment: {}".format(args.env))
        #print action[0]
        if args.use_convertor or args.env != args.model_env:
            corrected_action = translate[action[0]]
        else:
            corrected_action = action[0]
        #print env.unwrapped.get_action_meanings()[corrected_action]
        if restart:
            first = True
            restart = False
            corrected_action = 1
        if first:
            first = False
            corrected_action = 1

            #st()
        if user_play:
            c = getch()
            corrected_action = int(c)
            corrected_action = corrected_action % 4
            #print 

        #print env.unwrapped.get_action_meanings()[corrected_action]
        state, reward, done, info = env.step(corrected_action)
        if num_life > info['ale.lives']:
            num_life = info['ale.lives']
            restart = True
            #print "restart"
        else:
            restart = False
        #action = prob.max(1)[1].data.numpy()
        #state, reward, done, _ = env.step(action[0, 0])
        episode_length += 1
        reward_sum += reward
        done = done or episode_length >= args.max_episode_length
        if done:
            first = True
            num_life = 5
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(
                    reward_sum, reward_mean))

            break
