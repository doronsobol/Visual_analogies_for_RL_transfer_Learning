from __future__ import division, print_function
import time
import os
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')
#from cyclegan_based_options.test_options import TestOptions

from torch.multiprocessing import Process, Lock
from environment_for_render import atari_env
#from environment_new import atari_env
import environment_for_render as environment
#import environment_new as environment
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
from trainers import *
import random
from pdb import set_trace as st
from tools import *
from logger import Logger
parser = argparse.ArgumentParser(description='A3C_EVAL')
#parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument('--a2b',type=int,help="1 for a2b and others for b2a",default=1)
parser.add_argument('--gpu',type=int,help="gpu id",default=0)
parser.add_argument('--config',type=str,help="net configuration")
parser.add_argument('--weights',type=str,help="file location to the trained generator network weights")
parser.add_argument('--random', action='store_true',help="if to use random sampling")
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--model_env',
    default='Pong-v0',
    metavar='MENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--use_convertor',
    default=False,
    metavar='ENV',
    help='If should use the mapper')
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
    '--save-blurr-and-converted',
    action='store_true',
    help='save some of the images')
parser.add_argument(
    '--transform-action',
    action='store_true',
    help='Use static transformation for the actions')
parser.add_argument(
    '--keep-images',
    action='store_true',
    help='Watch game as it being played')
parser.add_argument(
    '--prob-action',
    default=False,
    metavar='R',
    help='Play with probabiliy instead of maximum action')
parser.add_argument(
    '--use-embeddings',
    action='store_true',
    help='Use embeddings of the image instead of the actual image (Experimental)')
parser.add_argument(
    '--render',
    action='store_true',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='CON',
    help='Skip rate for the images (when using an enviroment with no skip)')
parser.add_argument(
    '--convertor',
    type=int,
    default=1,
    metavar='CON',
    help='If should use the mapper (1 if should)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
parser.add_argument(
    '--labels-file', 
    default='/tmpt/t.txt', 
    metavar='LG', 
    help='Location of the labels file to save')
parser.add_argument(
    '--images-dir', 
    default='assault_images',
    metavar='LG', 
    help='directory of the images to save (should be a relative address)')
parser.add_argument(
    '--blurr', 
    action='store_true', 
    help='If to use blurness (SHould Be true)')
parser.add_argument(
    '--cuda', 
    action='store_true', 
    help='If to use cuda')
parser.add_argument(
    '--experiment-name', 
    default='name', 
    help='The name of the experiment')
parser.add_argument(
    '--deterministic',
    action='store_true',
    help='If to use the the enviroment with deterministic actions')
parser.add_argument(
    '--use-orig-model',
    action='store_true',
    help='Use the model of the source model instead of that of the target model.')
args = parser.parse_args()




logger = Logger("{}/{}".format(args.log_dir, args.experiment_name))

args.nThreads = 1   # test code only supports nThreads=1
args.batchSize = 1  #test code only supports batchSize=1
args.serial_batches = True # no shuffle

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]
model_env = atari_env("{}".format(args.model_env), env_conf, None, opts=args, test=True)
#st()
if 'DemonAttack' in args.model_env:
    num_of_model_actions = 4
elif 'Assault' in args.model_env:
    num_of_model_actions = 5
elif 'Pong' in args.model_env:
    num_of_model_actions = 6
elif 'Breakout' in args.model_env:
    num_of_model_actions = 4
elif 'Tennis' in args.model_env:
    num_of_model_actions = 18
else:
    raise Exception("Cant translate model_enviroment: {}".format(args.model_env))

model = A3Clstm(model_env.observation_space.shape[1], num_of_model_actions)#model_env.action_space)
model.eval()
if args.use_convertor:
    convertor_config = NetConfig(args.config)
    hyperparameters = {}
    for key in convertor_config.hyperparameters:
        exec ('hyperparameters[\'%s\'] = convertor_config.hyperparameters[\'%s\']' % (key,key))

    trainer = []
    exec ("trainer=%s(hyperparameters)" % hyperparameters['trainer'])
    trainer.gen.load_state_dict(torch.load(args.weights))
    trainer.cuda(args.gpu)
    distance_gan = trainer
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
print('loading from {0}{1}.dat'.format(args.load_model_dir, args.model_env))
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
env = atari_env("{}".format(args.env), env_conf, convertor, None, convertor_config, args)
convertor = convertor if args.convertor == 1 else None

#env = gym.wrappers.Monitor(env, "{}_monitor".format(args.env), force=True, video_callable=lambda x: True)
num_tests = 0
reward_total_sum = 0
print(env.unwrapped.get_action_meanings())
#st()
user_play = False


if args.use_orig_model:
    orig_model = A3Clstm(model_env.observation_space.shape[1], env.action_space.n)
    orig_model.eval()
    print('loading from {0}{1}.dat'.format(args.load_model_dir, args.env))
    saved_state_orig = torch.load(
        '{0}{1}.dat'.format(args.load_model_dir, args.env),
        map_location=lambda storage, loc: storage)


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
im_num = 0

with open(args.labels_file, 'w') as labels_file, open(args.labels_file + 'conv.txt', 'w') as c_labels_file:
    for i_episode in tqdm(range(args.num_episodes)):
        if 'Tennis' in args.env:
            print('restarting tennis')
            done = True
            state = env.reset()
            env.unwrapped.restore_state(np.load('Tennis.npy'))
        else:
            state = env.reset()
        episode_length = 0
        reward_sum = 0
        first = True
        restart = True
        num_life = 5
        for _ in tqdm(iter(int, 1)):
            #if args.render:
            #    if i_episode % args.render_freq == 0:
            #        env.render()
            if done:
                model.load_state_dict(saved_state)
                cx = Variable(torch.zeros(1, 512), volatile=True)
                hx = Variable(torch.zeros(1, 512), volatile=True)
                if args.use_orig_model:
                    orig_model.load_state_dict(saved_state_orig)
                    orig_cx = Variable(torch.zeros(1, 512), volatile=True)
                    orig_hx = Variable(torch.zeros(1, 512), volatile=True)
            else:
                cx = Variable(cx.data, volatile=True)
                hx = Variable(hx.data, volatile=True)
                if args.use_orig_model:
                    orig_cx = Variable(orig_cx.data, volatile=True)
                    orig_hx = Variable(orig_hx.data, volatile=True)


            im_to_save = environment.state_to_save
            #st()
            if args.keep_images:
                cv2.imwrite('{0}/{1:07d}.jpg'.format(args.images_dir,im_num), im_to_save)

            unmodified_state = state

            state = torch.from_numpy(state[0,:,:,:]).float()
            value, logit, (hx,cx), features, convs = model((Variable(
                state.unsqueeze(0), volatile=True), (hx,cx)))

            #value, logit = model(Variable(
            #    state.unsqueeze(0), volatile=True))
            prob = F.softmax(logit)
            if args.keep_images:
                #print('{0}/{1:07d}.jpg: {2}'.format(args.images_dir,im_num ," ".join([str(i) for i in features.data.squeeze()])), file=f_labels_file)
                print('{0}/{1:07d}.jpg: {2}'.format(args.images_dir,im_num ," ".join([str(i) for i in prob.data.squeeze()])), file=labels_file)
                #np.save('{0}/{1:07d}_1.npy'.format(args.images_dir,im_num) , convs[-2].data.numpy().flatten())
                #np.save('{0}/{1:07d}_2.npy'.format(args.images_dir,im_num) , convs[-1].data.numpy().flatten())
                #print('{0}/{1:07d}.jpg: {0}/{1:07d}_1.npy: {0}/{1:07d}_2.npy'.format(args.images_dir,im_num), file=c_labels_file)
                #print('{0}/{1:07d}.jpg: {0}/{1:07d}.npz'.format(args.images_dir,im_num) , file=c_labels_file)
                #np.savez('{0}/{1:07d}.npz'.format(args.images_dir,im_num), *convs[-2:])
            im_num += 1
            
            if args.prob_action:
                action = prob.multinomial().data.numpy()[0]
            else:
                action = prob.max(1)[1].data.numpy()

            if args.use_orig_model:
                prev_action = action
                orig_state = torch.from_numpy(unmodified_state[1,:,:,:]).float()
                _, orig_logit, (orig_hx,orig_cx), _, _ = orig_model((Variable(
                    orig_state.unsqueeze(0), volatile=True), (orig_hx,orig_cx)))
                orig_prob = F.softmax(orig_logit)
                action = orig_prob.max(1)[1].data.numpy()
                #print("action={}, playing: {}".format(model_env.unwrapped.get_action_meanings()[prev_action[0]], env.unwrapped.get_action_meanings()[action[0]]))
            #corrected_action = action[0] if action[0] < 4 else action[0] - 2
            if 'DemonAttack' in args.env:
                translate = [0, 1, 1,2,3]
            elif 'Tennis' in args.env and 'Tennis' in args.model_env or\
                    'Tennis' in args.env and args.use_orig_model:
                translate = [0, 1, 0, 3, 4, 0, 3, 4, 3, 4, 1, 11, 12, 1, 11, 12, 11, 12]
            elif 'Assault' in args.env:
                translate = [0, 2, 3,4]
            elif 'Breakout' in args.env and 'Tennis' in args.model_env:
                translate = [0 ,1 , 0, 2, 3, 0, 2, 3, 2, 3, 1, 2, 3, 1 , 2, 3, 2, 3]
            elif 'Pong' in args.env and 'Tennis' in args.model_env:
                translate = [0 ,1 , 0, 2, 3, 0, 2, 3, 2, 3, 1, 4, 5, 1 , 4, 5, 4, 5]
            elif 'Pong' in args.env:
                translate = [0,1,4,5]
            elif 'Breakout' in args.env:
                translate = [1,1,2,3, 2, 3]
            elif 'Tennis' in args.env:
                translate = [1,1,3,4, 3, 4]
            else:
                raise Exception("Cant translate enviroment: {}".format(args.env))
            #print action[0]
            if args.transform_action:
                #print("orig_action: {}".format(model_env.unwrapped.get_action_meanings()[action[0]]))
                corrected_action = translate[action[0]]
            else:
                corrected_action = action[0]
            if restart:
                first = True
                restart = False
                corrected_action = 1
            if first:
                first = False
                corrected_action = 1
                second = True
            if second:
                second = False
                corrected_action = 1
            if args.random  and np.random.choice([False, False, False, True]):
                corrected_action = random.choice(np.unique(translate))
            
                #st()
            if user_play:
                c = getch()
                corrected_action = int(c)
                corrected_action = corrected_action % 4
                #print 
            #print env.unwrapped.get_action_meanings()[corrected_action]

            #print env.unwrapped.get_action_meanings()[corrected_action]
            #print("corrected_action: {}".format(env.unwrapped.get_action_meanings()[corrected_action]))

            state, reward, done, info = env.step(corrected_action)
            logger.scalar_summary('value', value.data.numpy()[0,0], im_num)
            #print('it: {}, value: {}, action: {}, reward: {}'.format(im_num, value.data.numpy()[0,0], env.unwrapped.get_action_meanings()[corrected_action], reward))
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
            if 'Tennis' in args.env and reward != 0:
                done = True
                env.unwrapped.restore_state(np.load('Tennis.npy'))
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
