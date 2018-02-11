from __future__ import division
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')
import torch
import torch.nn.functional as F
#from old_environment import atari_env
from environment import atari_env
from utils import setup_logger, get_translator, get_num_of_actions
from model import A3Clstm
from torch.autograd import Variable
import time
import logging
from logger import Logger
from net_config import *
from utils import read_config, setup_logger
from trainers import *
import trainers
from tools import *


def test(args, shared_model, env_conf, counter, convertor, convertor_config):
    log = {}
    logger = Logger("{}/{}".format(args.log_dir, args.experiment_name))
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)

    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    #cuda_device = 0 if 1 == args.gpu else 1
    cuda_device = 0
    #env = atari_env(args.env, env_conf)

    if args.per_process_convertor:
        convertor = None

    env = atari_env("{}".format(args.env), env_conf, convertor, None, convertor_config, args)
    if 'Tennis' in args.env:
        env.reset()
        env.unwrapped.restore_state(np.load('Tennis.npy'))
    model = A3Clstm(env.observation_space.shape[1],  env.action_space.n)
    if args.cuda:
        model = model.cuda(cuda_device)
    model.eval()
    translator = get_translator(args)
    state = env.reset()
    transfered_state = state[0,:,:,:]
    state = state[1,:,:,:]
    state = torch.from_numpy(state).float()
    transfered_state = torch.from_numpy(transfered_state).float()
    reward_sum = 0
    done = True
    start_time = time.time()
    episode_length = 0
    num_tests = 0
    best_score = float('-inf')
    reward_total_sum = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            if 'Tennis' in args.env:
                env.reset()
                env.unwrapped.restore_state(np.load('Tennis.npy'))
            model.load_state_dict(shared_model.state_dict())
            if args.cuda:
                cx = Variable(torch.zeros(1, 512), volatile=True).cuda(cuda_device)
                hx = Variable(torch.zeros(1, 512), volatile=True).cuda(cuda_device)
            else:
                cx = Variable(torch.zeros(1, 512), volatile=True)
                hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        if args.cuda:
            var_state = Variable(state.unsqueeze(0),  volatile=True).cuda(cuda_device)
        else:
            var_state = Variable(state.unsqueeze(0),  volatile=True)

        value, logit, (hx, cx), _, _ = model((var_state, (hx, cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].cpu().data.numpy()

        #state, reward, done, _ = env.step(translator[action[0]])
        state, reward, done, _ = env.step(action)
        #done = done or episode_length >= args.max_episode_length
        done = done or episode_length >= args.max_episode_length or ('Tennis' in args.env and 0 != reward)
        reward_sum += reward

        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests

            num_of_steps = counter.value()//100
            logger.scalar_summary('episode_reward', reward_sum, num_of_steps)
            logger.scalar_summary('mean_reward', reward_mean, num_of_steps)
            logger.scalar_summary('episode_length', episode_length, num_of_steps)
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, episode_length, reward_mean))

            if reward_sum >= best_score:
                best_score = reward_sum
                model.load_state_dict(shared_model.state_dict())
                state_to_save = model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.env))

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            time.sleep(60)

        transfered_state = state[0,:,:,:]
        state = state[1,:,:,:]
        state = torch.from_numpy(state).float()
        transfered_state = torch.from_numpy(transfered_state).float()
