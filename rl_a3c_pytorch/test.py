from __future__ import division
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm, A3Clstm_embeddings
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from logger import Logger
from net_config import *
from utils import read_config, setup_logger
from trainers import *
import trainers
from tools import *
from utils import get_translator_from_source


def test(rank ,args, shared_model, env_conf, counter, convertor, convertor_config, counter_2=None, do_secondery=False, model_env_conf=None, logger_queue=None, do_transform=True):
    ptitle('Test Agent {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    log = {}
    if logger_queue is None:
        logger = Logger("{}/{}".format(args.log_dir, args.experiment_name))
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    if args.per_process_convertor and do_secondery:
        convertor_config = NetConfig(args.config)
        hyperparameters = {}
        for key in convertor_config.hyperparameters:
            hyperparameters[str(key)] = convertor_config.hyperparameters[str(key)]

        trainer = getattr(trainers, hyperparameters['trainer'])(hyperparameters)

        pretrained_gen = torch.load(
            '{0}'.format(args.weights), map_location=lambda storage, loc: storage)
        trainer.gen.load_state_dict(pretrained_gen)
        if gpu_id >= 0:
            trainer.cuda(gpu_id)
        distance_gan = trainer
        convertor = distance_gan
        if do_transform:
            num_of_actions = atari_env("{}".format(args.env), env_conf, None, None, convertor_config, args, test=True).action_space.n

        save_images=args.save_images
        save_images=False
        env = atari_env("{}".format(args.model_env), model_env_conf, convertor, None, convertor_config, args, test=True, save_images=save_images)
        env_id = args.model_env
        num_of_actions = env.action_space.n
        if do_transform:
            num_of_actions = atari_env("{}".format(args.env), env_conf, None, None, convertor_config, args, test=True).action_space.n
    else:
        distance_gan = None
        save_images=args.save_images
        save_images=False
        env = atari_env("{}".format(args.env), env_conf, None, None, convertor_config, args, test=True, save_images=save_images)
        num_of_actions = env.action_space.n
        env_id = args.env

    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    if do_secondery and do_transform:
        player.translate_test = True
        player.translator = get_translator_from_source(args.env, args.model_env)
    player.gpu_id = gpu_id
    if args.use_embeddings:
        player.model = A3Clstm_embeddings(num_of_actions)
    else:
        player.model = A3Clstm(
            player.env.observation_space.shape[1], num_of_actions)

    tmp_state = player.env.reset()
    if 'Tennis' in env_id:
        player.env.reset()
        player.env.unwrapped.restore_state(np.load('Tennis.npy'))
    player.env_id = env_id
    player.state = tmp_state[1,:,:,:]
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    run_test = True
    while run_test:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and player.info['ale.lives'] > 0 and not player.max_length:
            tmp_state = player.env.reset()
            if 'Tennis' in env_id:
                player.env.reset()
                player.env.unwrapped.restore_state(np.load('Tennis.npy'))
            state = tmp_state[1,:,:,:]
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.done or player.max_length:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            if do_secondery and do_transform:
                if args.co_train_expantion and args.per_process_convertor and 0 != args.pretrain_iterations and args.pretrain_iterations < counter_2.value():
                    run_test = False
                num_of_steps = counter_2.value()
                logger.scalar_summary('secondery_episode_reward', reward_sum, num_of_steps)
                logger.scalar_summary('secondery_mean_reward', reward_mean, num_of_steps)
                logger.scalar_summary('secondery_episode_length', player.eps_len, num_of_steps)
                log['{}_log'.format(args.env)].info(
                    "Secondery Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, {4}/{5}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean, counter_2.value(),args.co_train_steps))
            else:
                num_of_steps = counter.value()
                if logger_queue is None:
                    logger.scalar_summary('episode_reward', reward_sum, num_of_steps)
                    logger.scalar_summary('mean_reward', reward_mean, num_of_steps)
                    logger.scalar_summary('episode_length', player.eps_len, num_of_steps)
                    if counter_2 is not None:
                        logger.scalar_summary('counter_2', counter_2.value(), num_of_steps)
                else:
                    logger_queue.put(('episode_reward', reward_sum, num_of_steps))
                    logger_queue.put(('mean_reward', reward_mean, num_of_steps))
                    logger_queue.put(('episode_length',  player.eps_len, num_of_steps))
                    if counter_2 is not None:
                        logger_queue.put(('counter_2', counter_2.value(), num_of_steps))
                    
                    
                log['{}_log'.format(args.env)].info(
                    "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, {4}/{5}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean, num_of_steps, args.co_train_steps))

            state_to_save = player.model.state_dict()
            torch.save(state_to_save, '{0}{1}_last.dat'.format(
                args.save_model_dir, args.experiment_name))
            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.experiment_name))

            reward_sum = 0
            player.eps_len = 0
            tmp_state = player.env.reset()
            if 'Tennis' in env_id:
                player.env.reset()
                player.env.unwrapped.restore_state(np.load('Tennis.npy'))
            state = tmp_state[1,:,:,:]
            player.eps_len += 2

            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

