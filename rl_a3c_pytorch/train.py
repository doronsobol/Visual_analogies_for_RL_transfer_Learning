from __future__ import division, print_function
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')
from functools import reduce
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, get_translator, get_num_of_actions
from model import A3Clstm, A3Clstm_embeddings
from player_util import Agent
from torch.autograd import Variable
from net_config import *
from utils import read_config, setup_logger, get_translator_from_source
from trainers import *
import trainers
from tools import *


def train(rank, args, shared_model, optimizer, env_conf, counter, convertor, convertor_config, shared_teacher_model, teacher_optimizer, model_env_conf=None, do_transform=True):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    print("setting train {}/{} node on gpu: {}, conversion: {}".format(rank+1, args.workers,gpu_id, args.per_process_convertor))
    # Since pytorch has a problem with sharing GPU data we need to 
    # create the mapper model for each sub-process
    if args.per_process_convertor:
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
    else:
        distance_gan = None

    save_images = (rank == 0 or rank == args.workers) and args.save_images

    # If we need to convert the image and actions:
    if args.co_train_expantion and args.per_process_convertor:
        env = atari_env("{}".format(args.model_env), model_env_conf, convertor, None, convertor_config, args, save_images=save_images)
        env_id = args.model_env
        num_of_actions = env.action_space.n
        if do_transform:
            num_of_actions = atari_env("{}".format(args.env), env_conf, None, None, convertor_config, args).action_space.n 
    else:
        env = atari_env("{}".format(args.env), env_conf, convertor, None, convertor_config, args, save_images=save_images)
        env_id = args.env
        num_of_actions = env.action_space.n

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            if args.co_train:
                teacher_optimizer = RMSprop(shared_teacher_model.parameters(), lr=args.lr)
            optimizer = RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            if args.co_train:
                teacher_optimizer = Adam(shared_teacher_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
            optimizer = Adam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.env_id = env_id

    # If we want to transform the actions
    if args.co_train_expantion and args.per_process_convertor and do_transform:
        player.translate_test = True
        player.translator = get_translator_from_source(args.env, args.model_env)

    player.gpu_id = gpu_id
    # Experimental: using the embedding instead of the images
    if args.use_embeddings:
        player.model = A3Clstm_embeddings(num_of_actions)
    else:
        player.model = A3Clstm(
            player.env.observation_space.shape[1], num_of_actions)

    # Experimental
    if 0 != args.percp_loss:
        player.teacher_model = A3Clstm(env.observation_space.shape[1], get_num_of_actions(args))
        teacher_model_dict = player.teacher_model.state_dict()
        teacher_saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.model_env),
            map_location=lambda storage, loc: storage)
        teacher_saved_state = {k: v for k, v in teacher_saved_state.items() if 'conv' in k}
        teacher_model_dict.update(teacher_saved_state)
        player.teacher_model.load_state_dict(teacher_model_dict)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.teacher_model = player.teacher_model.cuda()
        player.teacher_model.eval()


    tmp_state = player.env.reset()
    # If the enviroment is Tennis, reset after every small game in order to align the sides.
    if 'Tennis' in env_id:
        player.env.reset()
        player.env.unwrapped.restore_state(np.load('Tennis.npy'))
    player.state = tmp_state[1,:,:,:]
    player.state = torch.from_numpy(player.state).float()
    transfered_state = tmp_state[0,:,:,:]
    player.transfered_state = torch.from_numpy(transfered_state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.transfered_state = player.transfered_state.cuda()
            player.model = player.model.cuda()
    player.model.train()

    if args.co_train:
        if args.use_convertor:
            num_of_actions = get_num_of_actions(args)
        else:
            num_of_actions = env.action_space.n
        player.teacher_model = A3Clstm(env.observation_space.shape[1], num_of_actions)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.teacher_model = player.teacher_model.cuda()
        player.teacher_model.train()


    player.eps_len += 2
    run_train = True
    while run_train:
        # Experimental train to networks with the same enviroment
        if args.co_train:
            player.co_train_prob = counter.value()/float(args.co_train_steps)
            if player.co_train_prob > 1:
                args.co_train = False
                player.args.co_train = False
                if args.per_process_convertor:
                    env = atari_env("{}".format(args.env), env_conf, None, None, convertor_config, args)
                    env.seed(args.seed + rank)
                    player.env = env
                    tmp_state = player.env.reset()
                    if 'Tennis' in env_id:
                        player.env.reset()
                        player.env.unwrapped.restore_state(np.load('Tennis.npy'))
                    player.state = tmp_state[1,:,:,:]
                    player.state = torch.from_numpy(player.state).float()
                    transfered_state = tmp_state[0,:,:,:]
                    player.transfered_state = torch.from_numpy(transfered_state).float()
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.state = player.state.cuda()
                            player.transfered_state = player.transfered_state.cuda()
            else:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.teacher_model.load_state_dict(shared_teacher_model.state_dict())
                else:
                    player.teacher_model.load_state_dict(shared_teacher_model.state_dict())
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        counter.increment(step)
        if args.co_train_expantion and args.per_process_convertor and 0 != args.pretrain_iterations and args.pretrain_iterations < counter.value():
            run_train = False
        if player.done:
            if player.info['ale.lives'] == 0 or player.max_length:
                player.eps_len = 0
            tmp_state = player.env.reset()
            if 'Tennis' in env_id:
                player.env.reset()
                player.env.unwrapped.restore_state(np.load('Tennis.npy'))
            state = tmp_state[1,:,:,:]
            #state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            transfered_state = tmp_state[0,:,:,:]
            player.transfered_state = torch.from_numpy(transfered_state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
                    player.transfered_state = player.transfered_state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data

        if args.co_train:
            teacher_R = torch.zeros(1, 1)
            if not player.done:
                if args.use_convertor:
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            var_transfered_state = Variable(player.transfered_state.cuda())
                    else:
                        var_transfered_state = Variable(player.transfered_state)
                    teacher_value, _, _, _, _ = player.teacher_model(
                        (var_transfered_state.unsqueeze(0), (player.teacher_hx, player.teacher_cx)))
                else:
                    teacher_value, _, _, _, _ = player.teacher_model(
                        (Variable(player.state.unsqueeze(0)), (player.teacher_hx, player.teacher_cx)))

                teacher_R = teacher_value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()
                if args.co_train:
                    teacher_R = teacher_R.cuda()

        player.values.append(Variable(R))
        if args.co_train:
            player.teacher_values.append(Variable(teacher_R))

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()

        if 0 != args.percp_loss: 
            loss = (1-args.percp_loss)*(policy_loss + 0.5 * value_loss) + args.percp_loss*player.percp_loss
        elif args.co_train:
            loss = player.co_train_prob*(policy_loss + 0.5 * value_loss) + (1-player.co_train_prob)*player.teacher_ce_loss
        else:
            loss = (policy_loss + 0.5 * value_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 100.0)
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()

        if args.co_train:
            teacher_policy_loss = 0
            teacher_value_loss = 0
            teacher_gae = torch.zeros(1, 1)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    teacher_gae = teacher_gae.cuda()
            teacher_R = Variable(teacher_R)
            for i in reversed(range(len(player.rewards))):
                teacher_R = args.gamma * teacher_R + player.rewards[i]
                teacher_advantage = teacher_R - player.teacher_values[i]
                teacher_value_loss = teacher_value_loss + 0.5 * teacher_advantage.pow(2)

                # Generalized Advantage Estimataion
                teacher_delta_t = player.rewards[i] + args.gamma * \
                    player.teacher_values[i + 1].data - player.teacher_values[i].data

                teacher_gae = teacher_gae * args.gamma * args.tau + teacher_delta_t

                teacher_policy_loss = teacher_policy_loss - \
                    player.teacher_log_probs[i] * \
                    Variable(teacher_gae) - 0.01 * player.teacher_entropies[i]

            player.teacher_model.zero_grad()

            teacher_loss = (teacher_policy_loss + 0.5 * teacher_value_loss)
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm(player.teacher_model.parameters(), 100.0)
            ensure_shared_grads(player.teacher_model, shared_teacher_model, gpu=gpu_id >= 0)

            teacher_optimizer.step()
        player.clear_actions()

