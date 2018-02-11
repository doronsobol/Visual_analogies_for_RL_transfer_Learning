from __future__ import division, print_function
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')
from functools import reduce
import torch
import torch.nn.functional as F
import torch.optim as optim
#from old_environment import atari_env
from environment import atari_env
from utils import ensure_shared_grads, get_translator, get_num_of_actions
from model import A3Clstm
from torch.autograd import Variable
from net_config import *
from utils import read_config, setup_logger
from trainers import *
import trainers
from tools import *


def train(rank, args, shared_model, optimizer, env_conf, counter, convertor, convertor_config):
    torch.manual_seed(args.seed + rank)

    #env = atari_env(args.env, env_conf)
    #curr_gpu = (rank+2)%args.gpu
    #curr_gpu = 0 if 1 == args.gpu else (rank%(args.gpu-1)+1)
    curr_gpu = 0 if rank < args.num_of_processes_on_first_gpu else 1
    #curr_gpu = 0 if 1 == args.gpu else (rank%(args.gpu-1)+1)
    print("setting train {}/{} node on gpu: {}".format(rank+1, args.workers,curr_gpu))
    if args.per_process_convertor:
        convertor_config = NetConfig(args.config)
        hyperparameters = {}
        for key in convertor_config.hyperparameters:
            hyperparameters[str(key)] = convertor_config.hyperparameters[str(key)]
            #exec ('hyperparameters[\'%s\'] = convertor_config.hyperparameters[\'%s\']' % (key,key))

        trainer = getattr(trainers, hyperparameters['trainer'])(hyperparameters)

        #trainer = []
        #print("trainer=%s(hyperparameters)" % hyperparameters['trainer'])
        pretrained_gen = torch.load(
            '{0}'.format(args.weights), map_location=lambda storage, loc: storage)
        trainer.gen.load_state_dict(pretrained_gen)
        if args.cuda:
            trainer.cuda(curr_gpu)
        distance_gan = trainer
        convertor = distance_gan
    else:
        distance_gan = None

    env = atari_env("{}".format(args.env), env_conf, convertor, None, convertor_config, args)
    if 'Tennis' in args.env:
        env.reset()
        env.unwrapped.restore_state(np.load('Tennis.npy'))
    model = A3Clstm(env.observation_space.shape[1], env.action_space.n)
    if args.cuda:
        model = model.cuda(curr_gpu)

    if 0 != args.percp_loss:
        teacher_model = A3Clstm(env.observation_space.shape[1], get_num_of_actions(args))
        teacher_model_dict = teacher_model.state_dict()
        teacher_saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.model_env),
            map_location=lambda storage, loc: storage)
        teacher_saved_state = {k: v for k, v in teacher_saved_state.items() if 'conv' in k}
        teacher_model_dict.update(teacher_saved_state)
        teacher_model.load_state_dict(teacher_model_dict)
        if args.cuda:
            teacher_model = teacher_model.cuda(curr_gpu)
        teacher_model.eval()

    _ = env.reset()
    action = env.action_space.sample()
    _, _, _, info = env.step(action)
    start_lives = info['ale.lives']
    translator = get_translator(args)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()
    env.seed(args.seed + rank)
    state = env.reset()
    transfered_state = state[0,:,:,:]
    state = state[1,:,:,:]
    state = torch.from_numpy(state).float()
    transfered_state = torch.from_numpy(transfered_state).float()
    done = True
    episode_length = 0
    current_life = start_lives
    percp_criterion = torch.nn.L1Loss()
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            if args.cuda:
                cx = Variable(torch.zeros(1, 512)).cuda(curr_gpu)
                hx = Variable(torch.zeros(1, 512)).cuda(curr_gpu)
            else:
                cx = Variable(torch.zeros(1, 512))
                hx = Variable(torch.zeros(1, 512))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        percp_loss = 0

        for step in range(args.num_steps):
            if args.cuda:
                var_state = Variable(state.unsqueeze(0)).cuda(curr_gpu)
            else:
                var_state = Variable(state.unsqueeze(0))

            value, logit, (hx, cx), _,convs = model((var_state,
                                            (hx, cx)))
            if 0 != args.percp_loss:
                if args.cuda:
                    var_transfered_state = Variable(transfered_state.cuda(curr_gpu))
                else:
                    var_transfered_state = Variable(transfered_state)
                transfered_convs = map(lambda x: x.detach(), teacher_model.convs(var_transfered_state.unsqueeze(0)))
                percp_loss += reduce(lambda x, y: x+y, map(lambda x: percp_criterion(x[0], x[1]), zip(convs, transfered_convs)))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
            #state, reward, done, info = env.step(translator[action.cpu().numpy()[0, 0]])
            state, reward, done, info = env.step(action.cpu().numpy())
            done = done or episode_length >= args.max_episode_length or ('Tennis' in args.env and 0 != reward)
            if args.count_lives:
                if current_life > info['ale.lives']:
                    done = True
                else:
                    current_life = info['ale.lives']
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                current_life = start_lives
                state = env.reset()
                if 'Tennis' in args.env:
                    env.reset()
                    env.unwrapped.restore_state(np.load('Tennis.npy'))

            transfered_state = state[0,:,:,:]
            state = state[1,:,:,:]
            state = torch.from_numpy(state).float()
            transfered_state = torch.from_numpy(transfered_state).float()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        counter.increment(step+1)
        R = torch.zeros(1, 1)
        R = R.cuda(curr_gpu) if args.cuda else R
        if not done:

            var_state = Variable(state.unsqueeze(0)).cuda(curr_gpu) if args.cuda else  Variable(state.unsqueeze(0))
            value, _, _, _, _ = model((var_state, (hx, cx)))
            R = value.data

        R = Variable(R).cuda(curr_gpu) if args.cuda else Variable(R)
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        gae = gae.cuda(curr_gpu) if args.cuda else gae
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            gae_var = Variable(gae).cuda(curr_gpu) if args.cuda else Variable(gae)
            policy_loss = policy_loss - \
                log_probs[i] * gae_var - 0.01 * entropies[i]

        optimizer.zero_grad()

        loss = (1-args.percp_loss)*(policy_loss + 0.5 * value_loss) + args.percp_loss*percp_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
