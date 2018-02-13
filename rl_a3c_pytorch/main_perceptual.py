from __future__ import print_function, division
import torch.multiprocessing as mp
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
sys.path.append('../src')
sys.path.append('../src/trainers')
sys.path.append('../src/tools')

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--a2b',type=int,help="1 for a2b and others for b2a",default=1)
parser.add_argument('--config',type=str,help="net configuration")
parser.add_argument('--weights',type=str,help="file location to the trained generator network weights")
parser.add_argument('--weights-1',type=str,help="file location to the secondary trained generator network weights")
parser.add_argument(
    '--model_env',
    default='Pong-v0',
    metavar='MENV',
    help='environment of the source')
#parser.add_argument(
#    '--convertor',
#    type=int,
#    default=1,
#    metavar='CON',
#    help='If should use convertor')
parser.add_argument(
    '--use-embeddings',
    action='store_true',
    help='Use embeddings of the image instead of the actual image (Experimental)')
parser.add_argument(
    '--use_convertor',
    action='store_true',
    help='If should use the mapper')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--percp-loss',
    type=float,
    default=0,
    metavar='T',
    help='If should use online perceptual loss')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--iteration-start',
    type=int,
    default=0,
    metavar='S',
    help='start iteration if continue training')
parser.add_argument(
    '--num_of_processes_on_first_gpu',
    type=int,
    default=8,
    metavar='W',
    help='Not in use')
parser.add_argument(
    '--test-workers',
    type=int,
    default=6,
    metavar='W',
    help='how many testing processes to use (default: 32)')
parser.add_argument(
    '--pre-workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use for the source (default: 32)')
parser.add_argument(
    '--pretrain_iterations',
    type=int,
    default=0,
    metavar='W',
    help='how many training iterations to use on the source')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment of the target domain')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--deterministic', 
    action='store_true',
    help='If to use the the enviroment with deterministic actions')
parser.add_argument(
    '--load-conv', 
    action='store_true',
    help='load a trained convolutions model')
parser.add_argument(
    '--load', 
    action='store_true',
    help='load a trained model')
parser.add_argument(
    '--save-score-level',
    type=int,
    default=20,
    metavar='SSL',
    help='reward score test evaluation must get higher than to save model')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--count-lives',
    default=False,
    metavar='CL',
    help='end of life is end of training episode.')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--load-model-file',
    default='trained_models/',
    metavar='LMD',
    help='file load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', 
    default='logs/', 
    metavar='LG', 
    help='folder to save logs')
parser.add_argument(
    '--render',
    action='store_true',
    help='Watch game as it being played')
parser.add_argument(
    '--blurr',
    action='store_true',
    help='if should blurr the frames')
parser.add_argument(
    '--per-process-convertor',
    action='store_true',
    help='Must be used when using the Mapper due to a bug in the memory sharing of pytorch')
parser.add_argument(
    '--co-train',
    action='store_true',
    help='Train the source with the target')
parser.add_argument(
    '--save_images',
    action='store_true',
    help='Save some of the images (for debuging purpuses)')
parser.add_argument(
    '--cuda',
    action='store_true',
    help='If should use cuda')
parser.add_argument(
    '--experiment-name', 
    default='name', 
    metavar='LG', 
    help='The name of the experiment')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--co-train-expantion',
    action='store_true',
    help='Pretrain the converted source')
parser.add_argument(
    '--co-train-steps',
    type=int,
    default=100000,
    metavar='SR',
    help='Number of steps to train the model on both the source and the target')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')

class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('l', initval)
        self.lock = Lock()

    def increment(self, inc=1):
        with self.lock:
            self.val.value += inc

    def value(self):
        with self.lock:
            return self.val.value


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    mp.set_start_method('spawn')
    import torch
    from torch.multiprocessing import Process, Lock, Value
    from environment import atari_env
    from utils import read_config
    from model import A3Clstm, A3Clstm_embeddings
    from train import train
    from test import test
    from shared_optim import SharedRMSprop, SharedAdam
    import time
    from utils import read_config, setup_logger, get_num_of_actions
    from trainers import *
    from tools import *
    from net_config import *
    from logger import Logger
    parser.add_argument('--gpu',type=int,help="gpu id",default=torch.cuda.device_count())
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)


    setup_json = read_config(args.env_config)
    model_env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.model_env:
            model_env_conf = setup_json[i]
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, opts=args)
    if args.use_embeddings:
        shared_model = A3Clstm_embeddings(env.action_space.n)
    else:
        shared_model = A3Clstm(env.observation_space.shape[1], env.action_space.n)

    if args.load:
        model_dict = shared_model.state_dict()

        pretrained_dict = torch.load(
            '{0}'.format(args.load_model_file), map_location=lambda storage, loc: storage)

        # 1. filter out unnecessary keys
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'critic' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict

        shared_model.load_state_dict(model_dict)
    if args.load_conv:
        model_dict = shared_model.state_dict()

        pretrained_dict = torch.load(
            '{0}'.format(args.load_model_file), map_location=lambda storage, loc: storage)

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv' in k}
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'critic' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict

        shared_model.load_state_dict(model_dict)
    shared_model.share_memory()

    if args.co_train:
        if args.use_convertor:
            num_of_actions = get_num_of_actions(args)
            state_path = '{0}{1}.dat'.format(args.load_model_dir, args.model_env)
        else:
            num_of_actions = env.action_space.n
            state_path = args.load_model_file

        shared_teacher_model = A3Clstm(env.observation_space.shape[1], num_of_actions)
        teacher_model_dict = shared_teacher_model.state_dict()
        teacher_saved_state = torch.load(state_path,
            map_location=lambda storage, loc: storage)
        #teacher_saved_state = {k: v for k, v in teacher_saved_state.items() if 'conv' in k}
        teacher_model_dict.update(teacher_saved_state)
        shared_teacher_model.load_state_dict(teacher_model_dict)
        shared_teacher_model.share_memory()
    else:
        shared_teacher_model = None

    if args.use_convertor and not args.per_process_convertor:
        convertor_config = NetConfig(args.config)
        hyperparameters = {}
        for key in convertor_config.hyperparameters:
            exec ('hyperparameters[\'%s\'] = convertor_config.hyperparameters[\'%s\']' % (key,key))

        trainer = []
        exec ("trainer=%s(hyperparameters)" % hyperparameters['trainer'])
        pretrained_gen = torch.load(
            '{0}'.format(args.weights), map_location=lambda storage, loc: storage)
        trainer.gen.load_state_dict(pretrained_gen)
        trainer.gen.eval()
        if args.cuda:
            trainer.cuda(0)
        if not args.per_process_convertor:
            trainer.share_memory() 
        distance_gan = trainer
    else:
        distance_gan = None
        convertor_config = None

    torch.set_default_tensor_type('torch.FloatTensor')

    done = True
    convertor = distance_gan

    env = atari_env("{}".format(args.env), env_conf, convertor, None, convertor_config, args)
    #convertor = convertor if args.convertor == 1 else None

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            if args.co_train:
                teacher_optimizer = SharedRMSprop(shared_teacher_model.parameters(), lr=args.lr)
                teacher_optimizer.share_memory()
            else:
                teacher_optimizer = None
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            if args.co_train:
                teacher_optimizer = SharedAdam(shared_teacher_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
                teacher_optimizer.share_memory()
            else:
                teacher_optimizer = None
            optimizer = SharedAdam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None
        teacher_optimizer = None

    processes = []
    counter = Counter(0)
    counter_1 = Counter(args.iteration_start)

    
    if args.co_train_expantion:
        p = Process(target=test, args=(0 ,args, shared_model, env_conf, counter, convertor, convertor_config, counter_1, True, model_env_conf, None))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for rank in range(0, args.workers):
        p = Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, counter_1, convertor, convertor_config, shared_teacher_model, teacher_optimizer, model_env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    if args.pretrain_iterations > 0:
        for job in processes:
            job.join()
        processes = []
        print("finished pretrain!!!!!!")

    if not args.co_train_expantion:
        print('replacing counters')
        counter = counter_1

    use_secondary = False
    if args.use_embeddings:
        env_conf, model_env_conf = model_env_conf, env_conf
        args.env, args.model_env = args.model_env, args.env
        use_secondary = True
        args.a2b = args.a2b^1
        args.weights = args.weights_1

    logger_queue = mp.Queue()
    for i in range(args.test_workers):
        p = Process(target=test, args=(i, args, shared_model, env_conf, counter, convertor, convertor_config, counter_1, use_secondary, model_env_conf, logger_queue, not args.use_embeddings))
        p.start()
        processes.append(p)
        time.sleep(10)
    if args.co_train_expantion:
        for rank in range(0, args.pre_workers):
            if not args.use_embeddings:
                args.per_process_convertor = False
            p = Process(
                    target=train, args=(rank, args, shared_model, optimizer, env_conf, counter, convertor, convertor_config, shared_teacher_model, teacher_optimizer, model_env_conf, not args.use_embeddings))
            p.start()
            processes.append(p)
            time.sleep(0.1)
    logger = Logger("{}/{}".format(args.log_dir, args.experiment_name))
    while True:
        log = logger_queue.get()
        logger.scalar_summary(log[0], log[1], log[2])
