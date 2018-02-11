from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import logging

def get_translator(args):
    if 'DemonAttack' in args.env:
        translate = [0, 1, 1,2,3]
    elif 'Assault' in args.env:
        translate = [0, 2, 3,4, 3, 4]
        #translate = [0, 2, 3,4]
    elif 'Pong' in args.env:
        translate = [0,1,4,5]
    elif 'Breakout' in args.env:
        translate = [0,1,2,3, 2, 3]
    elif 'Tennis' in args.env:
        translate = [1,1,11,12, 11, 12]
    else:
        raise Exception("Cant translate enviroment: {}".format(args.env))
    return translate


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def get_num_of_actions(args):
    if 'DemonAttack' in args.env:
        num_of_model_actions = 5
    elif 'Assault' in args.env:
        num_of_model_actions = 6
        #num_of_model_actions = 4
    elif 'Pong' in args.env:
        num_of_model_actions = 4
    elif 'Breakout' in args.env:
        num_of_model_actions = 6
    elif 'Tennis' in args.env:
        num_of_model_actions = 18
    else:
        raise Exception("Cant translate model_enviroment: {}".format(args.model_env))
    return num_of_model_actions

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        if shared_param.get_device() != param.grad.get_device():
            shared_param._grad = param.grad.cuda(shared_param.get_device())
        else:
            shared_param._grad = param.grad


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
