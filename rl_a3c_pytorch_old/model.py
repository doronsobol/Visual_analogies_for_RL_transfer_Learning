from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normalized_columns_initializer, weights_init, norm_col_init
from pdb import set_trace as st

class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def convs(self, inputs):
        c1 = self.conv1(inputs)
        r1 = F.relu(F.max_pool2d(c1, kernel_size=2, stride=2))
        c2 = self.conv2(r1)
        r2 = F.relu(F.max_pool2d(c2, kernel_size=2, stride=2))
        c3 = self.conv3(r2)
        r3 = F.relu(F.max_pool2d(c3, kernel_size=2, stride=2))
        c4 = self.conv4(r3)
        return [c3.view(c3.size(0),-1),c4.view(c3.size(0),-1)]

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        c1 = self.conv1(inputs)
        r1 = F.relu(F.max_pool2d(c1, kernel_size=2, stride=2))
        c2 = self.conv2(r1)
        r2 = F.relu(F.max_pool2d(c2, kernel_size=2, stride=2))
        c3 = self.conv3(r2)
        r3 = F.relu(F.max_pool2d(c3, kernel_size=2, stride=2))
        c4 = self.conv4(r3)
        r4 = F.relu(F.max_pool2d(c4, kernel_size=2, stride=2))

        x = r4.view(r4.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx), x, (c3.view(c3.size(0),-1),c4.view(c3.size(0),-1))


class lstm_A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(static_A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.lstm = nn.LSTM(1024, 512, batch_first=True)
        num_outputs = action_space
        #self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        #self.actor_linear.weight.data = normalized_columns_initializer(
        #    self.actor_linear.weight.data, 0.01)
        #self.actor_linear.bias.data.fill_(0)
        #self.critic_linear.weight.data = normalized_columns_initializer(
        #    self.critic_linear.weight.data, 1.0)
        #self.critic_linear.bias.data.fill_(0)

        self.num_outputs = num_outputs

        self.train()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_len = inputs.size(1)

        inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
        x = F.relu(F.max_pool2d(self.conv1(inputs), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
        x = x.view(batch_size, time_len, -1)

        output, (hx, cx) = self.lstm(x)
        actions = self.actor_linear(output)

        out = actions.view(-1, self.num_outputs)
        return out

class static_A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(static_A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        #self.actor_linear.weight.data = normalized_columns_initializer(
        #    self.actor_linear.weight.data, 0.01)
        #self.actor_linear.bias.data.fill_(0)
        #self.critic_linear.weight.data = normalized_columns_initializer(
        #    self.critic_linear.weight.data, 1.0)
        #self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.num_outputs = num_outputs

        self.train()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        sec_len = inputs.size(1)
        
        cx = Variable(torch.zeros(batch_size, 512)).cuda()
        hx = Variable(torch.zeros(batch_size, 512)).cuda()

        logits = []
        features = []
        for i in range(sec_len):
            x = F.relu(F.max_pool2d(self.conv1(inputs[:,i, :, :, :]), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
            x = x.view(x.size(0), -1)
            hx, cx = self.lstm(x, (hx, cx))
            actions = self.actor_linear(hx)
            logits += [actions]
            features += [hx]

        out = torch.stack(logits, dim=1)
        features = torch.stack(features, dim=1)
        out = out.view(-1, self.num_outputs)
        features = features.view(-1, 512)
        return out, features
