from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from random import random
import numpy as np
from utils import get_translator


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.teacher_model = None
        self.env = env
        self.current_life = 0
        self.state = state
        self.state_transfer = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.max_length = False
        self.percp_criterion = torch.nn.L1Loss()
        self.percp_loss = 0
        self.player_1_r = 0
        self.player_2_r = 0
        self.co_train_prob = 1
        self.teacher_ce_loss = 0
        self.teacher_entropies = []
        self.teacher_values = []
        self.teacher_log_probs = []
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.translator = get_translator(args)
        self.translate_test = False
        self.env_id = ''

    def soft_cross_entropy(self, logits, labels):
        log_softmax = F.log_softmax(logits, dim = 1)
        cross_entropy = torch.sum(-(labels * log_softmax), dim=1)
        return torch.mean(cross_entropy)

    def action_train(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(1, 512).cuda())
                    self.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                self.cx = Variable(torch.zeros(1, 512))
                self.hx = Variable(torch.zeros(1, 512))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)
        if self.args.co_train:
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.teacher_cx = Variable(torch.zeros(1, 512).cuda())
                        self.teacher_hx = Variable(torch.zeros(1, 512).cuda())
                else:
                    self.teacher_cx = Variable(torch.zeros(1, 512))
                    self.teacher_hx = Variable(torch.zeros(1, 512))
            else:
                self.teacher_cx = Variable(self.teacher_cx.data)
                self.teacher_hx = Variable(self.teacher_hx.data)
        value, logit, (self.hx, self.cx), _, convs = self.model(
            (Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))

        if self.args.co_train:
            if self.args.use_convertor:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        var_transfered_state = Variable(self.transfered_state.cuda())
                else:
                    var_transfered_state = Variable(self.transfered_state.unsqueeze(0))
            else:
                    var_transfered_state = Variable(self.state.unsqueeze(0))
            teacher_value, teacher_logit, (self.teacher_hx, self.teacher_cx), _, teacher_convs = self.teacher_model(
                (var_transfered_state, (self.teacher_hx, self.teacher_cx)))

        if 0 != self.args.percp_loss:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    var_transfered_state = Variable(self.transfered_state.cuda())
            else:
                var_transfered_state = Variable(self.transfered_state)
            transfered_convs = map(lambda x: x.detach(), self.teacher_model.convs(var_transfered_state.unsqueeze(0)))
            self.percp_loss += reduce(lambda x, y: x+y, map(lambda x: self.percp_criterion(x[0], x[1]), zip(convs, transfered_convs)))

        if self.args.co_train:
            teacher_prob = F.softmax(teacher_logit, dim=1)
            teacher_log_prob = F.log_softmax(teacher_logit, dim=1)
            teacher_entropy = -(teacher_log_prob * teacher_prob).sum(1)
            self.teacher_entropies.append(teacher_entropy)
            teacher_action = teacher_prob.multinomial().data
            teacher_log_prob = teacher_log_prob.gather(1, Variable(teacher_action))

            max_teacher_action = self.translator[teacher_prob.max(1)[1].data.cpu().numpy()[0]]
            max_teacher_action = Variable(torch.LongTensor([max_teacher_action]))
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    max_teacher_action = max_teacher_action.cuda()

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial().data
        log_prob = log_prob.gather(1, Variable(action))


        
        if self.args.co_train and self.co_train_prob < 1:
            self.teacher_ce_loss = self.cross_entropy(logit, max_teacher_action)
            if random() > self.co_train_prob:
                action = self.translator[teacher_action.cpu().numpy()[0][0]]
            else:
                action = action.cpu().numpy()
        else:
            action = action.cpu().numpy()

        if self.translate_test:
            action = self.translator[action[0][0]]
        tmp_state, self.reward, self.done, self.info = self.env.step(action)

        state = tmp_state[1,:,:,:]
        self.state = torch.from_numpy(state).float()
        transfered_state = tmp_state[0,:,:,:]
        self.transfered_state = torch.from_numpy(transfered_state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.transfered_state = self.transfered_state.cuda()
                self.state = self.state.cuda()
        self.eps_len += 1
        if 'Tennis' in self.env_id and 0 != self.reward:
            if self.reward == 1:
                self.player_1_r += 1
            elif self.reward == -1:
                self.player_2_r += 1
            if (self.player_1_r >= 4 or self.player_2_r >= 4) and np.absolute(self.player_2_r - self.player_1_r) > 1:
                self.done = True
                self.player_2_r = 0
                self.player_1_r = 0
        if self.eps_len >= self.args.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False
        self.reward = max(min(self.reward, 1), -1)
        if self.args.co_train:
            self.teacher_values.append(teacher_value)
            self.teacher_log_probs.append(teacher_log_prob)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        if self.done:
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.cx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
                    self.hx = Variable(torch.zeros(
                        1, 512).cuda(), volatile=True)
            else:
                self.cx = Variable(torch.zeros(1, 512), volatile=True)
                self.hx = Variable(torch.zeros(1, 512), volatile=True)
        else:
            self.cx = Variable(self.cx.data, volatile=True)
            self.hx = Variable(self.hx.data, volatile=True)
        value, logit, (self.hx, self.cx), _, _ = self.model(
            (Variable(self.state.unsqueeze(0), volatile=True), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()

        if self.translate_test:
            action = [self.translator[action[0]]]
        tmp_state, self.reward, self.done, self.info = self.env.step(action[0])
        state = tmp_state[1,:,:,:]
        self.state = torch.from_numpy(state).float()
        transfered_state = tmp_state[0,:,:,:]
        self.transfered_state = torch.from_numpy(transfered_state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
                self.transfered_state = self.transfered_state.cuda()
        self.eps_len += 1
        if 'Tennis' in self.env_id and 0 != self.reward:
            if self.reward == 1:
                self.player_1_r += 1
            elif self.reward == -1:
                self.player_2_r += 1
            if (self.player_1_r >= 4 or self.player_2_r >= 4) and np.absolute(self.player_2_r - self.player_1_r) > 1:
                self.done = True
                self.player_2_r = 0
                self.player_1_r = 0
        if self.eps_len >= self.args.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False
        return self


    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.teacher_entropies = []
        self.teacher_values = []
        self.teacher_log_probs = []
        self.percp_loss = 0
        self.player_1_r = 0
        self.player_2_r = 0
        self.teacher_ce_loss = 0
        return self

