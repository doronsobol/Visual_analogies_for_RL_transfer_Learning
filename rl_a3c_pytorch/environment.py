from __future__ import division
import sys
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box
#from skimage.color import rgb2gray
from cv2 import resize
#from skimage.transform import resize
#from scipy.misc import imresize as resize
import random
import cv2
import itertools
import torch
from skimage.color import rgb2gray
import torchvision.transforms as transforms
import torchvision
from tools import *
import pdb
from torch.autograd import Variable

tennis_median_im = cv2.imread('median.jpg', 0)

def create_xy_image(width=256):
  coordinates = list(itertools.product(range(width), range(width)))
  arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / float((width/2))
  new_map = np.transpose(np.float32(arr), [2, 0, 1])
  #xy = Variable(torch.from_numpy(new_map), requires_grad=False)
  xy = torch.from_numpy(new_map).clone()
  return xy

#xy = create_xy_image().cuda()
xy = create_xy_image()
xy = xy.unsqueeze(0).expand(1, xy.size(0), xy.size(1), xy.size(2))
state_to_save = np.zeros((80,80,3))

def dist_exp(im):
    e_im = np.exp(-im/6.)
    e_im = 1.0-e_im
    new_im = e_im*255.0
    return new_im.astype(np.uint8)

transformations = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

def frame2attention(frame, conf, trainer, obj, config, opts, env_id):
    global tennis_median_im
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    old_frame = frame
    orig_ata = frame
    dilation = frame
    if trainer is not None or opts.blurr:
        frame = cv2.resize(frame, (256, 256))
        frame = rgb2gray(frame)*255.
        orig_ata = frame

        if 'Pong' in env_id:
            median_im = 87.
            frame = np.absolute((frame - median_im)).astype(np.uint8)
        elif 'Tennis' in env_id:
            #median_im = 95.
            frame = np.absolute((frame - tennis_median_im)).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        if False:
            _, filtered_bin_image = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)
        else:
            _, filtered_bin_image = cv2.threshold(frame,20,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation = cv2.dilate(filtered_bin_image ,kernel,iterations = 1)
        #imgplot = plt.imshow(dilation)
        #plt.show()
        #cv2.imwrite(im_path_C, cv2.resize(dilation, (64, 64)))
        if opts.blurr:
            inv_im = 255-dilation
            inv_im_dist = cv2.distanceTransform(inv_im, cv2.DIST_L2, 3)
            exp_inv_im_dist = dist_exp(inv_im_dist)
            exp_im_dist = 255-exp_inv_im_dist

            exp_inv_im_dist = dist_exp(inv_im_dist/2)
            exp_im_dist_1 = 255-exp_inv_im_dist

            #frame = np.stack((exp_im_dist, exp_im_dist_1, dilation), axis=-1)
            frame = np.stack((dilation, exp_im_dist_1, exp_im_dist), axis=-1)
        else:
            frame = np.stack((dilation, dilation,dilation), axis=-1)
    return frame, orig_ata, old_frame, dilation

def atari_env(env_id, env_conf, convertor=None, flann=None, config=None, opts=None, test=False, save_images=False):
    args = opts
    if args.deterministic:
        env_id = env_id.split('-')[0] + 'NoFrameskip-'+ env_id.split('-')[1]
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.skip_rate)
    if not test:
        env = EpisodicLifeEnv(env)
    elif 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireEpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    #env = AtariRescale(env, env_conf)
    if args.convert_reward and 'Pong' in env_id:
        env = BreakoutReward(env)
    transform = transforms.Compose(transformations)
    env = AtariRescale(env, env_conf, convertor, transform, flann, config, opts, save_images, env_id)
    if not args.use_embeddings:
        env = NormalizedEnv(env)
    return env

"""
def _process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, conf["dimension2"]))
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame
"""
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def _atten_process_frame(frame, conf, trainer, obj, config, opts, save_images, env_id):
    global p_frame, pp_frame, im_num, state_to_save  
    global theimage, data, orig_ata, orig_heimage, canvas
    frame, orig_ata, old_frame, dilation = frame2attention(frame, conf, trainer, obj, config, opts, env_id)
    #im_path_C = os.path.join("im_log", str(obj.im_num) + '_C.png')
    blurred_frame = old_frame
    #orig_ata = frame[:,:,0]
    frame_to_render = frame
    frame_before_tranform = frame
    if trainer is not None:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = rgb2gray(frame)*255.

        #imgplot = plt.imshow(frame)
        #plt.show()
        if 1==opts.a2b:
            rotate_A = config.datasets['train_a']['rotation']
            rotate_B = config.datasets['train_b']['rotation']
        else:
            rotate_A = config.datasets['train_b']['rotation']
            rotate_B = config.datasets['train_a']['rotation']

        cols = frame.shape[1]
        rows = frame.shape[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_A,1)
        frame = cv2.warpAffine(frame, M, (cols,rows))


        frame = frame.transpose(2,0,1) 
        if not opts.blurr:
            frame = frame[-1:,:,:]
        final_data = torch.from_numpy((frame / 255.0 - 0.5) * 2).float().clone()
        #final_data = torch.FloatTensor((frame / 255.0 - 0.5) * 2)
        final_data = final_data.contiguous()
        final_data = final_data.resize_(1,final_data.size(0),final_data.size(1),final_data.size(2))


        use_xy = 1 == config.hyperparameters['gen']['use_xy']

        if use_xy:
            final_data = torch.cat([final_data, xy], 1)
        #final_data = final_data

        #xy = Variable(torch.from_numpy(new_map), requires_grad=False)
        #ForkedPdb().set_trace()
        if len(opts.gpu_ids) > 0:
            #final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2))).cuda(opts.gpu)
            with torch.cuda.device(trainer.gpu):
                final_data_in = Variable(final_data.cuda())
        else:
            final_data_in = Variable(final_data)

        orig_ata = frame[0]


        final_data_in = final_data_in.contiguous()
        #ForkedPdb().set_trace()
        if 1==opts.a2b:
            output_data = trainer.gen.forward_a2b(final_data_in)
            #output_data = (final_data,)
        else:
            output_data = trainer.gen.forward_b2a(final_data_in)
            #output_data = (final_data,)

        if opts.use_embeddings:
            embeddings = output_data[1].data.cpu().numpy()[0]
            #print(embeddings.shape)
            embeddings = np.stack((embeddings, embeddings))
            #print(embeddings.shape)
            return embeddings
        output_img = output_data[0].data.cpu().numpy()
        #print(output_img.size())
        new_output_img = np.transpose(output_img, [2, 3, 1, 0])
        new_output_img = new_output_img[:, :, :, 0]
        out_img = np.uint8(255 * (new_output_img / 2 + 0.5))
        #frame = 255-out_img
        frame = out_img


        cols = frame.shape[1]
        rows = frame.shape[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),360-rotate_B,1)
        #cv2.imwrite(im_path_E, frame)
        frame = cv2.warpAffine(frame, M, (cols,rows))
        frame_to_render = frame
        """
        if not args.gray:
            frame = frame[:,:,2]
            gray_im = cv2.resize(frame, (256,256))
            _, filtered_bin_image = cv2.threshold(gray_im,70,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            eroded = cv2.erode(filtered_bin_image ,kernel,iterations = 1)
            dilation = cv2.dilate(eroded ,kernel,iterations = 1)

            inv_im = 255-dilation
            inv_im_dist = cv2.distanceTransform(inv_im, cv2.DIST_L2, 3)
            exp_inv_im_dist = dist_exp(inv_im_dist)
            exp_im_dist = 255-exp_inv_im_dist
            frame = np.stack((exp_im_dist, exp_im_dist, exp_im_dist), axis=-1)
        """
    if opts.render:
        conv_frame = cv2.resize(frame, (256,256)) 
        orig_ata = cv2.resize(orig_ata, (256,256)) 
        if opts.blurr:
            conv_frame = rgb2gray(conv_frame)*255.

        data = np.concatenate((orig_ata, conv_frame), axis=1)
        data = cv2.resize(data, (256*2, 256)).astype(int)
        #cv2.imwrite('im_log/{}.jpg'.format(im_num), data)
        #st()
        im=Image.frombytes('L', (256*2,256), data.astype('b').tostring())
        photo = ImageTk.PhotoImage(image=im)
        canvas.create_image(0,0,image=photo,anchor=Tkinter.NW)
        master.update()
        data=np.roll(data,-1,1)

    if frame.shape[-1] != 3:
        dilation = np.stack((dilation, dilation,dilation), axis=-1)
        frame = np.stack((frame, frame, frame), axis=-1)
        
    if save_images and obj.im_num < 10:
        if not opts.per_process_convertor: 
            print('saving')
            data = cv2.resize(frame, (256, 256)).astype(int)
            cv2.imwrite('im_log/frame_{}_{}.jpg'.format(opts.experiment_name, obj.im_num), data)
            #torchvision.utils.save_image(torch.from_numpy(data), 'im_log/frame_{}.jpg')
        else:
            #print("printing image {}".format(opts.experiment_name, obj.im_num))
            data = cv2.resize(frame, (256, 256)).astype(int)
            #torchvision.utils.save_image(torch.from_numpy(data), 'im_log/new_frame_{}.jpg')
            cv2.imwrite('im_log/new_frame_{}_{}.jpg'.format(opts.experiment_name, obj.im_num), data)
            data = cv2.resize(dilation, (256, 256)).astype(int)
            #torchvision.utils.save_image(torch.from_numpy(data), 'im_log/new_dialation_{}.jpg')
            cv2.imwrite('im_log/new_dialation_{}_{}.jpg'.format(opts.experiment_name, obj.im_num), data)
            data = cv2.resize(frame_before_tranform, (256, 256)).astype(int)
            #torchvision.utils.save_image(torch.from_numpy(data), 'im_log/new_before_frame_{}.jpg')
            cv2.imwrite('im_log/new_before_frame_{}_{}.jpg'.format(opts.experiment_name, obj.im_num), data)

    dilation = rgb2gray(dilation)
    dilation = cv2.resize(dilation, (80, conf["dimension2"]))
    #dilation = cv2.resize(rgb2gray(dilation), (80, conf["dimension2"]))
    dilation = cv2.resize(dilation, (80, 80))
    dilation = np.reshape(dilation, [1, 80, 80])


    frame = rgb2gray(frame)
    frame = cv2.resize(frame, (80, conf["dimension2"]))
    #frame = cv2.resize(rgb2gray(frame), (80, conf["dimension2"]))
    frame = cv2.resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])

    old_frame = cv2.resize(old_frame, (80, 80))
    state_to_save = old_frame

    old_state = rgb2gray(blurred_frame)
    old_state = cv2.resize(old_state, (80, conf["dimension2"]))
    old_state = cv2.resize(old_state, (80, 80))
    old_state = np.reshape(old_state, [1, 80, 80])
    obj.im_num += 1

    #frame = np.stack((frame, dilation))
    if (opts.co_train_expantion): 
        frame = np.stack((frame, frame))
    else:
        frame = np.stack((frame, old_state))
    return frame

_process_frame = _atten_process_frame
class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf, convertor, transformation, flann , config, opts, save_images, env_id):
        super(AtariRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [2, 1, 80, 80])
        self.conf = env_conf
        self.convertor = convertor
        self.transformation = transformation
        self.im_num = 0
        self.flann = flann
        self.config = config
        self.opts = opts
        self.save_images = save_images
        self.env_id = env_id

    def _observation(self, observation):
        return _process_frame(observation, self.conf, self.convertor, self, self.config, self.opts, self.save_images, self.env_id)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = np.array([0, 0]).reshape((2,1,1,1))
        self.state_std = np.array([0, 0]).reshape((2,1,1,1))
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
                observation.mean(axis=(1,2,3), keepdims=True) * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
                observation.std(axis=(1,2,3), keepdims=True) * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
"""
class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])
        self.conf = env_conf

    def _observation(self, observation):
        return _process_frame(observation, self.conf)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

"""
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = random.randrange(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
        return obs


class BreakoutReward(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        self.side = 0
        self.threshold = 80


    def _reset(self):
        self.side = 0
        return self.env.reset()

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward != 0:
            reward = 0
            self.side = 0
            return obs, reward, done, info

        state = obs
        state = state[30:,:,:]
        state = rgb2gray(state)*255.
        state = state.astype(np.uint8)

        _, state = cv2.threshold(state,200,1,cv2.THRESH_BINARY)
        x_cord = state.sum(axis=0).argmax()
        if 0 == x_cord:
            return obs, reward, done, info
        new_side = 1 if x_cord < self.threshold else -1
        if 1 == new_side and -1 == self.side:
            reward = 1
        else:
            reward = 0

        self.side = new_side

        return obs, reward, done, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class FireEpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            obs, _, done, _ = self.env.step(1)
            if done:
                return obs, reward, done, info
            obs, _, done, _ = self.env.step(2)
            if done:
                return obs, reward, done, info
        self.lives = lives
        return obs, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            self.was_real_done = False
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.lives = 0
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info['ale.lives']
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

