from gym_minigrid.window import Window
from gym_minigrid import wrappers as mg_wrappers
from minigrid_env import *
from wrappers.atari_wrappers import make_atari, wrap_deepmind
from gym import wrappers

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

import xxhash

'''
    This class is the parent class that wraps the environment. 
'''

class VectorizedEnvironment:
    def __init__(self, num_levels, env_name, atari):
        self.num_levels = num_levels
        
        # Create environments
        self.envs = []
        if atari:
             self.envs.append(make_atari(env_name))
        else:
            if num_levels == 1:
                self.envs.append(make_minigrid_env(env_name))
            elif num_levels == 2:
                self.envs.append(make_minigrid_env(env_name))
                self.envs.append(make_minigrid_env_partial(env_name))
            else:
                self.envs.append(make_minigrid_env(env_name))
                self.envs.append(make_minigrid_env_partial(env_name))
                self.envs.append(make_minigrid_env_nxn(env_name))
        # Create windows for environments
        self.windows = []
        
        for i in range(num_levels):
            self.windows.append(Window("gym_minigrid_level_" + str(i)))
    
        #self.envs[0] = wrappers.Monitor(self.envs[0], 'videos/', force=True, video_callable=lambda episode_id: True)
    
    def show_windows(self, state):
        for i in range(self.num_levels):
            self.windows[i].show_img(state[i])
    
    def get_action_space(self):
        return self.envs[0].action_space.n
    
    def reset(self):
        state_array = []
        for env in self.envs:
            state_array.append(env.reset())
            
        return state_array
    
    def step(self, action):
        state_array = []
        reward_array = []
        done_array = []
        info_array = []
        
        for env in envs:
            next_state, reward, done, infos = env.step(action)
            state_array.append(next_state)
            reward_array.append(reward)
            done_array.append(done)
            info_array.append(infos)
            
        
        return state_array, reward_array, done_array, info_array

'''
    In this environment the observations are mapped to static hashes
'''
 
class HashVectorizedEnvironment(VectorizedEnvironment):
    def __init__(self, num_levels, env_name, atari):
        super().__init__(num_levels, env_name, atari)
    
    
    def reset(self):
        state_array = []
        obs_array = []
        
        for env in self.envs:
            obs = env.reset()
            obs_array.append(obs)
            state_array.append(xxhash.xxh64(obs).hexdigest())
            
        return state_array, obs_array
    
    
    def step(self, action):
        state_array = []
        reward_array = []
        done_array = []
        info_array = []
        obs_array = []
        
        for env in self.envs:
            next_state, reward, done, infos = env.step(action)
            obs_array.append(next_state)
            state_array.append(xxhash.xxh64(next_state).hexdigest())
            reward_array.append(reward)
            done_array.append(done)
            info_array.append(infos)
            
        
        return state_array, reward_array, done_array, info_array, obs_array


'''
    In this environment we use raw observations for the CNN (WIP)
'''
class CNNVectorizedEnvironment(VectorizedEnvironment):
    def __init__(self, num_levels, env_name, atari, net):
        super().__init__(num_levels, env_name, atari)
        self.net = net
    
    
    def reset(self):
        state_array = []
        obs_array = []
        
        for env in self.envs:
            obs = env.reset()
            obs_array.append(obs)
            state_array.append(self.net(Variable(torch.from_numpy(np.expand_dims(obs, axis=0).transpose((0, 3, 1, 2))).float() / 255.)).detach().numpy())

        return state_array, obs_array
    
    
    def step(self, action):
        state_array = []
        reward_array = []
        done_array = []
        info_array = []
        obs_array = []
        
        for env in self.envs:
            next_state, reward, done, infos = env.step(action)
            obs_array.append(next_state)
            state_array.append(self.net(Variable(torch.from_numpy(np.expand_dims(next_state, axis=0).transpose((0, 3, 1, 2))).float() / 255.)).detach().numpy())
            reward_array.append(reward)
            done_array.append(done)
            info_array.append(infos)
            
        
        return state_array, reward_array, done_array, info_array, obs_array
