#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import xxhash
from atari_wrappers import make_atari, wrap_deepmind
import nle



def redraw(img):
    if not args.agent_view:
        img = env.render()#('rgb_array', tile_size=args.tile_size)

    #window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    #redraw(obs)

def step(action):
    global last_obs
    obs, reward, done, info = env.step(action)
    #print(xxhash.xxh64(obs).hexdigest())
    #print(obs.shape)
    print(obs)
    #if len(last_obs) > 0:
    #    window.show_img(last_obs - obs)
    #print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        #if len(last_obs) > 0:
        #    redraw(last_obs - obs)
        #else:
        redraw(obs)
    last_obs = obs

    #ram = env.unwrapped._get_ram()
    #print(ram[42])
    #print(ram[43])
    
    #print(env.unwrapped.get_action_meanings())
    #help(env.unwrapped)


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        #step(env.actions.left)
        step(0)
        return
    if event.key == 'right':
        #step(env.actions.right)
        step(1)
        return
    if event.key == 'up':
        #step(env.actions.forward)
        step(2)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        #step(env.actions.pickup)
        step(3)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return
    


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

last_obs = []

args = parser.parse_args()

#env = gym.make(args.env)
#env = make_atari(args.env)
#env = wrap_deepmind(env)
env = gym.make("NetHackScore-v0")

if args.agent_view:
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
