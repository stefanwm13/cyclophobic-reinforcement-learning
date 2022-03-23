import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import xxhash

import pickle

def redraw(img, img2):
    if not args.agent_view:
        img = env1.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)
    window2.show_img(img2)

def reset():
    global last_obs
    global state_map

    if args.seed != -1:
        env.seed(args.seed)

    obs = env1.reset()
    obs2 = env2.observation(env1.gen_obs())['image']

    obs_key = xxhash.xxh64(obs).hexdigest()

    if obs_key not in state_map:
        state_map[obs_key] = 1
    else:
        state_map[obs_key] += 1

    last_obs = obs

    if hasattr(env, 'mission'):
        print('Mission: %s' % env1.mission)
        window.set_caption(env1.mission)

    redraw(obs, obs2)


def step(action):
    global last_obs
    global state_map
    global state_table

    obs, reward, done, info = env1.step(action)
    print(info)
    obs2 = env2.observation(env1.gen_obs())['image']

    obs_key = xxhash.xxh64(obs).hexdigest()
    print(obs_key)

    batch.append([obs, obs_key, obs2])

    last_obs_key = xxhash.xxh64(last_obs).hexdigest()
    if obs_key not in state_map:
        state_map[obs_key] = 1
    else:
        state_map[obs_key] += 1

    if last_obs_key not in state_table:
        state_table[last_obs_key] = [0] * 7
        state_table[last_obs_key][action] = np.linalg.norm(last_obs - obs)
    else:
        state_table[last_obs_key][action] = np.linalg.norm(last_obs - obs)

    print('step=%s, reward=%.2f' % (env1.step_count, reward))
    print("DIST: ", np.linalg.norm(last_obs - obs))
    print("S_COUNT: ", state_map[obs_key])
    print("W_DIST: ", np.linalg.norm(last_obs - obs) * (1 / state_map[obs_key]))
    print(state_table[last_obs_key])

    if done:
        print('done!')
        reset()
    else:
        redraw(obs, obs2)

    last_obs = obs


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        file = open('states.p', 'wb')
        pickle.dump(batch, file)
        file.close()
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

args = parser.parse_args()

env = gym.make(args.env)
env1 = None
env2 = None

batch = []
state_map = {}
state_table = {}

last_obs = None

if args.agent_view:
    #env1 = RGBImgNxNObsWrapper(env)
    env1 = RGBImgPartialObsWrapper(env)
    env1 = ImgObsWrapper(env1)
    env2 = RGBImgNxNObsWrapper(env)

    #env2 = RGBImgObsWrapper(env)
    #env2 = ImgObsWrapper(env2)

window = Window('gym_minigrid - ' + args.env)
window2 = Window('gym_minigrid2 -' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
window2.show(block=True)
