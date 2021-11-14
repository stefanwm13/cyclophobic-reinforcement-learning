
import time
import argparse
import numpy as np
import cv2
import gym
import gym_minigrid
from gym_minigrid import wrappers as mg_wrappers
from window import Window
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PreprocessFrameWrapper(gym.ObservationWrapper):
    """A wrapper that scales the observations from 210x160 to :py:attr:`width` x :py:attr:`height`.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
        width: int
            The target width of a given observation
        height: int
            The target height of a given observation
    """

    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
        self.dim = self.observation_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, self.dim), dtype=np.float32)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA )  # scale
        frame = frame / 255
        frame = np.asarray(frame, dtype=np.float32)
        print(frame)

        return frame


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)
        plt.imsave('Multi-Room.png', img)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info, pos = env.step(action)
    #print(env)
    print("POS : ", pos)
    #print("OBS: ", obs.shape)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    #print(obs['image'][:,:,0].shape)
    #pca = PCA(n_components=5)
    #pca_result = pca.fit_transform(obs['image'][:,:,0])
    #print(pca_result)
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

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

if args.agent_view:
    env = mg_wrappers.RGBImgObsWrapper(env)
    #env = FullyObsWrapper(env)
    env = mg_wrappers.ImgObsWrapper(env)
    #print(env)
    #env = FlatObsWrapper(env)
    env = mg_wrappers.PreprocessFrameWrapper(env, 120, 120)   # preprocess (scale down) the observations

    #env = mg_wrappers.RGBImgObsWrapper(env)     # extract the rgb image observation from the default state dict
    #env = mg_wrappers.FullyObsWrapper(env)      # use fully observable states
    #env = mg_wrappers.ImgObsWrapper(env)    

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
