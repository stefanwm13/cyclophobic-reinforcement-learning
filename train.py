"""
    train.py
        This class specifies the training flow of the agent. From here the alternating exploration and determination of causal objects is performed.
"""

import argparse
import time
import random

from vec_env import HashVectorizedEnvironment
from agent import Agent
from runner import Runner
from causal_helpers import CausalUtils

from runnables import Runnables

#from A2C.main import init_nn
from DQN.main import init_nn
#from non-linear.torch-ac

import numpy as np
import pickle
from pynput.keyboard import Listener, Key



class Train:
    def __init__(self, num_levels):
        self.agent = Agent(num_levels)
        self.show_window = False
        self.stop_time = 0

    def reset_agent(self, num_levels):
        self.agent = Agent(num_levels)


"""
    Key handler to debug the agent.
"""
def on_press(key):
    global runner
    global trainer
    if key == Key.space:
        if trainer.show_window == False:
            trainer.show_window = True
            print(trainer.show_window)
        else:
            trainer.show_window = False

    if key == Key.tab:
        if runner.explore:
            runner.explore = False
        else:
            runner.explore = True

        print("EXPLORATION: ", runner.explore)


    if str(key) == "'k'":
        trainer.stop_time = trainer.stop_time + 1
        print("ST INCREASE: ", trainer.stop_time)

    if str(key) == "'j'":
        if trainer.stop_time > 0:
            trainer.stop_time = trainer.stop_time - 1
            print("ST DECREASE: ", trainer.stop_time)
        else:
            print("NORMAL SPEED")


"""
    Main entry point.
"""
runner = None
trainer = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--file_name', type=str, default='cycle')

    args = parser.parse_args()

    file_name = args.file_name

    save = True

    trainer = Train(3)
    runner = Runner(trainer.agent, args.n_episodes)
    utils = CausalUtils(runner)

    runnable = Runnables().RunnableUnlockPickup()

    with Listener(on_press=on_press) as listener:
        for i in range(args.n_train):
            if save:
                runnable.run(args, i, save, trainer)
            else:
                runnable.load()
                runnable.run(args, i, save, trainer)

            runnable.test(args, i, trainer, runner)

        #init_nn("MiniGrid-DoorKeyY-8x8-v0", None)

        listener.join()
