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

import numpy as np
import pickle
from pynput.keyboard import Listener, Key



class Train:
    def __init__(self, num_levels):
        self.agent = Agent(num_levels)
        
    def reset_agent(self, num_levels):
        self.agent = Agent(num_levels)
    

"""
    Key handler to debug the agent.
"""
def on_press(key):
    global runner
    if key == Key.space:  
        if runner.show_window == False:
            runner.show_window = True
        else:
            runner.show_window = False
            
    if key == Key.tab:
        if runner.explore:
            runner.explore = False
        else:
            runner.explore = True
        
        print("EXPLORATION: ", runner.explore)

    
    if str(key) == "'k'":
        runner.stop_time = runner.stop_time + 1
        print("ST INCREASE: ", runner.stop_time)
        
    if str(key) == "'j'":
        runner.stop_time = runner.stop_time - 1
        print("ST DECREASE: ", runner.stop_time)
  
  
  
            
"""
    Main entry point.
"""
runner = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--file_name', type=str, default='cycle')
    
    args = parser.parse_args()
    
    file_name = args.file_name
    
    save = True
    success_traject = None
    best_action_dict = {}
    
    trainer = Train(1)
    runner = Runner(trainer.agent, args.n_episodes)
    utils = CausalUtils(runner)

    runnable = Runnables().RunnableAtari(trainer, runner, utils)
    with Listener(on_press=on_press) as listener:
        for i in range(args.n_train):
            if save:
                runnable.run(args, i, save)
            else:
                runner = runnable.load()
                best_action_dict = runnable.run(args, i, save)

            
            #init_nn("MiniGrid-DoorKeyY-8x8-v0", best_action_dict)
            
            
        listener.join()
