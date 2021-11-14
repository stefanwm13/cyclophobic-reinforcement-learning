 
import sys
sys.path.append('../')
import argparse
import time
import pickle
import random
from collections import Counter
import csv
from gym_minigrid.window import Window
from gym_minigrid import wrappers as mg_wrappers

from qtable import QTableFactory

'''
    This is the parent class of the agent. From here we specify the cycle and q agent.

'''

class TabularFunction:
    def __init__(self):
        self.qtable = QTableFactory.create_obs_qtable()
        self.batch = []


    def backup(self, i, gamma, alpha):
        print("Not implemented")
        
        
        
    def update(self, alpha, gamma):
        print("Not implemented")



    def store_transition(transition):
        self.batch.append(transition)
