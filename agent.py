"""
    This class represents the agent. It consists of n cycle-functions and n q-functions represented in q_function.py and cycle_function.py. 
"""

from q_function import QFunction
from cycle_function import CycleFunction

import numpy as np

class Agent:
    def __init__(self, num_levels):
        self.num_levels = num_levels
        
        self.qs = []
        self.cs = []
        self.cs_success = []
        self.qs_u = []
        self.cs_u = []
        self.cs_u_success = []
        
        for i in range(num_levels):
            self.qs.append(QFunction())
            self.cs.append(CycleFunction())
            self.cs_success.append(CycleFunction())
            self.qs_u.append(QFunction())
            self.cs_u.append(CycleFunction())
            self.cs_u_success.append(CycleFunction())
   
   
    """
        Evaluate the agent at given state and return action.
    """
    def evaluate(self, state, level):
        cycle_q_vec = self.cs[level].qtable.get_value(state[level], None)
        q_vec_0 = self.qs[level].qtable.get_value(state[level], None)
        
        
        return np.array(q_vec_0) + np.array(cycle_q_vec)
    
    """
        Store transition to current episodes batch which agent uses to determine cycles.
    """
    def store_transition(self, transition):
        for i in range(self.num_levels):
            state = transition[0][i]
            action = transition[1]
            reward = transition[2][i]
            next_state = transition[3][i]
            obs = transition[4][i]
            
            self.qs[i].batch.append((state, action, reward, next_state, obs))
            self.cs[i].batch.append((state, action, reward, next_state, obs))
            self.cs_success[i].batch.append((state, action, reward, next_state, obs))

            self.qs_u[i].batch.append((state, action, reward, next_state, obs))
            self.cs_u[i].batch.append((state, action, reward, next_state, obs))
            self.cs_u_success[i].batch.append((state, action, reward, next_state, obs))
    """
        Store successfull transitions in special buffers for experience replay.
    """
    def store_best_batch(self):
        for i in range(self.num_levels):
            self.qs[i].success_buffer.append(self.qs[i].batch)
            self.qs[i].success_traject.append(self.qs[i].batch)
            
    
    """
        Empty batch after every episode.
    """
    def clear_batch(self):
        for i in range(self.num_levels):
            self.qs[i].batch = []
            self.cs[i].batch = []    
            self.cs_success[i].batch =[]    
            self.qs_u[i].batch = []
            self.cs_u[i].batch = []    
            self.cs_u_success[i].batch = []   


            self.cs[i].cycle_hash_function = {}
            self.cs_success[i].cycle_hash_function = {}
            self.cs_u[i].cycle_hash_function = {}
            self.cs_u_success[i].cycle_hash_function = {}   
    
    """
        Update q and cycle functions of agent for every view.
    """
    def update(self, global_steps, successfull_episodes, alpha, gamma):
        #for i in range(self.num_levels):
        
        for i in range(self.num_levels):
            self.qs[i].update(alpha, gamma)
            self.cs[i].update_hash(global_steps, alpha, gamma, False)
            self.qs_u[i].update(1.0, gamma)
            self.cs_u[i].update_hash(global_steps, 1.0, gamma, True)
            
            if successfull_episodes > 650:    
                self.cs_success[i].update_hash(global_steps, alpha, gamma, False)
                self.cs_u_success[i].update_hash(global_steps, 1.0, gamma, True)
                #self.cs_success[1].update(alpha, gamma)
            
            #self.cs[1].update(alpha, gamma)
            #self.qs[1].update(alpha, gamma)

    
    """
        Replay successful trajectories.
    """
    def update_best(self, alpha, gamma):
        for i in range(self.num_levels):
            self.qs[i].update_best(alpha, gamma)

        
        
