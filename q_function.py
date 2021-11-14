from function import TabularFunction

from collections import deque
import random

'''
    This class describes an agent that is updated with tabular q-learning
'''
class QFunction(TabularFunction):
    def __init__(self):
        super().__init__()
        self.success_buffer = deque(maxlen=5)
        self.success_traject = []
        
    def backup(self, batch, i, alpha, gamma):
        state = batch[i][0]
        action = batch[i][1]

        if i < len(batch) - 1:
            next_action = batch[i+1][1]

        reward = 1 * batch[i][2]
        next_state = batch[i][3]


        # Calculate new q-value and update q_table
        old_q_value = self.qtable.get_value(state, action)
        if i < len(batch) - 1:
            # Do SARSA or Q-Learning update
            #print("SARSA")
            next_q_value = self.qtable.get_value(next_state, next_action)
            new_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_value)
        else:
            #print("LAST")
            # For last state in batch new value is the reward
            new_value = (1 - alpha) * old_q_value + alpha * reward

        #Update Q-Table
        self.qtable.set_value(state, action, new_value)
    
    
    
    def update_best(self, alpha, gamma):
        if len(self.success_buffer) > 0:
            #print("UPDATING WITH GOAL")
            s_batch = random.sample(self.success_buffer, 1)[0]
            
            reward_state = [i for i in range(len(s_batch)) if s_batch[i][2] > 0]

            if len(reward_state) > 0:
                for i in range(reward_state[0], -1, -1):
                    self.backup(s_batch, i, alpha=0.9, gamma=gamma)
        
    
    def update(self, alpha, gamma):
        # Propagate complete successful goal trajectory
        reward_state = [i for i in range(len(self.batch)) if self.batch[i][2] > 0]
        
        if len(reward_state) > 0:
            for i in range(reward_state[0], -1, -1):
                self.backup(self.batch, i, alpha=0.9, gamma=gamma)
                
       
            
            
            
            
            
            
            
