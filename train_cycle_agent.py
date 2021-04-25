"""
    Create tabular q-learning agent that determines cycle function
    Author: Stefan Wagner
    23-04-2021
    
"""

import sys
sys.path.append('../')
import time
import pickle
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from goal_qtable import GoalQTable
from minigrid_env import *


env = make_minigrid_env("MiniGrid-MultiRoomS13N4-v0")
#env = make_minigrid_env("MiniGrid-Empty-16x16-v0")

# Create agents q-table
rows = 17
cols = 17
direc = 4
print(env.action_space.n)

log_list = [10, 50, 100, 200, 400, 800, 1000, 2000, 3000]

goal_qtable = GoalQTable([rows, cols, direc])


def create_logging_table(rows, cols):
    count_table = np.zeros((rows-1, cols-1))

    return count_table


def log_count(table, row, col):
    table[int(row)][int(col)] = table[int(row)][int(col)] + 1
    
    return table


def save_log_data(i, goal_qtable, logging_table, ep_success):
    afile = open(r'results/successful_cycle_3+'+str(i)+'_.dat', 'wb')
    pickle.dump(ep_success, afile)
    afile.close()
    
    afile = open(r'results/count_cycle_3+'+str(i)+'_.dat', 'wb')
    pickle.dump(logging_table, afile)
    afile.close()

    # Save Q-Table
    afile = open(r'results/q_table_cycle_3.dat', 'wb')
    pickle.dump(goal_qtable.get_qtable(), afile)
    afile.close() 
 
def get_goal_states(batch):
    states = []
    
    # Get list of all states in trajectory
    for transition in batch:
        states.append(transition[0])
      
    # Get occurences of states
    occurences = Counter(states)

    # Save states as goal states if they appear more than once
    goal_states = []
    for key in occurences:
        if occurences[key] > 1:
            goal_states.append(key)
            
    return goal_states
            
            
def update_Q(batch, alpha, gamma):
    goal_states = get_goal_states(batch)
    #print(goal_states)
    
    for goal_state in goal_states:

        for i in range(len(batch)):
            state = batch[i][0]
            action = batch[i][1]
            
            if i < len(batch) - 1:
                next_action = batch[i+1][1]
            
            reward = -10 * batch[i][2]
            next_state = batch[i][3]
            
            # Add reward if cycle is detected
            if next_state == goal_state:
                reward += 0.5
            
            # Calculate new q-value and update q_table
            old_q_value = goal_qtable.get_qtable()[goal_state][state][action]
            
            if i < len(batch) - 1:
                # Do SARSA update
                next_q_value = goal_qtable.get_qtable()[goal_state][next_state][next_action]
                new_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_value)
            else:
                # For last state in batch value is the reward
                new_value = (1 - alpha) * old_q_value + alpha * reward
            
            #Update Q-Table
            goal_qtable.get_qtable()[goal_state][state][action] = new_value
    
    
def run(max_episodes, alpha, gamma, epsilon):
    max_episodes = max_episodes
    episodes = 0
    episodes_successful = 0
    
    replay_buffer = deque(maxlen=5)

    logging_table = create_logging_table(rows, cols)
    
    # Episode loop
    for i in range(max_episodes):
        steps_done = 0
        batch = []
        
        if i in log_list:
            save_log_data(i, goal_qtable, logging_table, episodes_successful)
           
        print("New episode: ", i+1)
        
        env.reset()
        state = "110"
        
        log_count(logging_table, "1", "1")
        
        while steps_done < 300:
            # Sample action
            if np.random.uniform(0,1) < epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                b = np.array(goal_qtable.get_qtable()[state][state])
                action = np.random.choice(np.flatnonzero(b == b.min())) #If multiple zeros choose one at radom
                        
            # Execute action in environment
            #env.render()
            _, reward, done, info, pos_dir = env.step(action)
            
            next_state = str(pos_dir[0][0])+str(pos_dir[0][1])+str(pos_dir[1]) #(x, y, direction)
            log_count(logging_table, str(pos_dir[0][0]), str(pos_dir[0][1]))
           
            # Save transition to trajectory
            batch.append((state, action, reward, next_state))
            
            if reward > 0:                
                #Save successful trajectory
                replay_buffer.append(batch)
                episodes_successful += 1
                break
            
            state = next_state
            steps_done = steps_done + 1
          
        # Update Q-Table
        update_Q(batch, alpha, gamma)
        
        # Update with successful trajectory - Not used right now
        #if len(replay_buffer) > 0:
        #    print("UPDATING WITH GOAL")
        #    update_Q(random.sample(replay_buffer,1)[0], alpha, gamma)
          
        episodes = episodes + 1

        print("EPISODES SUCCESFUL: ", episodes_successful)
        



if __name__ == "__main__":
    
   run(max_episodes=3000, alpha=0.9, gamma=0.95, epsilon=0.1)
    
  
