import random
import numpy as np
import copy
import time

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from sklearn.cluster import KMeans


from causal_helpers import CausalUtils, Trajectory
from causal_graph import Node, CausalGraph

from logger import Logger


"""
    This class wraps the main loop and interventional loop of the agent. 
"""

class Runner:
    def __init__(self, agent, max_episodes):        
        self.show_window = False
        self.explore = True
        self.burnin = True
        self.stop_time = 0
        
        self.iterations = 0
        self.max_episodes = max_episodes

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.2
        
        self.agent = agent
        self.env = None
        
        self.global_obs_arr = []
        self.duplicate_dict = {}
        self.logger = Logger()
    

    def change_agent(self, agent):
        self.agent = agent
    

    """
        Main loop of agent
    """
    def run(self, iterations, alpha, gamma, epsilon, cycle, causal_tools, max_episodes, exploration_threshold, level):
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cycle = cycle

        self.utils = CausalUtils(self)
        
        first_state_obs = None
        
        if causal_tools != None:
            best_action_dict = causal_tools[0]
            causal_states = causal_tools[1]
            non_causal_states = causal_tools[2]
            
            auxiliary_action_dict = {}
            for key in best_action_dict:
                auxiliary_action_dict[key] = []
            
            print(causal_states)
                
        if max_episodes != None:
            self.max_episodes = max_episodes
        
        episodes_successful = 0
        global_steps = 0
        
        
        #Main loop
        for i in range(self.max_episodes):
            episode_steps = 0
            done = 0
            causal_index_begin = 0
            causal_index_end = 2
            
            if causal_tools != None:
                causal_dict = {}
                for j in range(len(causal_states)):
                    causal_dict[causal_states[j]] = 0
                    
                    
            self.trajectory_nxn = Trajectory()
            self.trajectory = Trajectory()
                
            self.agent.clear_batch()
                        
            print("New episode: ", str(self.iterations) + "  " + str(i+1))

            state, obs = self.env.reset()  
                        
                        
            #Episode loop            
            while np.sum(done) == 0:                
                #Sample action
                if np.random.uniform(0,1) < self.epsilon and self.explore:
                    action = random.randint(0, self.env.get_action_space() - 1)
                else:
                    b = np.array(self.agent.evaluate(state, level))
                    action = np.argmax(b)
                
                #Execute actions that are relevant for current state of agent
                if causal_tools != None:
                    current_causal_states = causal_states[causal_index_begin:causal_index_end] 
                
                    #Check if state is a causal state and execute its action
                    if best_action_dict != None and state[1] in current_causal_states:
                        
                        if state[1] in non_causal_states and not causal_dict[state[1]]:
                            action = best_action_dict[non_causal_states[non_causal_states.index(state[1])]]
                        elif state[1] in non_causal_states and causal_dict[state[1]]:
                            pass
                        else:
                            action = best_action_dict[state[1]]
                        
                        causal_dict[state[1]] = 1
                     
                    #Increase rolling causal index which indicates which two objects are selected
                    sum = 0
                    for i in range(causal_index_end):
                        sum += causal_dict[causal_states[i]]
                        
                    if sum == causal_index_end and sum < len(causal_states):
                        causal_index_begin += 1
                        causal_index_end += 1
                    
            
                for i in range(level):
                    self.utils.add_trajectory_transition(level, self.trajectory, ((state[level], obs[level]), action))
                    
                time.sleep(self.stop_time)
                next_state, reward, done, info, next_obs = self.env.step(action)
               
                
                if self.show_window:
                    if obs != None:
                        self.env.show_windows(obs)
      
                    
                self.agent.store_transition((state, action, reward, next_state, obs))
                self.agent.update(episode_steps, episodes_successful, self.alpha, self.gamma)
        
                if np.sum(reward) > 0:
                    self.agent.store_best_batch()
                    episodes_successful += 1
                    
                state = next_state
                obs = next_obs
                episode_steps = episode_steps + 1
                global_steps = global_steps + 1
                
            #Log reward
            #self.logger.log("results/DoorKey-11x11_new.csv", global_steps, reward[0])
            
            if episodes_successful > exploration_threshold:
                self.explore = False
                
            self.agent.update_best(self.alpha, self.gamma)        
            
            print("EPISODES SUCCESSFUL: ", str(episodes_successful) + "  " + str(reward[0]))
      
            if causal_tools != None:
                if episodes_successful > 0:
                    self.add_auxiliary_states((self.trajectory, self.trajectory_nxn), auxiliary_action_dict, best_action_dict, 2)
            
     
    """
        - perform_interventions()
            This method performs systematic interventions on observations of the local view. For this a local causal graph is built for each observation and compared to the cycle q-vector. In general, the causal relevant action will not cause a cycle while being crucial for the agent success. We intervene on the local causal graph and for the action which yields the longest trajectory the q-vector value should be 0, thus representing that there is no cycle. 
    """
    def perform_interventions(self, level):
        obs_arr = []
        obs_arr_nxn = []
        self.state_obs_dict = {}
        self.state_obs_dict_nxn = {}
        C0_to_C1_dict = {}
        C0_to_C1_dict_nxn = {}
        counter_dict = {}
        
        causal_graph = CausalGraph()
        causal_graph_nxn = CausalGraph()
        self.trajectory = Trajectory()
        self.trajectory_nxn = Trajectory()
        self.utils = CausalUtils(self)
        
        obs_arr, C0_to_C1_dict = self.utils.extract_obs(level, obs_arr, C0_to_C1_dict)
        self.global_obs_arr = self.global_obs_arr + obs_arr

        obs_arr_nxn, C0_to_C1_dict_nxn = self.utils.extract_obs(2, obs_arr_nxn, C0_to_C1_dict_nxn)
        
        print(C0_to_C1_dict.keys())

        self.state_obs_dict = self.utils.fill_state_obs_dict(self.state_obs_dict, obs_arr)
        self.state_obs_dict_nxn = self.utils.fill_state_obs_dict(self.state_obs_dict_nxn, obs_arr_nxn)#

        action_state_dict = {}

        self.utils.fill_causal_state_dict(C0_to_C1_dict, self.state_obs_dict, causal_graph, action_state_dict)
        
        print(self.state_obs_dict_nxn.keys())
        #self.utils.fill_causal_state_dict(C0_to_C1_dict_nxn, state_obs_dict_nxn, causal_graph_nxn, action_state_dict)
        
        self.utils.build_global_causal_graph(level, self.trajectory, self.env, causal_graph)
        self.utils.build_global_causal_graph(2, self.trajectory_nxn, self.env, None)
                
        #causal_graph.draw()
        #plt.show()


        ##### Perform interventions
        state_action_vec = None
        best_action_dict = {}
        cycle_value_dict = {}
        
        actions_dict = {}
        causal_actions_dict = {}
        secondary_causal_states = []

        
        success_states = []
        causal_state_action_dict = {}
        for i in range(len(self.trajectory.state_action_vec)):
            success_states.append(self.trajectory.state_action_vec[i][0])
        
        for i in range(7):
            cycle_value_dict[str(i)] = 0
            causal_actions_dict[i] = []
            
           
        for key in C0_to_C1_dict.keys():
            c1_q_s_value = self.agent.cs_u_success[level].qtable.get_value(key, None)
            c1_q_value = self.agent.cs_u[level].qtable.get_value(key, None)
            
            if key not in success_states:
                continue
            
            # Get actions that are part of the successful trajectory for every key and do not cause cycle
            actions = []
            for i in range(len(self.trajectory.state_action_vec)):
                if key == self.trajectory.state_action_vec[i][0]:
                    actions.append(self.trajectory.state_action_vec[i][1])
        
            actions = list(set(actions))
            
            print("A: ", actions)
            
            # Get causally relevant actions
            causal_actions = []
            causal_action_value = {}
            non_causal_action_value = {}
            for action in actions:
                if c1_q_s_value[action] == 0:
                    causal_actions.append(action)
                    causal_action_value[action] = c1_q_value[action]

                else:
                    non_causal_action_value[action] = c1_q_s_value[action]
            
            # If all potential causal actions cause cycles take the one with least cycle probability
            if len(causal_actions) == len(actions):
                causal_actions = []
                causal_actions.append(max(causal_action_value, key=causal_action_value.get))
            
            if len(causal_actions) == 0:
                causal_actions = []
                causal_actions.append(max(non_causal_action_value, key=non_causal_action_value.get))
                
            # Get values and corresponding states for each action
            causal_actions_dict[causal_actions[0]].append((key, c1_q_value[causal_actions[0]]))
            
            # Summed up cycle values for each action
            for i in causal_actions:
                cycle_value_dict[str(i)] =  cycle_value_dict[str(i)] + c1_q_s_value[i]
            
            
            causal_state_action_dict[key] = causal_actions[0]
            
            parent_count = 0
            for i in range(len(self.trajectory.state_action_vec)):
                if key == self.trajectory.state_action_vec[i][0]:
                    nxn_key = self.trajectory_nxn.state_action_vec[i][0]
                    parent_count += 1

            
            if parent_count == 1:
                secondary_causal_states.append(key)
                
        
        #Save causal actions together with causal states
        for key in causal_state_action_dict:
            if cycle_value_dict[str(causal_state_action_dict[key])] > -0.5:
                best_action_dict[key] = causal_state_action_dict[key]
                
        non_causal_dict = {}
        for key in cycle_value_dict:
            if cycle_value_dict[key] < -0.1:
                non_causal_dict[key] = cycle_value_dict[key]
                
        max_non_causal_action = max(non_causal_dict, key=non_causal_dict.get)
        
        print("MNCA: ", max_non_causal_action)
        
        #Save state-action pair that is unique through all partial views 
        non_causal_states = []
        for state in secondary_causal_states:
            if causal_state_action_dict[state] == int(max_non_causal_action):
                print(causal_state_action_dict[state])

                non_causal_states.append(state)
                best_action_dict[state] = causal_state_action_dict[state]
                
    
        return best_action_dict, self.trajectory, non_causal_states


    '''
        Use states from other partial views to aid the agent. (WIP)
    '''
    def add_auxiliary_states(self, trajectories, auxiliary_action_dict, best_action_dict, level):
        obs_arr = []
        state_obs_dict = {}
        C0_to_C1_dict = {}
        counter_dict = {}
        
        trajectory = trajectories[0]
        trajectory_nxn = trajectories[1]
        
        causal_graph = CausalGraph()
        
        self.utils = CausalUtils(self)
        
        #obs_arr, C0_to_C1_dict = self.utils.extract_obs(level, obs_arr, C0_to_C1_dict)
        
        #self.global_obs_arr = self.global_obs_arr + obs_arr
        
        #print(C0_to_C1_dict.keys())

        for i in range(len(trajectory_nxn.state_action_vec)):
            self.global_obs_arr.append(trajectory_nxn.state_action_vec[i][0][1])

        state_obs_dict = self.utils.fill_state_obs_dict(state_obs_dict, self.global_obs_arr)
                            
        
        #print("AXD: ", auxiliary_action_dict)
        
        #for i in range(len(trajectory.state_action_vec)):
            #print((trajectory.state_action_vec[i][0][0], trajectory.state_action_vec[i][1]))
         
        #print("NXN")
        #for i in range(len(trajectory_nxn.state_action_vec)):
            #print((trajectory_nxn.state_action_vec[i][0][0], trajectory_nxn.state_action_vec[i][1]))
        
        #print(state_obs_dict.keys())
        
        previous_states = None
        next_states = None
        for key in best_action_dict:
            print(key)
            print(best_action_dict)
            
            for i in range(len(trajectory.state_action_vec)):
                if trajectory.state_action_vec[i][0][0] == key and trajectory.state_action_vec[i][1] == best_action_dict[key]:
                    
                    previous_states = [(self.trajectory_nxn.state_action_vec[i-2][0][0], self.trajectory_nxn.state_action_vec[i-2][1]) , (self.trajectory_nxn.state_action_vec[i-1][0][0], self.trajectory_nxn.state_action_vec[i-1][1])]
                    
                    if (previous_states[0][0], previous_states[0][1]) in auxiliary_action_dict[key]:
                        #print(previous_states[0][0] + "  " + "already present")
                        
                        if not np.sum(self.agent.cs_u[level].qtable.get_value(previous_states[0][0], None)) == 0 and np.argmax(self.agent.cs_u[level].qtable.get_value(previous_states[0][0], None)) != previous_states[0][1]:
                            #print("Remove: ", previous_states[0][0])
                            auxiliary_action_dict[key].remove((previous_states[0][0], previous_states[0][1]))
                        
                    else:
                        auxiliary_action_dict[key].append((previous_states[0][0], previous_states[0][1]))
                      
                    
                    if (previous_states[1][0], previous_states[1][1])  in auxiliary_action_dict[key]:
                        #print(previous_states[1][0] + "  " + "already present")
                        
                        if not np.sum(self.agent.cs_u[level].qtable.get_value(previous_states[1][0], None)) == 0 and np.argmax(self.agent.cs_u[level].qtable.get_value(previous_states[1][0], None)) != previous_states[0][1]:
                            #print("Remove: ", previous_states[1][0])
                            auxiliary_action_dict[key].remove((previous_states[1][0], previous_states[1][1]))
                            
                    else:
                        auxiliary_action_dict[key].append((previous_states[1][0], previous_states[1][1]))
                    
                    


    '''
        This function orders the causal objects according to the order they appeared in when solving the environment
    '''
    def build_causal_timeline(self, best_action_dict, trajectory):
        causal_dict = {}
        causal_states = []
        causal_state_indicator = []
        
        state_action_vec = trajectory.state_action_vec
        
        print(state_action_vec)
        
        for i in range(len(state_action_vec)):
            for key in best_action_dict:
                if state_action_vec[i][0] == key and state_action_vec[i][1] == best_action_dict[key]:
                    causal_states.append(key)
       
        print(causal_states)
        
        return causal_states
