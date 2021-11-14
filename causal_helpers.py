import networkx as nx
import matplotlib.pyplot as plt
import copy
import xxhash

from causal_graph import Node

from collections import Counter

"""
    This class allows for representing a trajectory, 
    and exchanging the action for the corresponding state that is intervened on.
"""
class Trajectory:
    def __init__(self):
        self.state_action_vec = []
        
        
    def add_state_action(self, state, action):
        self.state_action_vec.append((state, action))
    
    
    def replace_state_action(self, state, action):
        state_action_vec_copy = self.state_action_vec.copy()
        print("SAC_B: ", state_action_vec_copy)

        for i in range(len(state_action_vec_copy)):
            if state_action_vec_copy[i][0] == state:
                state_action_vec_copy[i] = (state_action_vec_copy[i][0], action)

        #print("SAC: ", state_action_vec_copy)
        return state_action_vec_copy

"""
    This class contains help methods needed to build the corresponding causal graphs.
"""
class CausalUtils:
    def __init__(self, runner):
        self.runner = runner
    
    
    def get_causal_order(self, agent, best_states):
        batch = agent.qs[1].success_traject[-1]
        
        best_states_list = []
        for i in range(len(best_states)):
            for j in range(len(best_states[i])):
                best_states_list.append(best_states[i][j])
                
                
        causal_trajectory = []
        for i in range(len(batch)):
            if batch[i][0] in best_states_list:
                causal_trajectory.append(batch[i][0])
        
     
        causal_states = []
        for i in range(len(causal_trajectory)-1):
            if [causal_trajectory[i], causal_trajectory[i+1]] in best_states:
                causal_states.append([causal_trajectory[i], causal_trajectory[i+1]])
            
        
        print(causal_states)
        return causal_states
    
     
    
    def get_causal_object_sequence(self,  best_action_dict):
        best_states = []
        best_state_pair = []
        
        for key in best_action_dict:
            print(key)
            if key == "PH":
                best_states.append(best_state_pair)
                best_state_pair = []
                
            elif best_action_dict[key] != -1:
                best_state_pair.append(key)
        
        best_states.append(best_state_pair)
        print(best_states)
        return best_states
    
    
    
    def extract_obs(self, level, obs_arr, C0_to_C1_dict):
        #for i in range(len(self.runner.agent.qs[level].success_traject)):
        state, obs = self.runner.env.reset()
        batch = self.runner.agent.qs[level].success_traject[-1]

        for j in range(len(batch)):
            obs_arr.append(obs[level])

            action = batch[j][1]
            
            C0_to_C1_dict[state[level]] = 1
            next_state, reward, done, info, obs = self.runner.env.step(action)
            
            state = next_state


        C0_to_C1_dict[state[level]] = 1


        return obs_arr, C0_to_C1_dict
    
    
    def fill_state_obs_dict(self, state_obs_dict, obs_arr):  
        for i in range(len(obs_arr)):
            state_hash = xxhash.xxh64(obs_arr[i]).hexdigest()
            #print(state_hash)
            #print(obs_arr[i])
            state_obs_dict[state_hash] = obs_arr[i]
            
            #if state_hash not in state_obs_dict_count:
                #state_obs_dict_count[state_hash] = 1
            #else:
                #state_obs_dict_count[state_hash] += 1
                
        return state_obs_dict
    
    
    
    def fill_causal_state_dict(self, C0_to_C1_dict, state_obs_dict, causal_graph, action_state_dict):
        for key in C0_to_C1_dict.keys():
            state_node = Node(key, state_obs_dict[key])
            
            causal_graph.add_node(state_node)
            
            action_state_dict[key] = []
    
    
    
    def add_trajectory_transition(self, level, trajectory, state_action):

        trajectory.add_state_action(state_action[0], state_action[1])

 
 
 
    def build_global_causal_graph(self, level, trajectory, env, causal_graph):
        batch = self.runner.agent.qs[level].success_traject[-1]

        state, _ = env.reset()
        state = state[level]
        for j in range(len(batch)):
            action = batch[j][1]
            
            trajectory.add_state_action(state, action)
 
            next_state, reward, done, info, obs = env.step(action)

            if causal_graph != None:
                causal_graph.add_next_state(state, next_state[level])
             
            state = next_state[level]
            
    
    
    def has_duplicates2(self, success_action_dict, max_value):
        duplicate_count = 0
        for key in success_action_dict:
            if success_action_dict[key] == success_action_dict[max_value]:
                duplicate_count += 1
                print(duplicate_count)
            
        
        return duplicate_count > 1
        
    
    
    def get_state_counts(self, full_batch, action_state_dict, counter_dict):
        for i in range(len(full_batch)):
            for j in range(len(full_batch[i])):
                state = full_batch[i][j][0]
                action = full_batch[i][j][1]
                
                action_state_dict[state].append(action)
        
                
        for key in action_state_dict.keys():
            count = Counter(action_state_dict[key])
            counter_dict[key] = count


        return counter_dict
    
    
