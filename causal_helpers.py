import networkx as nx
import matplotlib.pyplot as plt
import copy
import xxhash
import numpy as np

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


    def extract_obs(self, level, index, obs_arr, C0_to_C1_dict):
        #for i in range(len(self.runner.agent.qs[level].success_traject)):
        #state, obs = self.runner.env.reset()
        batch = self.runner.agent.qs[level].success_traject[index]

        for j in range(len(batch)):
            #obs_arr.append(obs[level])
            obs_arr.append(batch[j][4])

            state = batch[j][0]
            action = batch[j][1]

            C0_to_C1_dict[state] = 1
            #next_state, reward, done, info, obs = self.runner.env.step(action)

            #state = next_state

        #C0_to_C1_dict[state[level]] = 1

        return obs_arr, C0_to_C1_dict


    def fill_state_obs_dict(self, state_obs_dict, obs_arr):
        for i in range(len(obs_arr)):
            state_hash = xxhash.xxh64(obs_arr[i]).hexdigest()
            state_obs_dict[state_hash] = obs_arr[i]

        return state_obs_dict


    def add_trajectory_transition(self, level, trajectory, state_action):

        trajectory.add_state_action(state_action[0], state_action[1])


    def build_trajectories(self, level, index, trajectory):
        batch = self.runner.agent.qs[level].success_traject[index]

        for j in range(len(batch)):
            state = batch[j][0]
            action = batch[j][1]

            trajectory.add_state_action(state, action)


    def calculate_total_causal_estimate(self, causal_estimates):
        total_causal_estimate = [0] * 7

        # Sum up all causal estimates
        for causal_dict, _, _ in causal_estimates:
            for key in causal_dict:
                total_causal_estimate[int(key)] += causal_dict[key]

        total_causal_estimate[:] = [x / 5 for x in total_causal_estimate]
        print("Total Causal Estimate: ",total_causal_estimate)


        # Select actions where cycle value is sufficiently low
        causal_actions = []
        for i in range(len(total_causal_estimate)):
            if total_causal_estimate[i] > -0.1:
                causal_actions.append(i)

        print("Causal Actions: ", causal_actions)

        return causal_actions


    """
        Check injectivity or surjectivity
    """
    def check_projection(self):
        secondary_causal_states = []
        secondary_causal_actions = []
        pass


    """
        Build the dict that contains the salient state-action pairs.
    """
    def build_best_action_dict(self, causal_estimates, causal_actions):
        best_action_dict = {}
        success_trajectory = causal_estimates[4][1]

        for i in range(len(success_trajectory.state_action_vec)):
            action = success_trajectory.state_action_vec[i][1]
            state = success_trajectory.state_action_vec[i][0]
            #print("STATE: ", state)
            if action in causal_actions: #or action in secondary_causal_actions and state == secondary_causal_states[4]:
                best_action_dict[state] = action

        causal_state_list = []
        for key in best_action_dict:
            #if key != secondary_causal_states[4]:
            causal_state_list.append(key)

        # Get real secondary causal state
        #secondary_causal_states = list(set(secondary_causal_states) - set(causal_state_list))

        return best_action_dict, causal_state_list#, secondary_causal_states


    def get_best_actions(self, causal_estimates):
        causal_actions = self.calculate_total_causal_estimate(causal_estimates)

        # Sum up total parent counts for each state
        total_p_counts = {}
        for _, _, parent_count_dict in causal_estimates:
            print(parent_count_dict)
            for key in parent_count_dict:
                if key not in total_p_counts.keys():
                    total_p_counts[key] = np.asarray(parent_count_dict[key])
                else:
                    total_p_counts[key] = total_p_counts[key] + np.asarray(parent_count_dict[key])


        print("Total Parent Counts: ", total_p_counts)
        self.check_projection()

        #print(success_trajectory.state_action_vec)

        best_action_dict, causal_state_list = \
        self.build_best_action_dict(causal_estimates, causal_actions)

        return best_action_dict, causal_state_list, []


    # def get_state_counts(self, full_batch, action_state_dict, counter_dict):
    #     for i in range(len(full_batch)):
    #         for j in range(len(full_batch[i])):
    #             state = full_batch[i][j][0]
    #             action = full_batch[i][j][1]
    #
    #             action_state_dict[state].append(action)
    #
    #
    #     for key in action_state_dict.keys():
    #         count = Counter(action_state_dict[key])
    #         counter_dict[key] = count
    #
    #
    #     return counter_dict
