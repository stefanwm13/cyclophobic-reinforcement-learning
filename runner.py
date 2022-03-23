import random
import numpy as np
import copy
import time

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from causal_helpers import CausalUtils, Trajectory
from causal_graph import Node, CausalGraph

from logger import Logger


"""
    This class wraps the main loop and interventional loop of the agent.
"""

class Runner:
    def __init__(self, agent, max_episodes):
        self.explore = True
        self.burnin = True

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
        Check which action has to be performed at the salient state-action pair
    """
    def use_salient_state_action_pair(self, object_state, action):
        #Execute actions that are relevant for current state of agent
        current_causal_states = self.causal_states[self.causal_index_begin:self.causal_index_end]

        if object_state in current_causal_states:
            if object_state in self.secondary_causal_states and not self.causal_dict[object_state]:
                action = self.best_action_dict[object_state]
                self.causal_dict[object_state] = 1
            elif object_state not in self.secondary_causal_states:
                action = self.best_action_dict[object_state]
                self.causal_dict[object_state] = 1

            #print(self.causal_dict)
            # Check if index of current states needs to be increased
            sum = 0
            for i in range(self.causal_index_end):
                sum += self.causal_dict[self.causal_states[i]]

            if sum == self.causal_index_end and sum < len(self.causal_states):
                self.causal_index_begin += 1
                self.causal_index_end += 1

        return action


    """
        STAGE 3: The agent uses the learnt salient state-action pairs for navigation
    """

    def use_and_navigate(self, trainer, iterations, alpha, gamma, epsilon, cycle, max_episodes, exploration_threshold, level, causal_tools):
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cycle = cycle

        self.utils = CausalUtils(self)

        # Get sets that represent salient state-action pairs
        if causal_tools != None:
            self.best_action_dict = causal_tools[0]
            self.causal_states = causal_tools[1]
            self.secondary_causal_states = causal_tools[2]

        if max_episodes != None:
            self.max_episodes = max_episodes

        episodes_successful = 0
        global_steps = 0

        # EPISODE BEGIN
        for i in range(self.max_episodes):
            episode_steps = 0
            done = 0

            # Initialize indeces which select current salient state-action pair
            self.causal_index_begin = 0
            self.causal_index_end = 2

            if causal_tools != None and self.causal_states != None:
                self.causal_dict = {}
                for j in range(len(self.causal_states)):
                    self.causal_dict[self.causal_states[j]] = 0

            # Initialize trajectories representing hashes for the cropped views
            self.trajectory_nxn = Trajectory()
            self.trajectory = Trajectory()

            # Reset agents history
            self.agent.clear_batch()

            print("New episode: ", str(self.iterations) + "  " + str(i+1) + "  " + str(global_steps))

            # Reset environment (start)
            state, obs = self.env.reset()

            # (For plotting graphs)
            if global_steps >= 500000:
                break

            # STEPS
            while np.sum(done) == 0:
                #Sample action
                if np.random.uniform(0,1) < self.epsilon and self.explore:
                    action = random.randint(0, self.env.get_action_space() - 1)
                else:
                    b, show = np.array(self.agent.evaluate(state, level))
                    action = np.argmax(b)
                    #if show and episodes_successful > exploration_threshold:
                    #    plt.imshow(obs[2])
                    #    plt.show()

                # STAGE 3: If we are in third stage use salient state action pair
                if causal_tools != None:
                    action = self.use_salient_state_action_pair(state[1], action)

                # Execute action in environment
                time.sleep(trainer.stop_time)
                next_state, reward, done, info, next_obs = self.env.step(action)

                # Save trajectories to history
                #self.utils.add_trajectory_transition(2, self.trajectory_nxn, ((state[2], obs[2]), action))
                #self.utils.add_trajectory_transition(1, self.trajectory, ((state[1], obs[1]), action))

                self.agent.store_transition((state, action, reward, next_state, obs, info))
                self.agent.update(global_steps, episodes_successful, self.alpha, self.gamma)

                # (Logging: Show windows and state values)
                if trainer.show_window:
                    print(state[1] + "                  " + next_state[1])
                    print("CS0: ",self.agent.cs[0].qtable.get_value(state[0], None))
                    print("CS1: ",self.agent.cs[1].qtable.get_value(state[1], None))
                    print("CS2: ",self.agent.cs[2].qtable.get_value(state[2], None))
                    print("CUS2: ",self.agent.cs[2].curiosity_table.get_value(state[2], None))

                    if obs != None:
                        self.env.show_windows(obs)

                # Store best batches in history separately to replay them
                if np.sum(reward) > 0:
                    self.agent.store_best_batch()
                    episodes_successful += 1

                # Update state and obs to successors
                state = next_state
                obs = next_obs
                episode_steps = episode_steps + 1
                global_steps = global_steps + 1

            # (Log performance to file)
            #self.logger.log("results/ablation_cycle_norm/R-DoorKey-11x11_Norm_0" + "_run" + "0" + ".csv", global_steps, reward[0])

            # Stop exploring after threshold
            if episodes_successful > exploration_threshold:
                self.explore = False

            # Continue exploration if convergence not reached at defined exploration threshold
            if episodes_successful > exploration_threshold and reward[0] == 0:
                self.explore = True

            # Update agent with one of the best trajectories of the episode (Not really necessary)
            self.agent.update_best(self.alpha, self.gamma)

            # (Log loop sizes)
            #self.agent.save_loop_counter(global_steps)

            print("EPISODES SUCCESSFUL: ", str(episodes_successful) + "  " + str(reward[0]))


    """
        STAGE 1: The agent explores and learns the value functions
    """
    def explore_and_navigate(self, trainer, iterations, alpha, gamma, epsilon, cycle, max_episodes, exploration_threshold, level):
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.cycle = cycle

        self.utils = CausalUtils(self)

        if max_episodes != None:
            self.max_episodes = max_episodes

        episodes_successful = 0
        global_steps = 0

        tc_count = {}
        # EPISODE BEGIN
        for i in range(self.max_episodes):
            episode_steps = 0
            done = 0

            # Initialize trajectories representing hashes for the cropped views
            self.trajectory_nxn = Trajectory()
            self.trajectory = Trajectory()

            # Reset agents history
            self.agent.clear_batch()

            print("New episode: ", str(self.iterations) + "  " + str(i+1) + "  " + str(global_steps))

            # Reset environment (start)
            state, obs = self.env.reset()

            # (For plotting graphs)
            if global_steps >= 500000:
                break

            # STEPS
            while np.sum(done) == 0:
                #Sample action
                if np.random.uniform(0,1) < self.epsilon and self.explore:
                    action = random.randint(0, self.env.get_action_space() - 1)
                else:
                    b, show = np.array(self.agent.evaluate(state, level))
                    action = np.argmax(b)

                # Execute action in environment
                time.sleep(trainer.stop_time)
                next_state, reward, done, info, next_obs = self.env.step(action)

                # Save trajectories to history
                self.utils.add_trajectory_transition(2, self.trajectory_nxn, ((state[2], obs[2]), action))
                self.utils.add_trajectory_transition(1, self.trajectory, ((state[1], obs[1]), action))

                self.agent.store_transition((state, action, reward, next_state, obs, info))
                self.agent.update(global_steps, episodes_successful, self.alpha, self.gamma)

                # (Logging: Show windows and state values)
                if trainer.show_window:
                    print(state[1] + "                  " + next_state[1])
                    print("CS0: ",self.agent.cs[0].qtable.get_value(state[0], None))
                    print("CS1: ",self.agent.cs[1].qtable.get_value(state[1], None))
                    print("CS2: ",self.agent.cs[2].qtable.get_value(state[2], None))
                    print("CUS2: ",self.agent.cs[2].curiosity_table.get_value(state[2], None))
                    print("TC_COUNT: ", tc_count)

                    if obs != None:
                        self.env.show_windows(obs)


                if state[1] not in tc_count:
                    tc_count[state[1]] = np.zeros(7)
                    tc_count[state[1]] =  np.array(self.agent.cs[1].qtable.get_value(state[1], None))
                else:
                    tc_count[state[1]] =  np.array(self.agent.cs[1].qtable.get_value(state[1], None))

                # Store best batches in history separately to replay them
                if np.sum(reward) > 0:
                    self.agent.store_best_batch()
                    episodes_successful += 1

                # Update state and obs to successors
                state = next_state
                obs = next_obs
                episode_steps = episode_steps + 1
                global_steps = global_steps + 1

            # (Log performance to file)
            #self.logger.log("results/ablation_cycle_norm/R-DoorKey-11x11_Norm_0" + "_run" + "0" + ".csv", global_steps, reward[0])

            # Stop exploring after threshold
            if episodes_successful > exploration_threshold:
                self.explore = False

            # Continue exploration if convergence not reached at defined exploration threshold
            if episodes_successful > exploration_threshold and reward[0] == 0:
                self.explore = True

            # Update agent with one of the best trajectories of the episode (Not really necessary)
            self.agent.update_best(self.alpha, self.gamma)

            # (Log loop sizes)
            #self.agent.save_loop_counter(global_steps)

            print("EPISODES SUCCESSFUL: ", str(episodes_successful) + "  " + str(reward[0]))


    """
        STAGE 2: The agent extracts the salient state-action pairs from the first stage
    """
    def find_salient_state_action_pairs(self, level, index):
        obs_arr = []
        obs_arr_nxn = []
        self.state_obs_dict = {}
        self.state_obs_dict_nxn = {}
        C0_to_C1_dict = {}
        C0_to_C1_dict_nxn = {}

        self.trajectory = Trajectory()
        self.trajectory_nxn = Trajectory()
        self.utils = CausalUtils(self)

        obs_arr, C0_to_C1_dict = self.utils.extract_obs(level, index, obs_arr, C0_to_C1_dict)
        self.global_obs_arr = self.global_obs_arr + obs_arr

        obs_arr_nxn, C0_to_C1_dict_nxn = self.utils.extract_obs(2, index, obs_arr_nxn, C0_to_C1_dict_nxn)

        print(C0_to_C1_dict.keys())

        self.state_obs_dict = self.utils.fill_state_obs_dict(self.state_obs_dict, obs_arr)
        self.state_obs_dict_nxn = self.utils.fill_state_obs_dict(self.state_obs_dict_nxn, obs_arr_nxn)#

        self.utils.build_trajectories(level, index, self.trajectory)
        self.utils.build_trajectories(2, index, self.trajectory_nxn)

        ##### Perform interventions
        state_action_vec = None
        best_action_dict = {}
        cycle_value_dict = {}
        total_cycle_value_dict = {}
        action_occurence = [0] * 7

        actions_dict = {}
        secondary_cs_p_counts = {}

        success_states = []
        causal_state_action_dict = {}

        # Get all current states of successful trajectory
        for i in range(len(self.trajectory.state_action_vec)):
            success_states.append(self.trajectory.state_action_vec[i][0])

        # Initialize arrays to save total value dict
        for i in range(7):
            cycle_value_dict[str(i)] = 0
            total_cycle_value_dict[str(i)] = 0

        print(self.trajectory.state_action_vec)

        for key in C0_to_C1_dict.keys():
            c1_q_s_value = self.agent.cs_success[level].qtable.get_value(key, None)
            c1_q_value = self.agent.cs[level].qtable.get_value(key, None)

            if key not in success_states:
                continue

             # Get actions that are part of the successful trajectory for every key and do not cause cycle
            actions = []
            for i in range(len(self.trajectory.state_action_vec)):
                if key == self.trajectory.state_action_vec[i][0]:
                    actions.append(self.trajectory.state_action_vec[i][1])

            actions = list(set(actions))

            print("KEY: ", key)

            print("A: ", actions)
            print("C1_U_S: ", c1_q_s_value)
            print("C1_U: ", c1_q_value)

            # Calculate total cycle value by summing up cycle value for each state-action pair for the current key
            for action in actions:
                action_occurence[action] = action_occurence[action] + 1 #To be removed perhaps
                total_cycle_value_dict[str(action)] += c1_q_s_value[action]

            print(total_cycle_value_dict)

            self.get_secondary_causal_states(key, secondary_cs_p_counts)

        return total_cycle_value_dict, self.trajectory, secondary_cs_p_counts


    '''
        Check state_action pairs for injectivity
    '''
    def get_secondary_causal_states(self, key, secondary_cs_p_counts):
        action_parent_count = [0] * 7

        # Go through trajectory and check if view above can be mapped to different actions i.e. does W have several parents from V
        for i in range(len(self.trajectory.state_action_vec)):
            if key == self.trajectory.state_action_vec[i][0]:
                nxn_key = self.trajectory_nxn.state_action_vec[i][0]

                print("NXN_KEY: ", nxn_key)
                print("C1_U_2: ", self.agent.cs[2].qtable.get_value(nxn_key, None))

                parent_action = self.trajectory_nxn.state_action_vec[i][1]

                print("PARENT_ACTION: ", parent_action)

                #parent_count += 1
                action_parent_count[parent_action] += 1

                #plt.imshow(self.state_obs_dict_nxn[nxn_key])
                #plt.show()

        secondary_cs_p_counts[key] = action_parent_count
        print("P_COUNT: ", action_parent_count)
