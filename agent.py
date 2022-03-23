"""
    This class represents the agent. It consists of n cycle-functions and n q-functions represented in q_function.py and cycle_function.py.
"""

from q_function import QFunction
from cycle_function import CycleFunction
from loop_counter import LoopCounter

import numpy as np
from operator import itemgetter

class Agent:
    def __init__(self, num_levels):
        self.num_levels = num_levels

        self.qs = []
        self.cs = []
        self.cs_success = []
        self.qs_u = []
        self.cs_u = []
        self.cs_u_success = []
        #self.combined = QFunction()

        for i in range(num_levels):
            self.qs.append(QFunction())
            self.cs.append(CycleFunction())
            self.cs_success.append(CycleFunction())
            self.qs_u.append(QFunction())
            self.cs_u.append(CycleFunction())
            self.cs_u_success.append(CycleFunction())

            self.cs[i].set_loop_counter(LoopCounter())
        # self.combined_agent = CombinedFunction()




    """
        Evaluate the agent at given state and return action.
    """
    def evaluate(self, state, level):
        cycle_q_vec = self.cs[level].qtable.get_value(state[level], None)
        q_vec_0 = self.qs[level].qtable.get_value(state[level], None)
        c_vec_2 = self.cs[2].qtable.get_value(state[2], None)
        #cycle_q_vec_2 = self.cs[2].qtable.get_value(state[2], None)
        #curiosity_vec = self.cs[2].curiosity_table.get_value(state[2], None)

        # if all(i < 0 for i in c_vec_2) and not all(i < 0 for i in cycle_q_vec) :
            # # best_action_value_arr = []

            # # for i in range(len(c_vec_2)):
                # # if not curiosity_vec[i]:
                    # # best_action_value_arr.append((c_vec_2[i], i))



            # # if len(best_action_value_arr) > 0:
                # # max_tuple = max(best_action_value_arr, key=itemgetter(0))
                # # best_action = max_tuple[1]
                # # best_action_v = max_tuple[0]

            # best_action_v = c_vec_2[np.argmax(c_vec_2)]

            # if best_action_v > -0.01:
                # print(state[1])
                # print(c_vec_2)
                # print("CUS2: ", self.cs[2].curiosity_table.get_value(state[2], None))

                # return np.array(c_vec_2), True


        #print(state)
        #print("C0: ", self.cs[0].qtable.get_value(state[0], None))
        #print("C1: ", self.cs[1].qtable.get_value(state[1], None))
        #print("C2: ", self.cs[2].qtable.get_value(state[2], None))
        #print("C2_EX: ", np.array(q_vec_0) + np.array(cycle_q_vec))
        #combined_q = self.combined.qtable.get_value(state[1], None)


        return np.array(q_vec_0) + np.array(cycle_q_vec), False #+ np.array(q_vec_2) #+ np.array(cycle_q_vec_2)


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
            info = transition[5][i]

            self.qs[i].batch.append((state, action, reward, next_state, obs, info))
            self.cs[i].batch.append((state, action, reward, next_state, obs, info))
            self.cs_success[i].batch.append((state, action, reward, next_state, obs, info))

            self.qs_u[i].batch.append((state, action, reward, next_state, obs, info))
            self.cs_u[i].batch.append((state, action, reward, next_state, obs, info))
            self.cs_u_success[i].batch.append((state, action, reward, next_state, obs, info))
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
    def update(self, global_steps, successful_episodes, alpha, gamma):
        #for i in range(self.num_levels):

        for i in range(self.num_levels):
            self.qs[i].update(alpha, gamma)
            self.cs[i].update_hash(global_steps, alpha, gamma, False)
            self.qs_u[i].update(1.0, gamma)
            self.cs_u[i].update_hash(global_steps, 1.0, gamma, True)

            if successful_episodes > 100:
                #print("UPDATE")
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


    def save_loop_counter(self, steps):
        for i in range(self.num_levels):
            self.cs[i].loop_counter.save("loop_hashtable"+ str(i)+".p")

    #def update_combined(self, state):
        #q_vec_0 = self.qs[0].qtable.get_value(state[0], None)
        #q_vec_1 = self.qs[1].qtable.get_value(state[1], None)

        #q_vec_sum = np.array(q_vec_0) + np.array(q_vec_1)

        #for i in range(len(q_vec_0)):
            #self.combined.qtable.set_value(state[1], i, q_vec_sum[i])
