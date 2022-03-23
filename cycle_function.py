from function import TabularFunction
from qtable import QTableFactory

class CycleFunction(TabularFunction):
    def __init__(self):
        super().__init__()
        self.cycle = None
        self.cycle_state = None
        self.cycle_count = 0
        self.max_cycle_count = 1
        self.loop_size = 1
        self.update_index = -1

        self.cycle_hash_function = {}
        self.curiosity_table = QTableFactory.create_obs_qtable()
        self.loop_counter = None


    def set_loop_counter(self, loop_counter):
        self.loop_counter = loop_counter


    def reset_hyperparameters(self):
        self.cycle = None
        self.cycle_state = None
        self.cycle_count = 0

        self.loop_size = 1
        self.update_index = -1


    def backup(self, i, alpha, gamma, unnormalized):
        state = self.batch[i][0]
        action = self.batch[i][1]

        if i < len(self.batch) - 1:
            next_action = self.batch[i+1][1]

        reward = self.batch[i][2]
        next_state = self.batch[i][3]

        # Add reward if cycle is detected
        if self.cycle_state != None and next_state == self.cycle_state:
            reward -= 1.0

        # Calculate new q-value and update q_table
        old_q_value = self.qtable.get_value(state, action)
        if i < len(self.batch) - 1:
            # Do SARSA or Q-Learning update
            if self.cycle_state != None or self.cycle_state == False:
                print("SARSA")
                next_q_value = self.qtable.get_value(next_state, next_action)
            else:
                print("MAX")
                next_q_value = np.max(self.qtable.get_value(next_state, None))

            new_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_value)
        else:
            #print("LAST")
            # For last state in batch new value is the reward
            # if self.cycle_count > 0:
            #     normalization = self.loop_size / self.cycle_count
            #     if normalization < 1:
            #         normalization = 1
            #
            #     cycle_penalty = (reward / normalization)
            # else:
            #     cycle_penalty = (reward / self.loop_size)
            cycle_penalty = (reward / self.loop_size)
            # if self.cycle_count > 0:
            #     normalization = self.cycle_count
            #     cycle_penalty = (reward / self.loop_size) * normalization
            # else:
            #     cycle_penalty = (reward / self.loop_size)


            new_value = (1 - alpha) * old_q_value + alpha * cycle_penalty

        #Update Q-Table
        self.qtable.set_value(state, action, new_value)

        if self.loop_size == 1:
            self.curiosity_table.set_value(state, action, 1.0)



    # def update_hash(self, steps, alpha, gamma, unnormalized):
        # last_state = self.batch[len(self.batch)-1][0]
        # last_next_state = self.batch[len(self.batch)-1][3]

        # if last_state not in self.cycle_hash_function:
            # #print("NEW: ",last_state)
            # self.cycle_hash_function[last_state] = [steps, 1]
            # self.cycle_state = None
            # self.cycle = False
            # self.loop_size = 1
            # self.update_index = -1
        # else:
            # last_state_info = self.cycle_hash_function[last_state]

            # last_steps = last_state_info[0]
            # last_count = last_state_info[1]

            # #print(last_steps)
            # #print(steps)
            # self.cycle_count = last_count
            # self.update_index = len(self.batch) - 1
            # self.loop_size = (self.update_index - last_steps) # / last_count

            # # if self.loop_size < 1:
                # # self.loop_size = 1

            # print(self.loop_size)

            # self.cycle_state = self.batch[len(self.batch)-1][3]
            # self.cycle = True

            # self.cycle_hash_function[last_state][0] = steps
            # self.cycle_hash_function[last_state][1] = self.cycle_hash_function[last_state][1] + 1

        # if self.cycle:
            # self.backup(self.update_index, alpha, gamma, unnormalized)


        # self.reset_hyperparameters()



    def update_hash(self, steps, alpha, gamma, unnormalized):
        state_list = [self.batch[i][0] for i in range(len(self.batch))]

        for i in range(len(state_list)):
            if self.batch[-1][3] == state_list[i]:
                self.update_index = len(self.batch) - 1
                #print(update_index)
                #print(i)
                self.loop_size = (self.update_index - i) + 1
                self.cycle_count += 1

                if self.cycle_count > self.max_cycle_count:
                    self.max_cycle_count = self.cycle_count

                self.cycle_state = self.batch[len(self.batch)-1][3]
                self.cycle = True

        if self.cycle:

            #self.loop_size = self.loop_size

            #if self.loop_size < 1:
            #   self.loop_size = 1

            if self.loop_counter is not None:
                pos = self.batch[-1][-1]['pos_dir'][0]
                self.loop_counter.add_loop_count(self.loop_size, pos ,steps)

            self.backup(self.update_index, alpha, gamma, unnormalized)


        self.reset_hyperparameters()
