from function import TabularFunction

'''
    This class represents an agent that is updated via the cyclophobic intrinsic reward.
'''

class CycleFunction(TabularFunction):
    def __init__(self):
        super().__init__()
        self.cycle = None
        self.cycle_state = None
        self.cycle_count = 0
        self.loop_size = 1
        self.update_index = -1
        
        self.cycle_hash_function = {}
        
        
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
            if not unnormalized:
                new_value = (1 - alpha) * old_q_value + alpha * reward
            else:
                new_value = old_q_value + reward
        #Update Q-Table
        self.qtable.set_value(state, action, new_value / self.loop_size) 


    def update_hash(self, steps, alpha, gamma, unnormalized):
        state_list = [self.batch[i][0] for i in range(len(self.batch))]
       
        for i in range(len(state_list)):
            if self.batch[len(self.batch)-1][3] == state_list[i]:
                self.update_index = len(self.batch) - 1

                self.loop_size = (self.update_index - i) + 1
                self.cycle_count += 1
                
                self.cycle_state = self.batch[len(self.batch)-1][3]
                self.cycle = True 
       
        if self.cycle:
            self.loop_size = self.loop_size / self.cycle_count
            
            if self.loop_size < 1:
                self.loop_size = 1

            self.backup(self.update_index, alpha, gamma, unnormalized)
            
               
        
        self.reset_hyperparameters()
       
       
        
