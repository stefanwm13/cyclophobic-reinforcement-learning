import numpy as np

'''
    This collection of classes wraps different kinds of qtables and provides access to them.
'''

class QTableFactory:  
    @staticmethod
    def create(params=[8,8,4]):
        print(params)
        q_table = QTable(params)
        return q_table.get_qtable()
    
    @staticmethod 
    def create_loop_qtable(params=[8,8,4]):
        q_table = LoopQTable(params)
        return q_table
    
    @staticmethod
    def create_obs_qtable():
        return ObsQTable()
    
    @staticmethod
    def create_obs_count_table():
        return ObsCountQTable()


class QTable:
    def __init__(self, params):
        self.qtable = {}
        
        for i in range(1, params[0] - 1):
            for j in range(1, params[1] - 1):
                for k in range(params[2]):
                    self.qtable[str(i)+str(j)+str(k)] = [0] * 7
                        
                        
    def get_qtable(self):
        return self.qtable
    
    
    
class LoopQTable:
    def __init__(self, params):
        self.qtable = {}
        
        for i in range(1, params[0] - 1):
            for j in range(1, params[1] - 1):
                for k in range(params[2]):
                    self.qtable[str(i)+str(j)+str(k)] = 0
    
    def add_loop(self, state):
        self.qtable[state] = self.qtable[state] + 1
                        
    def get_qtable(self):
        return self.qtable
    
    

class ObsCountQTable:
    def __init__(self):
        self.qtable = {}
        
    
    def add_loop(self, state):
        if state not in self.qtable:
            self.qtable[state] = 0
        else:
            self.qtable[state] = self.qtable[state] + 1
                        
    def softmax(self, z):
        print(len(z.shape))
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div
    
    def get_softmax(self, state):
        
        counts = []
        index = 0
        state_index = 0
        for it_state in self.qtable:
            if it_state == state:
                state_index = index
            counts.append(self.qtable[it_state])
            index += 1
        
        if len(counts)>0:
            probs = self.softmax(np.asarray([counts]))
            #print(probs)
            #print(1 - probs[0][index-1])
        
            return 1 - probs[0][index-1]
        
        return 0
    
    def get_qtable(self):
        return self.qtable
        
    
    
class ObsQTable:
    def __init__(self):
        self.qtable = {}
    
    
    def set_value(self, state, action, value):
        if state in self.qtable:
            self.qtable[state][action] = value
        else:
            self.qtable[state] = [0] * 7
            self.qtable[state][action] = value
            
    
    def get_value(self, state, action):
        if action != None:
            if state in self.qtable:
                return self.qtable[state][action]
            else:
                self.qtable[state] = [0] * 7
                return self.qtable[state][action]
        else:
            if state in self.qtable:
                return self.qtable[state]
            else:
                self.qtable[state] = [0] * 7
                return self.qtable[state]
    
    def get_qtable(self):
        return self.qtable
