 


class GoalQTable:
    def __init__(self, params=[8,8,4]):
        self.q_table = QTable(params)
        
        
    def get_q_value(self, goal, state, action):
        return self.q_table.get_qtable()[goal][state][action]
        

    def set_q_value(self, goal, state, actions, value):
        self.q_table.get_qtable()[goal][state][action] = value
    
    def get_qtable(self):
        return self.q_table.get_qtable()


class QTable:
    def __init__(self, params):
        self.goal_qtable = GoalDict(params).get_goal_dict()
       
        for key in self.goal_qtable:
            self.goal_qtable[key] = {}
            for i in range(1, params[0] - 1):
                for j in range(1, params[1] - 1):
                    for k in range(params[2]):
                        self.goal_qtable[key][str(i)+str(j)+str(k)] = [0] * 7
                        
                        
    def get_qtable(self):
        return self.goal_qtable
    
    
class GoalDict:
    def __init__(self, params):
        self.goal_dict = {}
        
        for i in range(1, params[0] - 1):
            for j in range(1, params[1] - 1):
                for k in range(params[2]):
                    self.goal_dict[str(i)+str(j)+str(k)] = 0



    def get_goal_dict(self):
        return self.goal_dict
    
