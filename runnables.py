from vec_env import HashVectorizedEnvironment, CNNVectorizedEnvironment
from models import AtariCNN
from runner import Runner

import pickle

'''
    This class defines runnables which contain the training procedure for different environments
'''

class Runnables:
    
    class RunnableDoorKey:
        
        def __init__(self, trainer, runner, utils):
            self.trainer = trainer
            
            self.runner = runner
            self.utils = utils
            
            self.best_action_dict = {} 
            self.causal_states = []


            self.bead_file_name = "saved/bead_DK.p"
            self.trainer_file_name = "saved/trainer_DK.p"
            self.trainer_agent_file_name = "saved/tt_DK.p"
            self.runner_file_name = "saved/runner_DK.p"
            

        def run(self, args, i, save):
            if save:
                self.runner.env = HashVectorizedEnvironment(2, "MiniGrid-UnlockPickup-8x8-v0")
                self.runner.run(i, 0.9, 0.99, 0.2, True, None, None, 50) 
           
                self.save()
            
            print("FINISHED YELLOW")
            self.best_action_dict = self.runner.perform_interventions()
            self.causal_states = self.runner.build_causal_timeline(self.best_action_dict)
            
            self.runner.env = HashVectorizedEnvironment(2, "MiniGrid-UnlockPickupRand-8x8-v0")
            self.runner.run(i, 0.9, 0.99, 0.2, True, self.best_action_dict, 2000, 60)
            
    
        def save(self):
            print("SAVING")
            pickle.dump(self.best_action_dict, open(self.bead_file_name, "wb"))
            #pickle.dump(best_states, open("bs.p", "wb"))
            pickle.dump(self.trainer, open(self.trainer_file_name, "wb"))
            pickle.dump(self.trainer.agent.qs[1].success_traject, open(self.trainer_agent_file_name, "wb"))
            pickle.dump(self.runner, open(self.runner_file_name, "wb"))

    
    
        def load(self):
            self.best_action_dict = pickle.load(open(self.bead_file_name, "rb"))
            #best_states = pickle.load(open("bs.p", "rb"))
            self.trainer = pickle.load(open(self.trainer_file_name, "rb"))
            self.success_traject = pickle.load(open(self.trainer_agent_file_name, "rb"))
            self.runner = pickle.load(open(self.runner_file_name, "rb"))
        
        
        
    class RunnableUnlockPickup:
        
        def __init__(self, trainer, runner, utils):
            self.trainer = trainer
            self.runner = runner
            self.utils = utils

            self.best_action_dict = {} 
            self.causal_state_indicator = []
            self.causal_dict = {}
            
            self.bead_file_name = "saved/bead_UP.p"
            self.trainer_file_name = "saved/trainer_UP.p"
            self.trainer_agent_file_name = "saved/tt_UP.p"
            self.runner_file_name = "saved/runner_UP.p"

            
            
        def run(self, args, i, save):
            if save:
                self.runner.env = HashVectorizedEnvironment(1, "MiniGrid-UnlockPickup-8x8-v0", False)
                self.runner.run(i, 0.9, 0.99, 0, True, None, None, 60, 0)
                self.save()
                
            self.best_action_dict, trajectory, non_causal_states = self.runner.perform_interventions(1)

            self.causal_states = self.runner.build_causal_timeline(self.best_action_dict, trajectory)
            
            self.trainer.reset_agent(3)
            self.runner.change_agent(self.trainer.agent)
            self.runner.env = HashVectorizedEnvironment(3, "MiniGrid-UnlockPickupRand-8x8-v0", False)

            self.runner.run(i, 0.9, 0.99, 0, True, (self.best_action_dict, self.causal_states, non_causal_states), 3000, 10, 0)

            
            return self.best_action_dict

        
        
        def save(self):
            pickle.dump(self.best_action_dict, open(self.bead_file_name, "wb"))
            #pickle.dump(best_states, open("bs.p", "wb"))
            pickle.dump(self.trainer, open(self.trainer_file_name, "wb"))
            pickle.dump(self.trainer.agent.qs[1].success_traject, open(self.trainer_agent_file_name, "wb"))
            pickle.dump(self.runner, open(self.runner_file_name, "wb"))
        
        
        def load(self):
            self.best_action_dict = pickle.load(open(self.bead_file_name, "rb"))
            #best_states = pickle.load(open("bs.p", "rb"))
            self.trainer = pickle.load(open(self.trainer_file_name, "rb"))
            self.success_traject = pickle.load(open(self.trainer_agent_file_name, "rb"))
            self.runner = pickle.load(open(self.runner_file_name, "rb"))
            
            return self.runner

