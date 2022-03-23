from vec_env import HashVectorizedEnvironment, CNNVectorizedEnvironment
from models import AtariCNN
from runner import Runner
from agent import Agent
from causal_helpers import CausalUtils

import pickle

class Train:
    def __init__(self, num_levels):
        self.agent = Agent(num_levels)

    def reset_agent(self, num_levels):
        self.agent = Agent(num_levels)


class Runnables:
    class RunnableUnlockPickup:
        def __init__(self):
            self.best_action_dict = {}
            self.causal_states = []
            self.secondary_causal_states = []
            self.runner_list = []

            self.bead_file_name = "saved/bead_UP.p"
            self.trainer_file_name = "saved/trainer_UP.p"
            self.trainer_agent_file_name = "saved/tt_UP.p"
            self.runner_file_name = "saved/runner_UP.p"


        def run(self, args, i, save, trainer):
            if save:
                for i in range(1):
                    print("NEW RUNNER")
                    runner = Runner(trainer.agent, args.n_episodes)
                    utils = CausalUtils(runner)
                    #runner.env = HashVectorizedEnvironment(3, "MiniGrid-UnlockPickup-5x5-v0", False)
                    #runner.env = HashVectorizedEnvironment(3, "MiniGrid-DoorKeyY-8x8-v0", False)
                    runner.env = HashVectorizedEnvironment(3, "MiniGrid-MultiRoom-N2-S4-v0", False)


                    print(runner)
                    runner.explore_and_navigate(trainer, i, alpha=0.2, gamma=0.99, epsilon=0.4, cycle=True, max_episodes=None, exploration_threshold=50, level=2)

                    self.runner_list.append(runner)

                    trainer.reset_agent(3)

                #self.save()

            causal_estimates = []
            for runner in self.runner_list:
                causal_estimates.append(runner.find_salient_state_action_pairs(1, -1))

            utils = CausalUtils(self.runner_list[0])

            self.best_action_dict, self.causal_states, self.secondary_causal_states = utils.get_best_actions(causal_estimates)

            print(self.best_action_dict)
            print(self.causal_states)

            return self.best_action_dict


        def test(self, args, i, trainer, runner):
            runner.env = HashVectorizedEnvironment(3, "MiniGrid-DoorKeyRand-11x11-v0", False)
            #print(runner)
            #runner.env = HashVectorizedEnvironment(3, "MiniGrid-UnlockPickupRand-8x8-v0", False)
            print(runner.agent)
            runner.use_and_navigate(trainer, i, alpha=0.2, gamma=0.99, epsilon=0.1, cycle=True, max_episodes=100000, exploration_threshold=50, level=0, causal_tools=(self.best_action_dict, self.causal_states, self.secondary_causal_states))
            #runner.run(i, 0.9, 0.99, 0.1, True, (self.best_action_dict, self.causal_states, non_causal_states), 1000, 0, 2)


        def save(self):
            pickle.dump(self.best_action_dict, open(self.bead_file_name, "wb"))
            pickle.dump(self.causal_states, open("saved/cs.p", "wb"))
            pickle.dump(self.secondary_causal_states, open("saved/scs.p", "wb"))

            #pickle.dump(best_states, open("bs.p", "wb"))
            #pickle.dump(self.trainer, open(self.trainer_file_name, "wb"))
            #pickle.dump(self.trainer.agent.qs[1].success_traject, open(self.trainer_agent_file_name, "wb"))
            pickle.dump(self.runner_list, open(self.runner_file_name, "wb"))


        def load(self):
            self.best_action_dict = pickle.load(open(self.bead_file_name, "rb"))
            self.causal_states = pickle.load(open("saved/cs.p", "rb"))
            self.secondary_causal_states = pickle.load(open("saved/scs.p", "rb"))

            #best_states = pickle.load(open("bs.p", "rb"))
            #self.trainer = pickle.load(open(self.trainer_file_name, "rb"))
            #self.success_traject = pickle.load(open(self.trainer_agent_file_name, "rb"))
            self.runner_list = pickle.load(open(self.runner_file_name, "rb"))

            return self.runner_list



    class RunnableAtari:
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
                self.runner.env = HashVectorizedEnvironment(1, "MontezumaRevenge-v0", True)
                #self.runner.env = HashVectorizedEnvironment(1, "MiniGrid-UnlockPickup-8x8-v0", False)

                self.runner.run(i, 0.9, 0.99, 0.2, True, None, None, 60, 0)
                self.save()

            self.best_action_dict, trajectory, non_causal_states = self.runner.perform_interventions(1)
            self.causal_states = self.runner.build_causal_timeline(self.best_action_dict, trajectory)

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


    class RunnableUnlockPickupCNN:

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
                net = AtariCNN(7)
                self.runner.env = CNNVectorizedEnvironment(3, "MiniGrid-DoorKeyY-8x8-v0", False, net)
                #self.runner.env = HashVectorizedEnvironment(3, "MiniGrid-UnlockPickup-8x8-v0", False)

                self.runner.run(i, 0.9, 0.99, 0.2, True, None, None, 60, 0)

            exit()
            # self.best_action_dict, trajectory, non_causal_states = self.runner.perform_interventions(1)
            # self.causal_states = self.runner.build_causal_timeline(self.best_action_dict, trajectory)


            # #self.best_action_dict_3, trajectory3 = self.runner.perform_interventions(2)

            # #self.causal_states_3 = self.runner.build_causal_timeline(self.best_action_dict_3, trajectory3)

            # #exit()

            # #exit()

            # self.trainer.reset_agent(3)
            # self.runner.change_agent(self.trainer.agent)
            # #self.runner.env = HashVectorizedEnvironment(3, "MiniGrid-DoorKeyRand-8x8-v0", False)
            # self.runner.env = HashVectorizedEnvironment(3, "MiniGrid-UnlockPickupRand-8x8-v0", False)

            # self.runner.run(i, 0.9, 0.99, 0.1, True, (self.best_action_dict, self.causal_states, non_causal_states), 1000, 100, 0)


            return self.best_action_dict

            #print("FINISHED RED")
            #self.causal_dict, self.causal_state_indicator = self.runner.build_causal_timeline(self.best_action_dict)

            #self.runner.env = HashVectorizedEnvironment(2, "MiniGrid-UnlockPickup-v0")
            #self.runner.run(i, 0.9, 0.99, 0.2, True, self.best_action_dict, 2000, 60)

            return self.runner
