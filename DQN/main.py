#!/usr/bin/env python3

"""
Usage:

$ . ~/env/bin/activate

Example pong command (~900k ts solve):
    python main.py \
        --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
        --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
        --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
        --batch_size 32 --gamma 0.99 --log_every 10000

Example cartpole command (~8k ts to solve):
    python main.py \
        --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
        --replay_size 5000 --starts_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
        --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
"""

import argparse
import math
import random
from copy import deepcopy

import numpy as np
import xxhash
import torch
import torch.optim as optim
from DQN.helpers import ReplayBuffer
#from helpers import ReplayBuffer, make_atari, make_gym_env, wrap_deepmind, wrap_pytorch
from DQN.envs import make_env, make_env_partial
from DQN.models import DQN, CnnDQN

from torch.utils.tensorboard import SummaryWriter



USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU: GPU requested and available.")
    dtype = torch.cuda.FloatTensor
    dtypelong = torch.cuda.LongTensor
else:
    print("NOT Using GPU: GPU not requested or not available.")
    dtype = torch.FloatTensor
    dtypelong = torch.LongTensor


class Agent:
    def __init__(self, env, q_network, target_q_network):
        self.env = env
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.num_actions = env.action_space.n

    def act(self, state, epsilon):
        """DQN action - max q-value w/ epsilon greedy exploration."""
        if random.random() > epsilon:
            state = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
            q_value = self.q_network.forward(state)
            print(q_value)
            return q_value.max(1)[1].data[0]
        return torch.tensor(random.randrange(self.env.action_space.n))


def compute_td_loss(agent, batch_size, replay_buffer, optimizer, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(np.float32(state)).type(dtype)
    next_state = torch.tensor(np.float32(next_state)).type(dtype)
    action = torch.tensor(action).type(dtypelong)
    reward = torch.tensor(reward).type(dtype)
    done = torch.tensor(done).type(dtype)

    # Normal DDQN update
    q_values = agent.q_network(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # double q-learning
    online_next_q_values = agent.q_network(next_state)
    _, max_indicies = torch.max(online_next_q_values, dim=1)
    target_q_values = agent.target_q_network(next_state)
    next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

    expected_q_value = reward + gamma * next_q_value.squeeze() * (1 - done)
    loss = (q_value - expected_q_value.data).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    
    
def compute_cycle_loss(agent, batch_size, batch, optimizer, gamma):
    last_batch = batch[-batch_size:]
    states, next_states_t, actions, dones = [], [], [], []

    cycle_penalty = np.zeros((batch_size, 1))
    loop_sizes = np.zeros((batch_size, 1))
    loop_sizes[:] = np.NaN
    
    for (state, next_state, action, done) in last_batch:
        states.append(torch.tensor(np.float32(state)).type(dtype))
        actions.append(torch.tensor(action).type(dtypelong))
        next_states_t.append(torch.tensor(np.float32(next_state)).type(dtype))
        dones.append(torch.tensor(done).type(dtype))
    
    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states_t = torch.stack(next_states_t)
    dones = torch.stack(dones)

    #Check cycles
    for i in range(len(batch) - batch_size):
        current_state = np.array(batch[i][0])
        
        for j in range(len(last_batch)): #(0 - 20)
            #print(states)
            #print(next_states[j])
            next_states = last_batch[j][1]
            
            
            cycles = xxhash.xxh64(next_states).hexdigest() == xxhash.xxh64(current_state).hexdigest() #(8,) 
            
            loop_size = cycles * (((j + len(batch) - batch_size) - i) + 1)
            #loop_size = loop_size.astype(np.float)
            #loop_size[loop_size == 0] = np.nan
            #print(rewards.shape)
            
            if loop_size > 0:
                loop_sizes[j] = -1.0 / loop_size
            else:
                loop_sizes[j] = 0.0
    
    
    cycle_penalty = loop_sizes 
    #print(cycle_penalty)
    cycle_penalty = torch.from_numpy(cycle_penalty).cuda()
        
    q_values = agent.q_network(states)
    print(q_values)
    q_value = q_values.gather(1, actions.unsqueeze(1))
    
    online_next_q_values = agent.q_network(next_states_t)
    _, max_indicies = torch.max(online_next_q_values, dim=1)
    target_q_values = agent.target_q_network(next_states_t)
    next_q_value = torch.gather(target_q_values, 1, max_indicies.unsqueeze(1))

    expected_q_value = cycle_penalty + gamma * next_q_value.squeeze() * (1 - dones)
    
    loss = (q_value - expected_q_value).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    # xxhash.xxh64(state).hexdigest()


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay
    )


def soft_update(q_network, target_q_network, tau):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(q_network, target_q_network):
    for t_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        if t_param is param:
            continue
        new_param = param.data
        t_param.data.copy_(new_param)


def run_gym(params):
    #if params.CnnDQN:
    env = make_env(params.env)
    #env = make_atari(params.env)
    #env = wrap_pytorch(wrap_deepmind(env))
    q_network = CnnDQN(env.observation_space.shape, env.action_space.n)
    target_q_network = deepcopy(q_network)
    #else:
    #    env = make_gym_env(params.env)
    #    q_network = DQN(env.observation_space.shape, env.action_space.n)
    #    target_q_network = deepcopy(q_network)

    if USE_CUDA:
        q_network = q_network.cuda()
        target_q_network = target_q_network.cuda()

    agent = Agent(env, q_network, target_q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    replay_buffer = ReplayBuffer(params.replay_size)
    batch = []


    losses, all_rewards = [], []
    episode_reward = 0
    state = env.reset()
    
    
    tb = SummaryWriter()

    for ts in range(1, params.max_ts + 1):
        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, ts
        )
        
        #action = agent.act(state, epsilon)
        #state_test = torch.tensor(np.float32(state)).type(dtype).unsqueeze(0)
        action = 5
        #q_value = agent.q_network.forward(state_test)
        #print(q_value)
        next_state, reward, done, _ = env.step(action)#env.step(int(action.cpu()))
        #env.render()
        replay_buffer.push(state, action, reward, next_state, done)
        batch.append((state, next_state, action, done))

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            tb.add_scalar("Reward", episode_reward, ts)

            episode_reward = 0
            batch = []

        if len(replay_buffer) > params.start_train_ts:
            # Update the q-network & the target network
            q_loss = compute_td_loss(
                agent, params.batch_size, replay_buffer, optimizer, params.gamma
            )
            
            tb.add_scalar("Q-Loss", q_loss, ts)

            
            cycle_loss = 0
            if len(batch) > params.batch_size:
                cycle_loss = compute_cycle_loss(
                    agent, params.batch_size, batch, optimizer, params.gamma
                )
                
                tb.add_scalar("Cycle Loss", cycle_loss, ts)

            
            loss = cycle_loss#q_loss + cycle_loss
            
            losses.append(loss.data)

            if ts % params.target_network_update_f == 0:
                hard_update(agent.q_network, agent.target_q_network)

        if ts % params.log_every == 0:
            out_str = "Timestep {}".format(ts)
            if len(all_rewards) > 0:
                out_str += ", Reward: {}".format(all_rewards[-1])
            if len(losses) > 0:
                out_str += ", TD Loss: {}".format(losses[-1])
            print(out_str)


def init_nn(env_name, atari):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=env_name)
    parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=10000)
    parser.add_argument("--start_train_ts", type=int, default=2000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=1400000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    run_gym(parser.parse_args())
