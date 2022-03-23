import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable
import xxhash

from A2C.utils import mean_std_groups

import sys
sys.path.append("/home/stefan/Documents/cyclophobic-working")
from gym_minigrid.window import Window


def train(args, net, optimizer, env, env_p, cuda, best_action_dict):
    state = []
    
    obs = env.reset()
    #obs_p = env_p.reset()
    for i in range(args.num_workers):
        state.append(xxhash.xxh64(obs[i]).hexdigest())

    if args.plot_reward:
        total_steps_plt = []
        ep_reward_plt = []

    steps = []
    batch = []
    
    total_steps = 0
    ep_rewards = [0.] * args.num_workers
    render_timer = 0
    plot_timer = 0
    loss = 0
    epoch = 0
    
    #h = net.init_hidden(1)
    
    window = Window("gym_minigrid_level_1")

    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs = Variable(torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.)
            if cuda: obs = obs.cuda()

            # network forward pass
            #h = tuple([e.data for e in h])
#            policies, values, h = net(obs, h)
            #print(obs.shape)
            policies, values_ex = net(obs)

            policy = policies.view(args.num_workers, 1, 7)[:,0]
            #print(policy)
            probs = Fnn.softmax(policy, dim=1)
            actions = probs.multinomial(1).data
            #print(actions)

            #actions= torch.from_numpy(np.zeros((8,1))).type(torch.int64).cuda()

            #time.sleep(1)
            #env.render()
            #env_p.render()
            # gather env data, reset done envs and update their obs
            obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            #obs_p,rewards_p, dones_p, infos_p = env_p.step(actions.cpu().numpy())
            #window.show_img(obs.squeeze(0))
            
            next_state = []
            for i in range(args.num_workers):
                next_state.append(xxhash.xxh64(obs[i]).hexdigest())
            
            rewards_in = np.zeros_like(rewards)
          
          
            # reset the LSTM state for done envs
            masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1)
            if cuda: masks = masks.cuda()

            total_steps += args.num_workers
            for i, done in enumerate(dones):
                #if rewards[i] != 0.5:
                ep_rewards[i] += rewards[i]
                if done:
                    #batch = []
                    if args.plot_reward:
                        total_steps_plt.append(total_steps)
                        ep_reward_plt.append(ep_rewards[i])
                    ep_rewards[i] = 0

            if args.plot_reward:
                plot_timer += args.num_workers # time on total steps
                if plot_timer == 10000:
                    print("10000 steps passed")
                    x_means, _, y_means, y_stds = mean_std_groups(np.array(total_steps_plt), np.array(ep_reward_plt), args.plot_group_size)
                    fig = plt.figure()
                    epoch += 1
                    fig.set_size_inches(8, 6)
                    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 6))
                    plt.errorbar(x_means, y_means, yerr=y_stds, ecolor='xkcd:blue', fmt='xkcd:black', capsize=5, elinewidth=1.5, mew=1.5, linewidth=1.5)
                    plt.title('Training progress')
                    plt.xlabel('Total steps')
                    plt.ylabel('Episode reward')
                    plt.savefig('doorkey_8x8.png', dpi=200)
                    plt.clf()
                    plt.close()
                    plot_timer = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, 'model.tar')

            rewards = torch.from_numpy(rewards).float().unsqueeze(1)
            rewards_in = torch.from_numpy(rewards_in).float().unsqueeze(1)

            if cuda: rewards = rewards.cuda()
            if cuda: rewards_in = rewards_in.cuda()

            
            #value, index = torch.max(rewards.squeeze(1).unsqueeze(0), dim=1)
            
            batch.append((state, next_state))
            steps.append((rewards, rewards_in, masks, actions, policy, policies, values_ex))

            state = next_state
            
        ######### TD LOSS    
        reward_in = check_cycle(args, batch)
        reward_in = torch.from_numpy(reward_in.squeeze(1)).float()

        if cuda: reward_in = reward_in.cuda()
        #print(reward_in)


        masks_b, actions_b, policy_b = [], [], []
        for (_, _, masks, actions, policy, _, _) in steps:
            masks_b.append(masks)
            actions_b.append(actions)
            policy_b.append(policy)
            
        masks_b = torch.stack(masks_b)
        actions_b = torch.stack(actions_b)
        policy_b = torch.stack(policy_b)

        q_value = policy_b.squeeze(1).gather(1, actions_b.squeeze(1))
        
        td = q_value.squeeze(1) - reward_in.detach()
        td_loss = td.pow(2).mean()
        
        
        #print(td_loss)
        
        #####################
        
        ######## POLICY GRADIENT LOSS
        final_obs = Variable(torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.)
        if cuda: final_obs = final_obs.cuda()
        #_, final_values, h = net(final_obs, h)
        _, final_values_ex = net(final_obs)
        steps.append((None, None, None, None, None, None, final_values_ex))

        actions, policy, policies, values_ex, returns_ex, advantages_ex = process_rollout(args, steps, cuda)
        
        advantages = advantages_ex 
        
        probs = Fnn.softmax(policy)
        log_probs = Fnn.log_softmax(policy)
        log_action_probs = log_probs.gather(1, Variable(actions))

        policy_loss = (-log_action_probs * Variable(advantages)).sum()
        values_ex_loss = (.5 * (values_ex - Variable(returns_ex)) ** 2.).sum()
        value_loss = values_ex_loss
        entropy_loss = (log_probs * probs).sum()
        
        ############################

        loss = td_loss + policy_loss + value_loss * args.value_coeff + entropy_loss * args.entropy_coeff
        #print(h)
        
        loss.backward()

        nn.utils.clip_grad_norm(net.parameters(), args.grad_norm_limit)
        optimizer.step()
        optimizer.zero_grad()

        # cut LSTM state autograd connection to previous rollout
        steps = []

    env.close()
    

def check_cycle(args, batch):
    last_batch = np.array(batch)[-args.rollout_steps:]
    cycle_penalty = np.zeros((args.rollout_steps, args.num_workers))
    loop_sizes = np.zeros((args.rollout_steps, args.num_workers))
    loop_sizes[:] = np.NaN
    
    for i in range(len(batch) - args.rollout_steps): # Loop over states of complete episode (0 - 400...)

        # Loop over last batch
        states = np.array(batch[i][0])
        next_states = last_batch[:, 1]
                 
        for j in range(len(next_states)): #(0 - 20)
            #print(states)
            #print(next_states[j])
            
            cycles = next_states[j] == states #(8,) 
            #print(cycles)
            
            loop_size = cycles * (((j + len(batch) - args.rollout_steps) - i) + 1)
            loop_size = loop_size.astype(np.float)
            loop_size[loop_size == 0] = np.nan
            #print(rewards.shape)
            
            loop_sizes[j] = loop_size
    
    cycle_penalty = -1.0 / loop_sizes 
    cycle_penalty = np.nan_to_num(cycle_penalty, nan=0, posinf=0)
    #print(loop_sizes)

    return cycle_penalty


def compute_td_loss(args, steps, test, cuda):
    masks_b, actions_b, policy_b = [], [], []
    for (_, _, masks, actions, policy, _, _, _) in steps:
        masks_b.append(masks)
        actions_b.append(actions)
        policy_b.append(policy)
            
    masks_b = torch.stack(masks_b)
    actions_b = torch.stack(actions_b)
    policy_b = torch.stack(policy_b)
    
    q_value = policy_b.gather(1, actions_b).squeeze(2)
    
    print(q_value.shape)
    print(test)
    
    
    loss = (q_value.squeeze(1) - reward_in).pow(2).mean()
    
 
    print(loss)
    
    return loss
    
def process_rollout(args, steps, cuda):
    # bootstrap discounted returns with final value estimates
    _, _, _, _, _, _, last_values_ex = steps[-1]
    returns_ex = last_values_ex.data
    
    advantages_ex = torch.zeros(args.num_workers, 1)

    if cuda: advantages_ex = advantages_ex.cuda()

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, rewards_in, masks, actions, policy, policies, values_ex = steps[t]
        _, _, _, _, _, _, next_values_ex = steps[t + 1]

        returns_ex = rewards + returns_ex * args.gamma * masks


        deltas_ex = rewards + next_values_ex.data * args.gamma * masks - values_ex.data
        
        advantages_ex = advantages_ex * args.gamma * args.lambd * masks + deltas_ex


        out[t] = actions, policy, policies, values_ex, returns_ex, advantages_ex

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))
