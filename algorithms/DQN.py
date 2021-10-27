# References:
# https://jonathan-hui.medium.com/rl-dqn-deep-q-network-e207751f7ae4
# https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/#:~:text=Deep%20Q%2DNetworks,is%20generated%20as%20the%20output.
# http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf
# https://github.com/openai/gym/wiki/MountainCar-v0

# TODO:
# - Use Wandb to report the results 

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from copy import deepcopy
import random
from math import tanh
# from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)
random.seed(0)

class Net(nn.Module):
    def __init__(self, observations_dim, actions_dim, hidden_dim=500):
        super(Net, self).__init__()
        self._input_layer = nn.Linear(observations_dim, hidden_dim)
        self._hidden1 = nn.Linear(hidden_dim, hidden_dim)
        # self._hidden2 = nn.Linear(64, 32)
        self._output_layer = nn.Linear(hidden_dim,actions_dim)

    def forward(self, x):
        x = F.relu(self._input_layer(x))
        x = F.relu(self._hidden1(x))
        # x = F.relu(self._hidden2(x))
        x = self._output_layer(x)
        return x

class ReplayMemory:
    def __init__(self, observation_size, action_size, replay_size=1000):
        self.replay = deque(maxlen=replay_size)
        self.observation_size = observation_size
        self.action_size = action_size

    def sample(self, num_samples=100):
        sample = np.array(random.sample(list(self.replay), num_samples))#np.random.choice(self.replay, num_samples)#min(len(self.replay),num_samples))
        st  = sample[:,:self.observation_size] 
        at  = sample[:,self.observation_size:self.observation_size+self.action_size]
        rt  = sample[:,self.observation_size+self.action_size:self.observation_size+self.action_size+1]
        st1 = sample[:,self.observation_size+self.action_size+1:]
        return (st, at, rt, st1)#, len(sample)

    # Sample is (s_i, a_i, r_i, s_{i+1})
    def add(self, s, a, r, s1):
        self.replay.append([*s, *a, r, *s1])

    def len(self):
        return len(self.replay)

class VanillaDQN:
    def __init__(self, env, hidden_dim=500, replay_size=1000, restore=None):
        self.env = env
        self.observations_dim = self.env.observation_space.shape[0] # For later it can be changed to be ok with other shapes
        self.actions_dim = self.env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(self.observations_dim, self.actions_dim, hidden_dim).to(self.device)
        self.target_net = deepcopy(self.net)
        self.replay = ReplayMemory(self.observations_dim, 1, replay_size)
        self.loss = nn.MSELoss()
        self.best_reward = -1e8

        if(restore is not None):
            self.net = torch.load(restore)

    def sample_action(self, observation):
        if(isinstance(observation, np.ndarray)):
            observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        q_values = self.net.forward(observation)
        return torch.argmax(q_values).int().sum().item()

    def _compute_loss(self, batch, gamma):
        st, at, rt, st1 = batch
        st1_torch = torch.from_numpy(st1).float().unsqueeze(0).to(self.device)
        rt_torch = torch.from_numpy(rt).float().unsqueeze(0).view(len(rt),1).to(self.device)#
        st_torch = torch.from_numpy(st).float().unsqueeze(0).to(self.device)
        at_torch = torch.from_numpy(at).long().unsqueeze(0).to(self.device)
        target = rt_torch + gamma*(torch.max(self.target_net.forward(st1_torch),2)[0]).view(len(rt),1)
        predicted = self.net.forward(st_torch).gather(2, at_torch)[0]
        return self.loss(target, predicted)

    def train(self, render=False,
                    num_epochs=100, num_steps=1000,
                    eps_prob=0.5, target_update_freq=500, batch_size=100, gamma=0.99, learning_rate=1e-3,
                    save_flag=False, save_file_name=None, save_file_path=None, return_rewards=False,
                    reward_shaping_func=None,
                    special_termination_condition=None,
                    ):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        total_num_steps = 0
        reward_list = []
        for i in range(num_epochs):
            # self.target_net = deepcopy(self.net)
            obs = self.env.reset()
            epoch_reward = 0
            for t in range(num_steps):
                if(render):
                    self.env.render()
                total_num_steps += 1
                obs_torch = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) # test without -> This should be the correct to change from numpy to torch and add it on cuda
                # States(Observation) -> DQN -> Q-values for all the actions
                q_vals = self.net.forward(obs_torch)
                # Select an action based on epsilon-greedy policy with the current Q-Network
                action = self.sample_action(obs)
                if(np.random.rand() < eps_prob):
                    action = self.env.action_space.sample()
                # Perform the action
                observation, r, done, _ = self.env.step(int(action))
                reward = 0
                if(reward_shaping_func is None):
                    reward = r
                else:
                    reward = reward_shaping_func(r, observation)

                # print(reward, abs(observation[0]-0.5), abs(observation[1])*2)
                epoch_reward += r
                # Store the transition (s_t, a_t, r_t, s_{t+1})
                self.replay.add(list(deepcopy(obs)), list([action]), reward, list(observation))
                obs = deepcopy(observation)
                if(self.replay.len() >= batch_size):
                    # Sample random batch
                    batch = self.replay.sample(batch_size)
                    # Compute loss
                    loss = self._compute_loss(batch, gamma)
                    # Optimize the network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if(total_num_steps % target_update_freq == 0):
                    self.target_net = deepcopy(self.net)

                if(special_termination_condition is not None and special_termination_condition(observation)):
                    print(f"Done special termination condition")
                    done = True

                                    
                if(done or t == num_steps-1):
                    print(f"Epoch {i+1} (reward): {epoch_reward}")
                    reward_list.append(epoch_reward)
                    if(self.best_reward < epoch_reward and save_flag):
                        self.save(save_file_path, save_file_name)
                        self.best_reward = epoch_reward
                    break

        if(return_rewards):
            return reward_list

    def save(self, path, name):
        print(f"Saved the model to {path+name}")
        torch.save(self.net, path+name)

if __name__ == '__main__':
    pass
    
