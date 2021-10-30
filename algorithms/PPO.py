# Implementation of PPO algorithm based on Actor-Critic architechture
# References:
# - https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
# - https://stable-baselines3.readthedocs.io/

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from copy import deepcopy
import random
from math import tanh
import wandb
import os
from torch.distributions.categorical import Categorical



# Save states, rewards, done, actions, probs, values 
class Memory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.probs = []
        self.values = []

    def add(self, state, reward, done, action, prob, value):
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)

    def len(self):
        return len(self.states)

    def sample(self, batch_size=64):
        data_length = self.len()
        batch_start = np.arange(0, data_length, batch_size)
        indicies = np.arange(data_length, dtype=np.int64)
        batches = [indicies[i:i+batch_size] for i in batch_start]        
        state_batches = []
        reward_batches = []
        done_batches = []
        action_batches = []
        prob_batches = []
        val_batches = []
        for batch in batches:
            state_batch = []
            reward_batch = []
            done_batch = []
            action_batch = []
            prob_batch = []
            val_batch = []
            for batch_idx in batch:
                state_batch.append(self.states[batch_idx])
                reward_batch.append(self.rewards[batch_idx])
                done_batch.append(self.dones[batch_idx])
                action_batch.append(self.actions[batch_idx])
                prob_batch.append(self.probs[batch_idx])
                val_batch.append(self.values[batch_idx])
            state_batches.append(np.array(state_batch))
            reward_batches.append(np.array(reward_batch))
            done_batches.append(np.array(done_batch))
            action_batches.append(np.array(action_batch))
            prob_batches.append(np.array(prob_batch))
            val_batches.append(np.array(val_batch))

        return np.array(state_batches), np.array(reward_batches), np.array(done_batches), np.array(action_batches), np.array(prob_batches), np.array(val_batches), len(batches)

    def get(self):
        pass


class Network(nn.Module):
    def __init__(self, 
                 observations_dim, actions_dim,
                 inp_hidden_dim=500, out_hidden_dim=500,
                 lr=1e-3, device=None):
        super(Network, self).__init__()
        self.input_layer = nn.Linear(observations_dim, inp_hidden_dim)
        self.output_layer = nn.Linear(out_hidden_dim,actions_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _to(self):
        print(f"## Device: {self.device}")
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError("Implementation of forward for the network")

class ActorNetwork(Network):
    def __init__(self, *args, **kwargs):
        kwargs["inp_hidden_dim"] = 512
        kwargs["out_hidden_dim"] = 256
        super(ActorNetwork, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self._to()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        dist = F.softmax(x, dim=-1)

        dist = Categorical(dist)
        
        return dist


class CriticNetwork(Network):
    def __init__(self, *args, **kwargs):
        kwargs["inp_hidden_dim"] = 512
        kwargs["out_hidden_dim"] = 256
        super(CriticNetwork, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self._to()


    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

class PPO:
    def __init__(self, env, device=None,
                 lr=1e-3, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
                 restore=None):

        self.env = env
        self.observations_dim = self.env.observation_space.shape[0] # For later it can be changed to be ok with other shapes
        self.actions_dim = self.env.action_space.n
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.rollout_buffer = Memory()
        self.best_reward = -1e8

        self.lr = lr
        self.n_steps = n_steps  # Adapted from PPO stable baselines
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef

        self.actor_net = ActorNetwork(observations_dim=self.observations_dim, actions_dim=self.actions_dim, lr=lr)
        self.critic_net = CriticNetwork(observations_dim=self.observations_dim, actions_dim=1, lr=lr)

        if(restore is not None):
            self.actor_net = torch.load(restore["actor"])
            self.critic_net = torch.load(restore["critic"])

    def _compute_loss(self, batch):
        pass

    def _comput_advatage(self, rewards, dones, values):
        # print(rewards, dones, values)
        rewards_len = len(rewards)
        advantage = np.zeros(rewards_len, dtype=np.float32)
        for i in range(rewards_len-1):
            discount = 1
            a_t = 0
            for j in range(i, rewards_len-1):
                a_t += discount*(rewards[j] + self.gamma*values[j+1]*(1-int(dones[j])) - values[j])
                discount *= self.gamma*self.gae_lambda
            advantage[i] = a_t
        return advantage

    def get_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor_net.device)
        distribution = self.actor_net(state)
        value = self.critic_net(state)
        action = distribution.sample()

        probs = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def _collect_rollout(self, render=False, reward_shaping_func=None, special_termination_condition=None, num_steps=None):
        observation = self.env.reset()
        self.rollout_buffer.reset()
        total_num_steps = 0
        done = False
        epoch_reward = 0
        while not done:
            if(render):
                self.env.render()
            total_num_steps += 1
            action, prob, value = self.get_action(observation)
            observation, r, done, _ = self.env.step(action)


            reward = 0
            if(reward_shaping_func is None):
                reward = r
            else:
                reward = reward_shaping_func(r, observation)
            self.rollout_buffer.add(observation, reward, done, action, prob, value)

            epoch_reward += reward
            if(special_termination_condition is not None and special_termination_condition(observation)):
                    print(f"Done special termination condition")
                    done = True
            if(done or (num_steps is not None and total_num_steps == num_steps-1)):
                break
        return total_num_steps, epoch_reward

    def train(self, render=False,
                    save_flag=False, save_file_name=None, save_file_path=None, return_rewards=False,
                    reward_shaping_func=None,
                    num_steps = None,
                    special_termination_condition=None,
                    wandb_flag=False
                    ):

        total_num_steps = 0
        reward_list = []
        while self.n_steps > total_num_steps:
            epoch_steps, epoch_reward = self._collect_rollout(render=render, reward_shaping_func=reward_shaping_func, special_termination_condition=special_termination_condition, num_steps=num_steps)
            print(f"Epoch {total_num_steps} (reward): {epoch_reward}")
            total_num_steps += 1
            if(wandb_flag):
                wandb.log({"epoch_reward": epoch_reward})
            reward_list.append(epoch_reward)
            if(self.best_reward < epoch_reward and save_flag):
                self.save(save_file_path, save_file_name)
                self.best_reward = epoch_reward

            for epoch in range(self.n_epochs):
                states, rewards, dones, actions, old_probs, values, num_batches = self.rollout_buffer.sample()
                for batch in range(1):
                    # print(len(rewards[batch]), len(dones[batch]), len(values[batch]))
                    advantage = self._comput_advatage(rewards[batch], dones[batch], values[batch])
                    advantage = torch.tensor(advantage).to(self.actor_net.device)
                    values = torch.tensor(values[batch]).to(self.actor_net.device)
                    # for b in range(len(rewards[batch])):
                    state = torch.tensor(states[batch], dtype=torch.float).to(self.actor_net.device)
                    old_prob = torch.tensor(old_probs[batch]).to(self.actor_net.device)
                    action = torch.tensor(actions[batch]).to(self.actor_net.device)

                    distribution = self.actor_net(state)
                    critic_value = torch.squeeze(self.critic_net(state))
                    new_probs = distribution.log_prob(action)

                    prob_ratio = new_probs.exp() / old_prob.exp()

                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.clip_range, 1+self.clip_range)*advantage[batch]
                    # clipped surrogate loss
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns-critic_value)**2
                    critic_loss = critic_loss.mean()

                    total_loss = actor_loss + self.vf_coef*critic_loss
                    self.actor_net.optimizer.zero_grad()
                    self.critic_net.optimizer.zero_grad()
                    total_loss.backward()
                    self.actor_net.optimizer.step()
                    self.critic_net.optimizer.step()
        if(return_rewards):
            return reward_list

    def save(self, path, name):
        actor_log_dir = os.path.join(path, "actor_ppo")
        critic_log_dir = os.path.join(path, "critic_ppo")
        if not os.path.exists(actor_log_dir):
            os.makedirs(actor_log_dir + '/')
        if not os.path.exists(critic_log_dir):
            os.makedirs(critic_log_dir + '/')

        print(f"Saved actor model to {actor_log_dir}")
        torch.save(self.actor_net, actor_log_dir+name)

        print(f"Saved critic model to {critic_log_dir}")
        torch.save(self.critic_net, critic_log_dir+name)
