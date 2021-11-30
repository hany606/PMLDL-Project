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
        # print(batch_size)
        # print(batch_start.shape)
        indices = np.arange(data_length, dtype=np.int64)
        np.random.shuffle(indices) 
        batches = [indices[i:i+batch_size] for i in batch_start]     
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
                state_batch.append(np.array(self.states[batch_idx]))
                reward_batch.append(np.array(self.rewards[batch_idx]))
                done_batch.append(np.array(self.dones[batch_idx]))
                action_batch.append(np.array(self.actions[batch_idx]))
                prob_batch.append(np.array(self.probs[batch_idx]))
                val_batch.append(np.array(self.values[batch_idx]))
            state_batches.append(np.array(state_batch))
            reward_batches.append(np.array(reward_batch))
            done_batches.append(np.array(done_batch))
            action_batches.append(np.array(action_batch))
            prob_batches.append(np.array(prob_batch))
            val_batches.append(np.array(val_batch))
        return np.array(state_batches), np.array(reward_batches), np.array(done_batches), np.array(action_batches), np.array(prob_batches), np.array(val_batches), len(batches)

    def get(self):
        pass

# ----------------------------- Networks -----------------------------
class Network(nn.Module):
    def __init__(self, 
                 observations_dim, actions_dim,
                 inp_hidden_dim=500, out_hidden_dim=500,
                 lr=1e-3, device=None):
        super(Network, self).__init__()
        self.input_layer = nn.Linear(observations_dim, inp_hidden_dim)
        self.output_layer = nn.Linear(out_hidden_dim, actions_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.lr = lr

    def _to(self):
        print(f"## Device: {self.device}")
        self.to(self.device)

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        print(self.parameters)


    def forward(self, x):
        raise NotImplementedError("Implementation of forward for the network")

class ActorNetwork(Network):
    def __init__(self, *args, **kwargs):
        kwargs["inp_hidden_dim"] = 512
        kwargs["out_hidden_dim"] = 128
        super(ActorNetwork, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self._init_optimizer()
        self._to()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        probs = F.softmax(x, dim=-1)
        dist = Categorical(probs)
        
        return dist


class CriticNetwork(Network):
    def __init__(self, *args, **kwargs):
        kwargs["inp_hidden_dim"] = 512
        kwargs["out_hidden_dim"] = 128
        super(CriticNetwork, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self._init_optimizer()
        self._to()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x
# --------------------------------------------------------------

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
        self.best_avg_reward = -1e8

        self.compare_eps = 1e-8

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
            self.restore(restore)

    def restore(self, restore):
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

    # Just for evaluation usage (feedforward only for the actor network)
    # TODO: remove it later
    def sample_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor_net.device)
        distribution = self.actor_net(state)
        action = torch.squeeze(distribution.sample()).item()
        return action

    def _collect_rollout(self, 
                         render=False,
                         reward_shaping_func=None, 
                         special_termination_condition=None, 
                         num_steps=200):
        # Throw away the old collected data (old batch)
        self.rollout_buffer.reset()
        total_num_steps = 0
        epochs_reward = []
        epochs_original_reward = []
        while total_num_steps < num_steps:
            observation = self.env.reset()
            done = False
            epoch_reward = 0
            original_reward = 0
            while not done:
                if(render):
                    self.env.render()
                total_num_steps += 1
                # with torch.no_grad():
                action, prob, value = self.get_action(observation)
                new_observation, r, done, _ = self.env.step(action)

                reward = 0
                original_reward += r
                if(reward_shaping_func is None):
                    reward = r
                else:
                    reward = reward_shaping_func(r, observation)

                if(special_termination_condition is not None and special_termination_condition(observation)):
                        # print(f"Done special termination condition")
                        done = True
                # print(observation)
                # print(reward)
                # print(done)
                # print(f"action: {action}")
                # print(prob)
                # print(value)
                self.rollout_buffer.add(observation, reward, done, action, prob, value)
                observation = new_observation

                epoch_reward += reward
                if(done or total_num_steps >= num_steps):
                    break
            epochs_reward.append(epoch_reward)
            epochs_original_reward.append(original_reward)
            # print(f"Num of episodes: {len(epochs_reward)}")
        return total_num_steps, np.mean(epochs_reward), np.mean(epochs_original_reward)

    def train1(self, render=False,
                    save_flag=False, save_file_name=None, save_file_path=None, return_rewards=False,
                    reward_shaping_func=None,
                    num_steps = None,
                    special_termination_condition=None,
                    wandb_flag=False,
                    update_num_epochs=1
                    ):
        # epochs_

        total_num_steps = 0
        original_reward_list = []
        # Training loop
        for epoch in range(self.n_epochs):
            # Collect the training batch 
            epoch_steps, epoch_reward, epoch_original_reward = self._collect_rollout(render=render,
                                                              reward_shaping_func=reward_shaping_func,
                                                              special_termination_condition=special_termination_condition, 
                                                              num_steps=num_steps)
            original_reward_list.append(epoch_original_reward)
            avg_reward = np.mean(original_reward_list[-50:])
            print(f"Epoch {epoch} (reward): {epoch_original_reward}, (mod_reward): {epoch_reward}, (mean of last 50 epoch) {avg_reward}")
            total_num_steps += epoch_steps
            if(wandb_flag):
                wandb.log({"epoch_reward": epoch_original_reward})
            if(self.best_reward - epoch_original_reward <= self.compare_eps and self.best_avg_reward - avg_reward <= self.compare_eps and save_flag):
                self.save(save_file_path, save_file_name)
                self.best_reward = epoch_original_reward
                self.best_avg_reward = avg_reward
            self.learn(update_num_epochs)
        if(return_rewards):
            return original_reward_list

    def train2(self,render=False,
                    save_flag=False, save_file_name=None, save_file_path=None, return_rewards=False,
                    reward_shaping_func=None,
                    num_steps = None,
                    special_termination_condition=None,
                    wandb_flag=False,
                    update_num_epochs=1
                    ):
        # epochs_

        total_num_steps = 0
        reward_list = []
        update_freq = 20
        # Training loop
        for epoch in range(self.n_epochs):
            observation = self.env.reset()
            done = False
            epoch_reward = 0
            while not done:
                with torch.no_grad():
                    action, prob, val = self.get_action(observation)
                new_observation, reward, done, info = self.env.step(action)
                total_num_steps += 1
                epoch_reward += reward
                self.rollout_buffer.add(observation, reward, done, action, prob, val)
                if total_num_steps % update_freq == 0:
                    self.learn(update_num_epochs)
                    self.rollout_buffer.reset()

                observation = new_observation
            reward_list.append(epoch_reward)
            avg_reward = np.mean(reward_list[-50:])
            avg_reward10 = np.mean(reward_list[-10:])
            print(f"Epoch {epoch} (reward): {epoch_reward}, (mean of the last 50 epoch) {avg_reward}")
            if(wandb_flag):
                wandb.log({"epoch_reward": epoch_reward})
            if(self.best_reward - epoch_reward <= self.compare_eps and self.best_avg_reward - avg_reward10 <= self.compare_eps and save_flag):
                self.save(save_file_path, save_file_name)
                self.best_reward = epoch_reward
                self.best_avg_reward = avg_reward10
        if(return_rewards):
            return reward_list

    def train(self, *args, **kwargs):
        train_version = kwargs.get("train_version", 1)
        if("train_version" in kwargs.keys()):
            del kwargs["train_version"]
        rewards = None
        if(train_version == 1):
            rewards = self.train1(*args, **kwargs)
        elif(train_version == 2):
            rewards = self.train2(*args, **kwargs) 
        return rewards

    def learn(self, update_num_epochs):
        for update_epoch in range(update_num_epochs):
            states, rewards, dones, actions, old_probs, values, num_batches = self.rollout_buffer.sample(batch_size=self.batch_size)
            advantage = self._comput_advatage(self.rollout_buffer.rewards, self.rollout_buffer.dones, self.rollout_buffer.values)
            advantage = torch.tensor(advantage).to(self.actor_net.device)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for batch in range(num_batches):
                # Change arrays to torch tensors
                values_tensor = torch.tensor(values[batch], dtype=torch.float).to(self.actor_net.device)
                state = torch.tensor(states[batch], dtype=torch.float).to(self.actor_net.device)
                old_prob = torch.tensor(old_probs[batch], dtype=torch.float).to(self.actor_net.device)
                action = torch.tensor(actions[batch], dtype=torch.float).to(self.actor_net.device)

                # Get the distribution for the control policy
                distribution = self.actor_net(state)
                new_probs = distribution.log_prob(action)

                # Infer the value from the critic network
                critic_value = torch.squeeze(self.critic_net(state))

                ratio = new_probs.exp() / old_prob.exp()
                # ratio = torch.exp(new_probs - old_prob)

                # clipped surrogate loss
                weighted_probs = advantage[batch] * ratio
                weighted_clipped_probs = advantage[batch]*torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values_tensor
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.vf_coef*critic_loss
                self.actor_net.optimizer.zero_grad()
                self.critic_net.optimizer.zero_grad()
                total_loss.backward()
                self.actor_net.optimizer.step()
                self.critic_net.optimizer.step()  

    def save(self, path, name):
        # TODO: bug as it saves empty folders, and does not save in them, easy to fix, but you need to change run_agent.py file as well
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
