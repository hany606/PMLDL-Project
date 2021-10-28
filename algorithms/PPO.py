# Implementation of PPO algorithm based on Actor-Critic architechture
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

    def sample(self, num_sampels=64):
        sample = np.array(random.sample(list(self.replay), num_sampels))#np.random.choice(self.replay, num_samples)#min(len(self.replay),num_samples))


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
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError("Implementation of forward for the network")

class ActorNetwork(Network):
    def __init__(self, *args, **kwargs):
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(256, 256)
        del kwargs["inp_hidden_dim"]
        del kwargs["out_hidden_dim"]
        super(ActorNetwork, self).__init__(*args, **kwargs, inp_hidden_dim=512, out_hidden_dim=256)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

class CriticNetwork(Network):
    def __init__(self, *args, **kwargs):
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(256, 256)
        del kwargs["inp_hidden_dim"]
        del kwargs["out_hidden_dim"]
        super(CriticNetwork, self).__init__(*args, **kwargs, inp_hidden_dim=512, out_hidden_dim=256)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x


class PPO:
    def __init__(self,
                 lr=1e-3, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 restore=None):
        self.lr = lr
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range

        self.actor_net = ActorNetwork(lr=lr)
        self.critic_net = CriticNetwork(lr=lr)

        if(restore is not None):
            self.actor_net = torch.load(restore["actor"])
            self.critic_net = torch.load(restore["critic"])

    def _compute_loss(self, batch):
        pass

    def _comput_advatage(self):
        pass

    def train(self, render=False,
                    num_epochs=100, num_steps=1000,
                    eps_prob=0.5, target_update_freq=500, batch_size=100, gamma=0.99, learning_rate=1e-3,
                    save_flag=False, save_file_name=None, save_file_path=None, return_rewards=False,
                    reward_shaping_func=None,
                    special_termination_condition=None,
                    wandb_flag=False
                    ):

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
                action = ?
                observation, r, done, _ = self.env.step(int(action))

                if(special_termination_condition is not None and special_termination_condition(observation)):
                    print(f"Done special termination condition")
                    done = True

                                    
                if(done or t == num_steps-1):
                    print(f"Epoch {i+1} (reward): {epoch_reward}")
                    if(wandb_flag):
                        wandb.log({"epoch_reward": epoch_reward})
                    reward_list.append(epoch_reward)
                    if(self.best_reward < epoch_reward and save_flag):
                        self.save(save_file_path, save_file_name)
                        self.best_reward = epoch_reward
                    break

    def save(self, path, name):
        actor_log_dir = os.path.join(path, name, "actor_ppo")
        if not os.path.exists(actor_log_dir):
            os.makedirs(actor_log_dir + '/')
        critic_log_dir = os.path.join(path, name, "critic_ppo")
        if not os.path.exists(critic_log_dir):
            os.makedirs(critic_log_dir + '/')

        print(f"Saved actor model to {actor_log_dir}")
        torch.save(self.actor_net, actor_log_dir)

        print(f"Saved critic model to {critic_log_dir}")
        torch.save(self.critic_net, critic_log_dir)
