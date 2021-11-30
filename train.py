# About: Train the agent using specific policy
import argparse
import random
import numpy as np
import sys
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Train without any sim2real extra methods")
parser.add_argument('--algo', type=str, help="Enter the algorithm name")
parser.add_argument('--env', default="mountaincar", type=str, help="Enter the environment name")
parser.add_argument('--seed', default=None, type=int, help="Enter the random seed")
parser.add_argument('--headless', default=True, type=int, help="Enter the random seed")
args = parser.parse_args()


random_seed = datetime.now().microsecond//1000 if args.seed is None else args.seed

np.random.seed(random_seed)
random.seed(random_seed)


import gym
from time import sleep
from algorithms.DQN import VanillaDQN
from algorithms.PPO import PPO
from algorithms.SAC import SAC
import matplotlib.pyplot as plt
import wandb
from configs import Config as cfg


algorithm = args.algo.lower()
envs_dict = {"mountaincar": "MountainCar-v0", "acrobot":"Acrobot-v1", "cartpole":"CartPole-v1"} 
wandb_flag = True
wandb_experiment_config = {"algorithm": algorithm,
                           "wandb": wandb_flag,
                           "seed": random_seed
                          }
env_name = envs_dict[args.env]

num_epochs = 500
configs = cfg(num_epochs).config

notes = f"Running RL algorithm ({wandb_experiment_config['algorithm']}) for env: {env_name}"

if(wandb_flag):
        wandb.init(
                project="PMLDL-Project",
                group=env_name+f"_{wandb_experiment_config['algorithm']}",
                config=wandb_experiment_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                notes=notes,
        )


env = gym.make(env_name)

agent = None
rewards = None
config = configs[args.env][args.algo]

num_epochs = config["num_epochs"]
num_steps = config["num_steps"]
config_reward_shaping_func = config["reward_shaping_func"]
config_special_termination_condition = config["special_termination_condition"]

if(algorithm == "dqn"):
    target_update_freq = config["target_update_freq"]
    eps_prob = config["eps_prob"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    agent = VanillaDQN(env, hidden_dim=64)
    # With num_steps=200 it improved the training and increased the reward
    rewards = agent.train(  return_rewards=True, 
                                    save_flag=True, 
                                    render=False, 
                                    save_file_path=f"./zoo/dqn/{args.env}", 
                                    save_file_name="best_model_dqn",
                                    num_epochs=num_epochs, 
                                    batch_size=batch_size, 
                                    target_update_freq=target_update_freq,  
                                    eps_prob=eps_prob, 
                                    learning_rate=learning_rate, 
                                    num_steps=num_steps,
                                    reward_shaping_func=config_reward_shaping_func,
                                    special_termination_condition=config_special_termination_condition,
                                    wandb_flag=wandb_flag,
                                    )
elif(algorithm == "ppo"):
    update_num_epochs = config["update_num_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    vf_coef = config["vf_coef"]
    clip_range = config["clip_range"]
    gae_lambda = config["gae_lambda"]
    gamma = config["gamma"]

    agent = PPO(env, lr=learning_rate, n_steps=num_steps, batch_size=batch_size, n_epochs=num_epochs,
                gae_lambda=gae_lambda, clip_range=clip_range, gamma=gamma, vf_coef=vf_coef)
    rewards = agent.train(  return_rewards=True, 
                            save_flag=True, 
                            render=False, 
                            save_file_path=f"./zoo/ppo/{args.env}", 
                            save_file_name="best_model_ppo",
                            reward_shaping_func=config_reward_shaping_func,#reward_shaping_func,
                            special_termination_condition=config_special_termination_condition,#special_termination_condition,
                            wandb_flag=wandb_flag,
                            num_steps = num_steps,
                            update_num_epochs=update_num_epochs
                            )

elif(algorithm == "sac"):
    alpha = config["alpha"]
    beta = config["beta"]
    tau = config["tau"]
    agent = SAC(env=env, alpha=alpha, beta=beta, tau=tau, input_dims=env.observation_space.shape, n_actions=env.action_space.n)
    rewards = agent.train(  return_rewards=True, 
                            save_flag=True, 
                            save_file_path=f"./zoo/sac/{args.env}", 
                            save_file_name="best_model_sac",
                            reward_shaping_func=config_reward_shaping_func,#reward_shaping_func,
                            special_termination_condition=config_special_termination_condition,#special_termination_condition,
                            wandb_flag=wandb_flag,
                            num_steps = num_steps,
                            n_epochs=num_epochs
                            )


# dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=5000, render=True, eps_prob=0.1, learning_rate=0.003, num_steps=1000)

# plt.plot([i+1 for i in range(len(rewards))], rewards)
# plt.xlabel("Epochs")
# plt.ylabel("Reward")
# plt.savefig(f"./zoo/ppo/{args.env}/best_model_ppo.png")
# plt.show()


observation = env.reset()

done = False
total_reward = 0
frames = []
while not done:
    if(args.headless == False):
        frame = env.render(mode = 'rgb_array')
        frames.append(frame)
        sleep(0.01)
    action = agent.sample_action(observation)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
print(f"Total reward: {total_reward}, Done flag: {done}")
