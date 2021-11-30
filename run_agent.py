# About: Running an agent with a specific policy

# TODO:
# - Create cli argument to know which policy to run and which agent to choose from the trained agents
 

import gym
import numpy as np
from algorithms.DQN import VanillaDQN
from algorithms.PPO import PPO
from algorithms.SAC import SAC
from torch import load
from time import time
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
from time import sleep

parser = argparse.ArgumentParser(description="Train without any sim2real extra methods")
parser.add_argument('--algo', type=str, help="Enter the algorithm name")
parser.add_argument('--env', default="mountaincar", type=str, help="Enter the environment name")
args = parser.parse_args()
algorithm = args.algo.lower()
envs_dict = {"mountaincar": "MountainCar-v0", "acrobot":"Acrobot-v1", "cartpole":"CartPole-v1"} 
env_name = envs_dict[args.env]


# Source: http://www.pinchofintelligence.com/getting-started-openai-gym/
def display_frames_as_gif(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    # display(display_animation(anim, default_mode='loop'))



def get_action(state, policy=None):
    if policy is None:
        return env.action_space.sample()
    else:
        return policy.sample_action(state)

env = gym.make(env_name)
agent = None
if(algorithm == "dqn"):
    agent = VanillaDQN(env, restore=f"good_zoo/dqn/{args.env}/best_model_dqn")
elif(algorithm == "ppo"):
    agent = PPO(env, restore={"actor":f"good_zoo/ppo/{args.env}/actor_ppobest_model_ppo", "critic":f"good_zoo/ppo/{args.env}/critic_ppobest_model_ppo"})
elif(algorithm == "sac"):
    agent = SAC(env=env, restore={"actor":f"good_zoo/sac/{args.env}/actor_sacbest_model_sac", "critic_1":f"good_zoo/sac/{args.env}/critic1_sacbest_model_sac", "critic_2":f"good_zoo/sac/{args.env}/critic2_sacbest_model_sac",
                              "target_value":f"good_zoo/sac/{args.env}/target_value_sacbest_model_sac",
                              "value":f"good_zoo/sac/{args.env}/value_sacbest_model_sac"})

observation = env.reset()
done = False
total_reward = 0
frames = []
while not done:
    frames.append(env.render(mode = 'rgb_array'))
    action = get_action(observation, agent)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
    sleep(0.01)
display_frames_as_gif(frames, f"media/gif/{algorithm}.gif")
print(f"Total reward: {total_reward}, Done flag: {done}")


rewards_list = []
for i in range(100):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = get_action(observation, agent)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    rewards_list.append(total_reward)
    print(f"Total reward: {total_reward}, Done flag: {done}")

print(f"Mean reward for 100 episode: {np.mean(rewards_list)}")
plt.clf()
plt.plot([i+1 for i in range(100)], rewards_list)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.savefig("./good_zoo/dqn/best_model_dqn_testing.png")
plt.show()

# env.close()
    