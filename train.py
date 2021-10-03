# About: Train the agent using specific policy

import gym
from models.DQN import VanillaDQN
import matplotlib.pyplot as plt


# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make('MountainCar-v0')
dqn_agent = VanillaDQN(env, hidden_dim=64)


num_epochs = 500
batch_size = 1000
target_update_freq = 5000
eps_prob = 0.1000
learning_rate = 0.003
num_steps = 200
# With num_steps=200 it improved the training and increased the reward
rewards = dqn_agent.train(  return_rewards=True, 
                            save_flag=True, 
                            render=False, 
                            save_file_path="./zoo/dqn/", 
                            save_file_name="best_model_dqn",
                            num_epochs=num_epochs, 
                            batch_size=batch_size, 
                            target_update_freq=target_update_freq,  
                            eps_prob=eps_prob, 
                            learning_rate=learning_rate, 
                            num_steps=num_steps
                          )
# dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=5000, render=True, eps_prob=0.1, learning_rate=0.003, num_steps=1000)

plt.plot([i+1 for i in range(num_epochs)], rewards)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.savefig("./zoo/dqn/best_model_dqn.png")
plt.show()
