# About: Train the agent using specific policy

import gym
from algorithms.DQN import VanillaDQN
from algorithms.PPO import PPO
import matplotlib.pyplot as plt
import wandb


wandb_experiment_config = {"algorithm": 
                                "DQN"
                          }
env_name = "CartPole-v1"
notes = f"Running RL algorithm ({wandb_experiment_config['algorithm']}) for env: {env_name}"

wandb.init(
        project="PMLDL-Project",
        group=env_name+f"_{wandb_experiment_config['algorithm']}",
        config=wandb_experiment_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        notes=notes,
)


# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make(env_name)
dqn_agent = VanillaDQN(env, hidden_dim=64)
#ppo_agent = PPO(env)

# After testing with the original reward of the environment, nothing was improved in training
# So, I have changed the reward function, to test different behavior and found some improvements
# Multiple reward functions have been tested to conform with specific needed behaviors:
# - Move with fast right and left -> Correlated with velocity [2nd observation]
# - Move closer to the goal -> Correlated with the position [1st observation]
# Some observations:
# - When only the position is in the reward (or the position dominated) it makes it only try to go up not by going right and left but just go right
# - When only the velocity is in the reward (or the velocity dominated) it makes it only to move fast right and left and don't care about the real goal (position)
reward_shaping_func = lambda r, obs: r + abs(obs[1])*10-abs(obs[0]-0.5) 
special_termination_condition = lambda obs: obs[0] > 0.48

num_epochs = 1000
batch_size = 256
target_update_freq = 1
eps_prob = 0.91
learning_rate = 0.002
num_steps = 500
# With num_steps=200 it improved the training and increased the reward
rewards = dqn_agent.train(  return_rewards=True, 
                            save_flag=True, 
                            render=False, 
                            save_file_path="./zoo/dqn/", 
                            save_file_name="best_model_dqn1",
                            num_epochs=num_epochs, 
                            batch_size=batch_size, 
                            target_update_freq=target_update_freq,  
                            eps_prob=eps_prob, 
                            learning_rate=learning_rate, 
                            num_steps=num_steps,
                            reward_shaping_func= None,
                            special_termination_condition=None,
                            wandb_flag=True,
                          )
# rewards = ppo_agent.train(  return_rewards=True, 
#                             save_flag=True, 
#                             render=False, 
#                             save_file_path="./zoo/ppo/", 
#                             save_file_name="best_model_ppo",
#                             reward_shaping_func=reward_shaping_func,
#                             special_termination_condition=special_termination_condition,
#                             wandb_flag=True,
#                             num_steps = num_steps,
#                           )

# dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=5000, render=True, eps_prob=0.1, learning_rate=0.003, num_steps=1000)

plt.plot([i+1 for i in range(num_epochs)], rewards)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.savefig("./zoo/dqn/best_model_dqn1.png")
plt.show()
