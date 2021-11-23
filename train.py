# About: Train the agent using specific policy

import gym
from algorithms.DQN import VanillaDQN
from algorithms.PPO import PPO
import matplotlib.pyplot as plt
import wandb
import argparse

parser = argparse.ArgumentParser(description="Train without any sim2real extra methods")
parser.add_argument('--algo', type=str, help="Enter the algorithm name")
parser.add_argument('--env', default="mountaincar", type=str, help="Enter the environment name")
args = parser.parse_args()


algorithm = args.algo.lower()
envs_dict = {"mountaincar": "MountainCar-v0", "acrobot":"Acrobot-v1", "cartpole":"CartPole-v1"} 
wandb_flag = False#True
wandb_experiment_config = {"algorithm": algorithm,
                           "wandb": wandb_flag, 
                          }
env_name = envs_dict[args.env]
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


# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make(env_name)

agent = None

if(algorithm == "dqn"):
        num_epochs = 500
        batch_size = 1000
        target_update_freq = 5000
        eps_prob = 0.1000
        learning_rate = 0.003
        num_steps = 200
        agent = VanillaDQN(env, hidden_dim=64)
        # With num_steps=200 it improved the training and increased the reward
        rewards = agent.train(  return_rewards=True, 
                                        save_flag=True, 
                                        render=False, 
                                        save_file_path="./zoo/dqn/", 
                                        save_file_name="best_model_dqn",
                                        num_epochs=num_epochs, 
                                        batch_size=batch_size, 
                                        target_update_freq=target_update_freq,  
                                        eps_prob=eps_prob, 
                                        learning_rate=learning_rate, 
                                        num_steps=num_steps,
                                        reward_shaping_func=reward_shaping_func,
                                        special_termination_condition=special_termination_condition,
                                        wandb_flag=wandb_flag,
                                        )
elif(algorithm == "ppo"):
        num_epochs = 500
        batch_size = 5
        learning_rate = 0.0003
        num_steps = int(500)
        update_num_epochs = 4

        agent = PPO(env, lr=learning_rate, n_steps=num_steps, batch_size=batch_size, n_epochs=num_epochs)
        rewards = agent.train2(  return_rewards=True, 
                                save_flag=True, 
                                render=False, 
                                save_file_path=f"./zoo/ppo/{args.env}", 
                                save_file_name="best_model_ppo",
                                reward_shaping_func=None,#reward_shaping_func,
                                special_termination_condition=None,#special_termination_condition,
                                wandb_flag=wandb_flag,
                                num_steps = num_steps,
                                update_num_epochs=update_num_epochs
                                )

# dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=5000, render=True, eps_prob=0.1, learning_rate=0.003, num_steps=1000)

plt.plot([i+1 for i in range(num_epochs)], rewards)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.savefig(f"./zoo/ppo/{args.env}/best_model_ppo.png")
plt.show()


observation = env.reset()

done = False
total_reward = 0
frames = []
while not done:
    frames.append(env.render(mode = 'rgb_array'))
    action = agent.sample_action(observation)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
print(f"Total reward: {total_reward}, Done flag: {done}")
