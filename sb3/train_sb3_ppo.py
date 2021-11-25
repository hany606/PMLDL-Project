import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

parser = argparse.ArgumentParser(description="Train without any sim2real extra methods")
parser.add_argument('--algo', type=str, help="Enter the algorithm name")
parser.add_argument('--env', default="mountaincar", type=str, help="Enter the environment name")
args = parser.parse_args()


algorithm = args.algo.lower()
envs_dict = {"mountaincar": "MountainCar-v0", "acrobot":"Acrobot-v1", "cartpole":"Cartpole-v1"} 

wandb_flag = True
wandb_experiment_config = {"algorithm": "ppo",
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


# Create environment
env = gym.make('MountainCar-v0')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5), callback=WandbCallback())
# Save the agent
model.save("zoo/ppo/sb3_mountainCar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("zoo/ppo/sb3_mountainCar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()