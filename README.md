# PMLDL-Project
This repository is for Practical Machine Learning and Deep Learning course project at Innopolis University Fall 2021

## Project directory structure

├── media: photos and gifs

├── algorithms: different algorithms

└── zoo: trained agents

### How to Run it?

In order to train
```bash
python3 train.py --algo <algo> --env <env>
```
* replace \<algo> with (ppo, sac, or dqn)
  
* replace \<env> with (cartpole, mountaincar, acrobot)
  
  
In order to evaluate and render the episode of the agent with the trained model. The default directory is good_zoo directory, to change it you need to change the code. Later, we will make it through cli
```bash
python3 run_agent.py --algo <algo> --env <env>
```

* replace \<algo> with (ppo, sac, or dqn)
  
* replace \<env> with (cartpole, mountaincar, acrobot)

#### A demo for DQN:

Trained Agent for cartpole:

![Trained Agent](https://github.com/hany606/PMLDL-Project/blob/main/media/gif/dqn.gif)

Rewards:

Training reward:

![Best model training](https://github.com/hany606/PMLDL-Project/blob/main/zoo/dqn/best_model_dqn.png)

Testing reward:

![Best model training](https://github.com/hany606/PMLDL-Project/blob/main/zoo/dqn/best_model_dqn_testing.png)


## TODO:
- [X] Implement DQN
- [X] Change the implementation of DQN to be more generalized for any environment
- [X] Add support for wandb or Tensorboard
- [X] Create cli interface for run_agent.py
- [X] Implement PPO
- [X] Implement SAC
- [X] Create Benchmark using different envs and between different agorithms
