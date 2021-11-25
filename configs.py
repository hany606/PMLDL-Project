class Config:
    def __init__(self, num_epochs):
        # -------------------- Cartpole =================
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
        # -------------------------------------------------------------------
        acrobot_reward_shaping_func = lambda r, obs: r + (abs(obs[4]) + abs(obs[5]))*0.01


        self.config = {
                        "cartpole": {
                                    "ppo": {
                                            "num_steps": 100,
                                            "update_num_epochs": 4,
                                            "learning_rate":0.0003,
                                            "batch_size": 5,
                                            "num_epochs": num_epochs,
                                            "special_termination_condition": None,
                                            "reward_shaping_func": None
                                            },
                                    "dqn":{
                                            "num_epochs": num_epochs,
                                            "batch_size": 1000,
                                            "target_update_freq": 5000,
                                            "eps_prob": 0.1000,
                                            "learning_rate": 0.03,
                                            "num_steps": 100,
                                            "special_termination_condition": None,
                                            "reward_shaping_func": None
                                            }
                                    },
                        "mountaincar": {
                                    "ppo": {
                                            "num_steps": 500,
                                            "update_num_epochs": 4,
                                            "learning_rate":0.003,
                                            "batch_size": 5,
                                            "num_epochs": num_epochs,
                                            "special_termination_condition": special_termination_condition,
                                            "reward_shaping_func": reward_shaping_func
                                            },
                                    "dqn":{
                                            "num_epochs": num_epochs,
                                            "batch_size": 1000,
                                            "target_update_freq": 5000,
                                            "eps_prob": 0.1000,
                                            "learning_rate": 0.0003,
                                            "num_steps": 500,
                                            "special_termination_condition": special_termination_condition,
                                            "reward_shaping_func": reward_shaping_func
                                            }
                                    },

                        "acrobot": {
                                    "ppo": {
                                            "num_steps": 100,
                                            "update_num_epochs": 5,
                                            "learning_rate":0.0003,
                                            "batch_size": 10,
                                            "num_epochs": num_epochs,
                                            "special_termination_condition": None,
                                            "reward_shaping_func": acrobot_reward_shaping_func
                                            },
                                    "dqn":{
                                            "num_epochs": num_epochs,
                                            "batch_size": 1000,
                                            "target_update_freq": 5000,
                                            "eps_prob": 0.1000,
                                            "learning_rate": 0.0003,
                                            "num_steps": 100,
                                            "special_termination_condition": None,
                                            "reward_shaping_func": acrobot_reward_shaping_func
                                            }
                                    },
                        
                        }