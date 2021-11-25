import gym
from time import sleep
env = gym.make("Acrobot-v1")

observation = env.reset()

done = False
total_reward = 0
frames = []
while not done:
    frame = env.render(mode = 'rgb_array')
    frames.append(frame)
    sleep(0.01)
    action = 0#env.action_space.sample()
    print(action)
    # action = 
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    break
    total_reward += reward
print(f"Total reward: {total_reward}, Done flag: {done}")
