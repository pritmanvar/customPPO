import gymnasium as gym
from ppo import PPO
import numpy as np

env = gym.make('Ant-v4', render_mode='rgb_array')
print(env.action_space.sample())
model = PPO(env)
model.learn(100)
breakpoint()


import matplotlib.pyplot as plt


plt.plot(model.rewards, label='Normalized Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.savefig('rewards_per_episode.png')