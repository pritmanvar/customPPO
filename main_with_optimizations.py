import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from ppo_with_optimizations import PPO
import os

os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

environment_name = str(input("Please enter environment name: "))
env = gym.make(environment_name, render_mode='rgb_array')
print(environment_name)
model = PPO(env)
model.learn(1000000)

plt.plot(model.rewards, label='Normalized Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.savefig(f'rewards_per_episode_with_optimizations_{environment_name}.png')