import gymnasium as gym
from ppo_with_optimizations import PPO
import numpy as np

env = gym.make('Pendulum-v1')
print(env.action_space.sample())
model = PPO(env)
model.learn(10000)

print(model.rewards)

import matplotlib.pyplot as plt


plt.plot(model.rewards, label='Normalized Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.savefig('rewards_per_episode.png')