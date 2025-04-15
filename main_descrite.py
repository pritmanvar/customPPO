import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from ppo_with_descrite import PPO

env = gym.make('Acrobot-v1')
print(env.action_space.sample())
print(env.action_space)
print("HII" if env.action_space.shape else "HELLO")
print(env.action_space.__class__.__name__)
model = PPO(env)
model.learn(1000000)

plt.plot(model.rewards, label='Normalized Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.savefig('rewards_per_episode_descrite.png')