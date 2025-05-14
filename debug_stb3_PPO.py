from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym

env = gym.make('Ant-v4', render_mode='rgb_array')
model = PPO("MlpPolicy", env)
# model.learn(timesteps)
breakpoint()
