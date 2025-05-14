import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from my_stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def __getattr__(self, name):
        # First try to get the attribute from the wrapped env.
        # If not available, try the unwrapped env.
        try:
            return getattr(self.env, name)
        except AttributeError:
            return getattr(self.env.unwrapped, name)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.square(action)
        return control_cost
    
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = reward_cost = self.control_cost(action)
        costs = ctrl_cost = np.sum(costs)
        combined_reward = rewards - costs
        
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = [rewards/8 - cost for cost in reward_cost]

        if self.render_mode == "human":
            self.render()
        return observation, reward, combined_reward, terminated, False, info


env = CustomEnvWrapper(gym.make("Ant-v4"))
model = PPO("MlpPolicy", env, verbose=1)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print(model.predict(env.reset()[0]))