import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from ppo_multi_action import PPO as PPO_MultiAction
from ppo import PPO
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

        reward = [rewards - cost for cost in reward_cost]

        if self.render_mode == "human":
            self.render()
        return observation, reward, combined_reward, terminated, False, info

timesteps = int(input("Enter the number of timesteps: "))
env_multi_action = CustomEnvWrapper(gym.make('Ant-v4'))
model_multi_action = PPO_MultiAction(env_multi_action)
model_multi_action.learn(timesteps)

env = gym.make('Ant-v4', render_mode='rgb_array')
model = PPO(env)
model.learn(timesteps)


results_df = pd.DataFrame(columns=["Step", "Multi-Reward-Mean", "Multi-Reward-Combined", "Single-Reward", "Mean - Single", "Combined - Single", "Multi reward ctrl", "Single reward ctrl", "reward ctrl multi - single"])
for i in range(10000):
    inv = np.random.randint(0, 2)
    obs, _ = env.reset() if inv == 0 else env_multi_action.reset()

    action_multi_action, _ = model_multi_action.get_action(obs)
    action, _ = model.get_action(obs)
    
    # Squeeze actions to match environment expectations
    action_multi_action = np.squeeze(action_multi_action)
    action = np.squeeze(action)
    
    
    obs_multi_action, reward_multi_action, combined_reward, terminated_multi_action, truncated_multi_action, info_multi_action = env_multi_action.step(action_multi_action)
    obs, reward, terminated, truncated, info = env.step(action)
    
    results_df.loc[i] = [
        i + 1,
        np.mean(reward_multi_action),
        combined_reward,
        reward,
        np.mean(reward_multi_action) - reward,
        combined_reward - reward,
        info_multi_action['reward_ctrl'],
        info['reward_ctrl'],
        info_multi_action['reward_ctrl'] - info['reward_ctrl']
    ]

results_df.to_csv(f"ant_v4_results_{timesteps}.csv", index=False)