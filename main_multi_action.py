import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from ppo_multi_action import PPO as PPO_MultiAction
from ppo import PPO

env = gym.make('Pendulum-v1')
print(env.action_space.sample())

for i in range(10):
    model = PPO(env)
    model.learn(100000)

    total_rewards = sum(model.rewards)
    plt.plot(model.rewards, label='Normalized Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.savefig(f'rewards_per_episode${i}.png')
    
    model = PPO_MultiAction(env)
    model.learn(100000)

    total_rewards_multi_action = sum(model.rewards)
    plt.plot(model.rewards, label='Normalized Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.savefig(f'rewards_per_episode${i}_multi_action.png')
    
    print({"normal": total_rewards, "multi_action": total_rewards_multi_action, "multi_action - normal": total_rewards_multi_action - total_rewards})