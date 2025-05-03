import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import itertools

from network import FeedForwardNN


class PPO:
    def __init__(self, env):
        self._init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.num_actors = self.act_dim  # One actor per action dimension

        # Initialize multiple actor networks and shared critic
        self.actor = [FeedForwardNN(self.obs_dim, 1) for _ in range(self.num_actors)]
        self.critic = FeedForwardNN(self.obs_dim, self.num_actors)

        # Fixed covariance matrix
        self.cov_var = torch.full((1,), 0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Optimizer for all actor networks
        actor_params = itertools.chain(*[a.parameters() for a in self.actor])
        self.actor_optim = Adam(actor_params, lr=self.lr, weight_decay=1e-4)
        self.critic_optim = Adam(
            self.critic.parameters(), lr=self.lr, weight_decay=1e-4
        )

        self.rewards = []

    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALg STEP 5
            # Calculate the advantage function
            advantage_k = batch_rtgs - V.detach()
            
            # Normalize the advantages
            advantage_k = (advantage_k - advantage_k.mean()) / (
                advantage_k.std() + 1e-10
            )

            for _ in range(self.n_updates_per_iteration):
                # calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratio
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calculate surrogate losses
                surr1 = ratios * advantage_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage_k

                # calculate action loss
                losses = [
                    (-torch.min(surr1[:, i], surr2[:, i])).mean()
                    for i in range(self.num_actors)
                ]
                total_loss = sum(losses)

                self.actor_optim.zero_grad()
                total_loss.backward()
                self.actor_optim.step()


                critic_loss = nn.MSELoss()(V, batch_rtgs.squeeze())

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
                print(total_loss, critic_loss)

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        log_probs = []

        for i, actor in enumerate(self.actor):
            mean = actor(batch_obs)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Select actions for actor i
            acts_i = batch_acts[:, i]  # Assuming batch_acts shape = [batch_size, num_actors, act_dim]
            acts_i = acts_i.unsqueeze(1)  # Convert from [4887] to [4887,1]
            log_prob = dist.log_prob(acts_i)
            log_probs.append(log_prob)

        return V, torch.stack(log_probs).T

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()[0]
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)

                # Get list of actions from each actor
                actions, log_probs = self.get_action(obs)

                # Step environment with all actions
                actions = actions.flatten()
                obs, rew, _, done, _, _ = self.env.step(actions)

                # Save each timestep's info
                batch_acts.append(actions)  # each actions = [num_actors, act_dim]
                batch_log_probs.append(log_probs)  # log_probs = [num_actors]
                ep_rews.append(rew)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # self.rewards += torch.tensor(batch_rews, dtype=torch.float).flatten().tolist()

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # Query action network for mean action
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        actions = []
        log_probs = []

        obs_tensor = obs_tensor.flatten()
        for actor in self.actor:
            mean = actor(obs_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.

        return torch.stack(actions).detach().numpy(), torch.stack(log_probs).detach()

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # iterate through each episode backwards to maintain the same order in batch_rtgs

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = np.array(rew) + (np.array(discounted_reward) * self.gamma)
                batch_rtgs.insert(0, discounted_reward)

        # convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048      # rollout buffer size per update
        self.max_timesteps_per_episode = 1000      # MuJoCo default time limit
        self.gamma = 0.99      # discount factor
        self.n_updates_per_iteration = 10        # epochs per update
        self.clip = 0.2       # PPO clipping ε
        self.lr = 3e-4      # learning rate