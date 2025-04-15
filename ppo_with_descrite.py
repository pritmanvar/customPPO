import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        self._init_hyperparameters()

        # Initialize the environment
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
        self.act_type = env.action_space.__class__.__name__
        
        # Initialize the networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # Create variables for the matrix
        self.cov_var = torch.full((self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr, weight_decay=1e-4)
        
        self.rewards = []
    def learn(self, total_timesteps):
        t_so_far = 0
        
        while t_so_far < total_timesteps:
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALg STEP 5
            # Calculate the advantage function
            advantage_k = batch_rtgs - V.detach()
            
            # Normalize the advantages
            advantage_k = (advantage_k - advantage_k.mean()) / (advantage_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):
                # calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                # Calculate ratio
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                # calculate surrogate losses
                surr1 = ratios * advantage_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage_k
                
                # calculate action loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                # Claculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        
        # Calculate the logs probabilities of batch actions using most recent actin network.
        # This segment of code is similar to that in get_action()
        if self.act_type == 'Box':
            mean = self.actor(batch_obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)
        else:
            # Discrete action space
            logits = self.actor(batch_obs)
            probs = Categorical(logits=logits)
            log_probs = probs.log_prob(batch_acts)
        
        return V, log_probs
    
    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        
        # Number of timesteps run so far this batch
        t = 0
        
        while t < self.timesteps_per_batch:
            # Rewards this epishode
            ep_rews = []

            obs = self.env.reset()[0]
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps run this batch so far
                t += 1
                
                # Collect observation
                batch_obs.append(obs)
                                
                action, log_prob = self.get_action(obs)
                
                obs, rew, done, _, _ = self.env.step(action)
                
                print(rew, action)
                
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                if done:
                    break
            
            # Colect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
        flat_rewards = [r for ep in batch_rews for r in ep]
        self.rewards += torch.tensor(flat_rewards, dtype=torch.float).tolist()
        
        # Reshape data as tensor in the shape specified before returning.
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def get_action(self, obs):
        if self.act_type == 'Box':
            # Query action network for mean action
            mean = self.actor(obs)
            # Create our multivariate normal distribution
            dist = MultivariateNormal(mean, self.cov_mat)
            
            # sample an actin from the distribution and get its log prob
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Return the sampled action and the log prob of that action
            # Note that I'm calling detach() since the action and log_prob  
            # are tensors with computation graphs, so I want to get rid
            # of the graph and just convert the action to numpy array.
            # log prob as tensor is fine. Our computation graph will
            # start later down the line.
            return action.detach().numpy(), log_prob.detach()
        else:
            # Discrete action space
            logits = self.actor(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)
            return action.item(), log_prob.detach()
        
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # iterate through each episode backwards to maintain the same order in batch_rtgs
        
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
        # convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        
        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95                        # Discount factor
        self.n_updates_per_iteration = 5         # Number of opochs per iteration
        self.clip = 0.2                          # clip thresold
        self.lr = 0.005