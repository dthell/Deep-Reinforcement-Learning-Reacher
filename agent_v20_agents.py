import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_v20_agents import Actor, Critic, ActorCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUMBER_UPDATE_PASSES = 20  # number of learning passes every time the learning from the replay buffer is called
NOISE_DECAY = 0.99999   # speed decay of the OU noise
GAE_LAMBDA = 0.0        # GAE lambda parameter
BLEND_WEIGHT = 1.0      # Weight to apply to the critic loss to minimize in the A2C algorithm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_DDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, learning_frequency = 10, lr_actor = LR_ACTOR, lr_critic = LR_CRITIC, weight_decay = WEIGHT_DECAY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        #start with same weights
        self.soft_update(self.actor_local, self.actor_target, 1.0)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        #start with same weights
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_ratio = NOISE_DECAY

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # 20 elements represent over 99% of infinite sum for lamba up to 0.8
        self.GAE_lambdas = np.array([[(1-GAE_LAMBDA)*GAE_LAMBDA**i] for i in range(20)])
        # normalization as the array is not infinite to avoid biasing rewards
        self.GAE_lambdas = self.GAE_lambdas / np.sum(self.GAE_lambdas)

        # Learning frequency (to update the networks every N timesteps)
        self.learning_freq = learning_frequency
        self.steps = 0
        self.num_passes = NUMBER_UPDATE_PASSES
        
        # store the number of parallel agents
        self.num_agents = num_agents
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        reward_GAE = []
        # compute GAE reward over at most 20 future steps or whatever is available to avoid crashing
        for _ in range(self.num_agents):
            reward_GAE.append(np.sum(self.GAE_lambdas[:min(20,len(reward))] * reward[-min(20,len(reward)):]))
            
        # Save experience / reward
        for a in range(self.num_agents):
            self.memory.add(state[a], action[a], reward_GAE[a], next_state[a], done[a])
        
        self.steps += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.steps % self.learning_freq == 0:
            for _ in range(self.num_passes):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            self.steps = 0

    def act(self, state, add_noise=True, train_mode = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        actions = []
        with torch.no_grad():
            actions=self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            for a in range(self.num_agents):
                actions[a] += self.noise.sample() * self.noise_ratio
            self.noise_ratio = self.noise_ratio * NOISE_DECAY
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        #self.noise_ratio = NOISE_DECAY

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip gradient to improve stability
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Agent_A2C():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, learning_frequency = 10, lr = LR_ACTOR, weight_decay = WEIGHT_DECAY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Critic Network (first layer shared)
        self.actor_critic = ActorCritic(state_size, action_size, random_seed, num_agents).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.blend_weight = BLEND_WEIGHT
        self.lr = lr

        # to store experiences before gradients, will be emptied after each learning step as A2C is an on-policy algorithm
        self.experiences = []
        # 20 elements represent over 99% of infinite sum for lamba up to 0.8
        self.GAE_lambdas = np.array([[(1-GAE_LAMBDA)*GAE_LAMBDA**i] for i in range(20)])
        # normalization as the array is not infinite to avoid biasing rewards
        self.GAE_lambdas = self.GAE_lambdas / np.sum(self.GAE_lambdas)

        # Learning frequency (to update the network every N timesteps)
        self.learning_freq = learning_frequency
        self.steps = 0
        
        # store the number of parallel agents
        self.num_agents = num_agents
        self.std = np.ones(self.num_agents)
    
    def step(self, action, log_prob, value, reward, done, next_state):
        """Save experience in memory."""
                
        # inverse done boolean for future use
        not_done = [not c for c in done]
        # Save experience / reward
        self.experiences.append([action, log_prob, value, reward, not_done, next_state])
        
        self.steps += 1

    def act(self, state, add_noise=True, train_mode = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        if train_mode == False:
            self.actor_critic.eval()
        action, log_prob, value = self.actor_critic(state)
        if train_mode == True:
            self.actor_critic.train()
        
        return action, log_prob, value

    def learn(self, experiences, gamma = GAMMA):
        """Update policy and value parameters using given set of past experiences from current policy.
        Advantage = r + γ * critic(next_state) - critic(state)

        Params
        ======
            experiences (Array[torch.Tensor]): array of (a, log_prob, value, r, done, next_state) tuples 
            gamma (float): discount factor
        """

        advantage = [None] * (len(experiences) - 1)
        log_probs = [None] * (len(experiences) - 1)
        returns = [None] * (len(experiences) - 1)
        values = [None] * (len(experiences) - 1)
        
        _, _, value, _, _, _ = experiences[len(experiences)-1]
        _return = value.detach()  # get last value
        
        # compute advantages
        # loop in decreasing order until -1 index to fill i = 0
        for i in range(len(experiences)-2, -1, -1):  
            action, log_prob, value, reward, not_done, next_state = experiences[i]
            _, _, next_value, _, _, _ = experiences[i+1]
            _reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
            _not_done = torch.tensor(not_done,device=device).unsqueeze(1).float()
            _return = _reward + gamma * _not_done * _return                     # compute discounted return
            returns[i] = _return
            advantage[i] = (_reward  + gamma * next_value.detach() * _not_done - value.detach() )
            log_probs[i] = log_prob
            values[i] = value
            
        log_probs = torch.cat(log_probs)
        advantage = torch.cat(advantage)
        returns = torch.cat(returns)
        values = torch.cat(values)

        # compute current policy loss and critic loss
        policy_loss = - log_probs * advantage
        value_loss = F.mse_loss(returns, values)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        (policy_loss + self.blend_weight * value_loss).mean().backward()

        # clip gradient to improve stability
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 5)
        self.optimizer.step()
        
        # clear out experiences as policy has been updated
        self.experiences[:] = []

    def adapt_learning_rate(self, ratio = 0.999):
        self.lr *= ratio
        if self.lr < 1e-7:
            self.lr = 1e-7
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)