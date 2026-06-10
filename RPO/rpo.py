
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, noise_scale):
        super().__init__()
        self.noise_scale = noise_scale
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            )
        self.actor = nn.Linear(hidden_dim, action_dim) # logits
        self.critic = nn.Linear(hidden_dim, 1) # value
    
    def forward(self, x):
        x = self.layer(x)

        # Bring value down to scalar
        return (self.actor(x), self.critic(x).squeeze(-1))

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)

        clean_logits, value = self(state_t)  # just calls forward

        # RPO: add Gaussian noise
        noise = torch.randn_like(clean_logits) * self.noise_scale
        noisy_logits = clean_logits + noise

        # Softmax logits internally with torch.distributions.Categorical
        clean_dist = Categorical(logits=clean_logits)
        noisy_dist = Categorical(logits=noisy_logits)

        action = noisy_dist.sample()
        clean_log_prob = clean_dist.log_prob(action)
        noisy_log_prob = noisy_dist.log_prob(action)
        
        return (action, clean_log_prob, noisy_log_prob, value)
    
    def evaluate_actions(self, states, actions):
        # Re-score the picks that select_action made
        states_t = torch.as_tensor(states, dtype=torch.float32)

        logits, values = self(states_t)

        dist = Categorical(logits=logits)

        new_log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.long))
        entropy = dist.entropy()

        return (new_log_probs, entropy, values)

# Turn one batch of experiences into two training signals
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda, num_envs, n_steps):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.stack(values, dim=0)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    last_value = torch.as_tensor(last_value, dtype=torch.float32)

    last_value = last_value.unsqueeze(dim=0)
    values = torch.cat((values, last_value), dim=0) # so V(t+1) always exists even for the last real step

    advantages = []
    gae = torch.zeros(num_envs) # running accumulator.

    # We walk backwards thru time
    for t in reversed(range(len(rewards))): # T = len(rewards)
        # mask zeros delta and gae when dones[t] is true
        # if the episode continued at step t, mask is 1.
        # if not, mask is 0.
        # When episode ends, the next state is from a new
        # unrelated episode, mask erases the fake connection there
        mask  = 1.0 - dones[t]

        # reward gained + discounted value of landing - value expected
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]

        # Blend in the advantage from all the following steps too
        # GAE recursion
        gae = delta + gamma * gae_lambda * mask * gae

        # Prepend the result bc we iterate backwards
        advantages.insert(0, gae)
    
    # convert to tensors
    advantages = torch.stack(advantages, dim=0)
    values_t = values[:-1]
    
    # Return is how wrong the critic was (advantage) + what the critic guessed (value)
    returns = advantages + values_t

    # normalize
    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

    return (advantages, returns)

