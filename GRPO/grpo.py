
import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            )
        self.actor = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        return self.actor(self.layer(x))

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        logits = self(state_t)  # this just calls forward
        
        # Softmax logits internally with torch.distributions.Categorical
        dist = Categorical(logits=logits)

        action = dist.sample() # gives action as an tensor with dtype int I think
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob)

    def evaluate_actions(self, states, actions):
        # Re-score the picks that select_action made
        states_t = torch.as_tensor(states, dtype=torch.float32)
        logits = self(states_t)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.long))
        entropy = dist.entropy()
        return (new_log_probs, entropy)

def compute_score(rewards):
    scores = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
    return scores.flatten() # flattens (num_groups, group_size) to (num_groups * group_size,)
    # means it gives a list of group_size numbers of scores from the same situation that describe
    # how much better that attempt was than group_mates
    # and does that for num_groups times
