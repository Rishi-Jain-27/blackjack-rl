# Import environment
import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple as TupleSpace

# Import ML libraries
import torch
import numpy as np
from torch.distributions import Categorical

# Import stuff from other files
from grpo import Actor, compute_score

# Import yaml for hyperparams
import yaml

# Itertools for indefinite looping
import itertools

# os for directories
import os

# MPL for plotting
import matplotlib.pyplot as plt

# datetime for datetime
from datetime import datetime, timedelta

# argparse for CLI training
import argparse

# for printing date and time
DATE_FORMAT = "%y-%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GRPOAgent:
    def __init__(self, hyperparameter_set):
        # Get hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.gamma = hyperparameters['gamma']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.learning_rate = hyperparameters['learning_rate']
        self.group_size = hyperparameters['group_size'] # we can do vectorized envs with this
        self.num_groups = hyperparameters['num_groups']
        self.n_epochs = hyperparameters['n_epochs']
        self.minibatch_size = hyperparameters['minibatch_size']
        self.clip_ratio = hyperparameters['clip_ratio']
        self.entropy_coef = hyperparameters['entropy_coef']
        self.max_grad_norm = hyperparameters['max_grad_norm']
        self.kl_coef = hyperparameters['kl_coef']
        self.stop_on_reward = hyperparameters['stop_on_reward']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def collect_rollout(self, envs, actor, rollout_idx):
        states = []
        actions = []
        old_log_probs = []
        hand_ids = []
        returns_grid = np.zeros((self.num_groups, self.group_size), dtype=np.float32)

        for group in range(self.num_groups):
            group_seed = group + self.num_groups * rollout_idx

            state, _ = envs.reset(seed=[group_seed] * self.group_size) # reset all groups, using the same seed
            live = np.ones(self.group_size, dtype=bool)

            while live.any():
                # convert tuple into a num_envs by 3 tensor
                states_table = np.stack(state, axis=-1).astype(np.float32)
                state_t = torch.as_tensor(states_table)

                logits = actor(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample().numpy()
                log_prob = dist.log_prob(torch.tensor(action))

                next_state, reward, terminated, truncated, _ = envs.step(action)
                done = terminated | truncated # or can't do arrays, | can

                # Record to the arrays
                hand_id = group * self.group_size + np.arange(self.group_size)
                hand_ids.append(hand_id[live])

                states.append(states_table[live]) # add just the rows for the environments still alive this turn
                actions.append(action[live])
                old_log_probs.append(log_prob.detach()[live])

                just_finished = done & live # find envs that finished on this turn
                returns_grid[group, just_finished] = reward[just_finished]
                live[just_finished] = False

                state = next_state

        # Before concat states is a list of the envs that are still alive for each turn
        # turn per-turn chunks into lists
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        hand_ids = np.concatenate(hand_ids, axis=0)

        # Does GRPO math and flattens
        scores_per_hand = compute_score(torch.as_tensor(returns_grid))

        # Then get the scores for each hand's id
        scores = scores_per_hand[torch.as_tensor(hand_ids)]

        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)

        rollout_idx += 1

        return states, actions, old_log_probs, scores, returns_grid.flatten(), rollout_idx

    def train(self, render=False):
        envs = gym.vector.SyncVectorEnv(make_env(self.env_id, render=render, **self.env_make_params) for _ in range(self.group_size))

        obs_space = envs.single_observation_space
        assert isinstance(obs_space, TupleSpace)
        num_states = len(obs_space.spaces)

        act_space = envs.single_action_space
        assert isinstance(act_space, Discrete)
        num_actions = int(act_space.n)

        actor = Actor(num_states, num_actions, self.hidden_dim)
        actor_optimizer = torch.optim.Adam(
            params=actor.parameters(),
            lr=self.learning_rate)

        rewards_per_episode = []
        mean_rewards = []
        best_mean_reward = float('-inf')

        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        self._log(log_message)
        
        rollout_idx = 1

        for i in itertools.count():
            # Getting the rollout & optimizing
            states, actions, old_log_probs, scores, episode_returns, rollout_idx = self.collect_rollout(envs, actor, rollout_idx)
            self.optimize(actor, actor_optimizer, states, actions, old_log_probs, scores)

            # Everything else — tracking, logging, saving, auto-stopping
            rewards_per_episode.extend(episode_returns)
            mean_reward = np.mean(rewards_per_episode[-100:])
            mean_rewards.append(mean_reward)
            
            if mean_reward > best_mean_reward:
                # log
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Episode {i} | New best mean reward: {mean_reward:.3f}")
                self._log(log_message)

                best_mean_reward = mean_reward

                torch.save(actor.state_dict(), self.MODEL_FILE)

            # Update graph every 30 seconds
            if datetime.now() - last_graph_update_time > timedelta(seconds=30):
                self.save_graph(mean_rewards)
                last_graph_update_time = datetime.now()
            
            # Auto stopping condition
            if mean_reward >= self.stop_on_reward and len(rewards_per_episode) >= 100:
                log_message = "Solved"
                self._log(log_message)

                break

# This is nearly ripped straight from ppo
    def optimize(self, actor, optimizer, states, actions, old_log_probs, scores):
        for epoch in range(self.n_epochs):
            # We want to go through all items n states, so find states.shape[0]
            # and make that the batch size

            # Squeeze multiple gradient steps out of one batch of environment interaction
            perm = torch.randperm(states.shape[0])
            # Walk through the rollout in chunks
            for start in range(states.shape[0] // self.minibatch_size):
                # Split things up into minibatches
                mb_idx = perm[start * self.minibatch_size : (start + 1) * self.minibatch_size]
                states_mb = states[mb_idx]
                actions_mb = actions[mb_idx]
                old_log_probs_mb = old_log_probs[mb_idx]
                scores_mb = scores[mb_idx]

                # Re-run the current network on the states and actions
                new_log_probs, entropy = actor.evaluate_actions(states_mb, actions_mb)

                # Importance-sampling ratio — numerically stable by subtracting in log space
                ratio = torch.exp(new_log_probs - old_log_probs_mb)

                # Push up actions with positive scores, push down with negative
                surr1 = ratio * scores_mb

                # Same but with ratio clamped — the trust region
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * scores_mb

                # Takes the smaller of the two surrogates per sample, negates bc optimizers want to minimize
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calc loss
                loss = policy_loss - self.entropy_coef * entropy.mean()
                
                # Zero grad + backprop
                optimizer.zero_grad()
                loss.backward()

                # Clip grad to guard against outlier minibatches
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                
                # Apply clipped gradients
                optimizer.step()
    
    def run(self, render=True):
        env = gym.make(self.env_id, render_mode="human", **self.env_make_params)
        
        # doing all this because of pylance
        obs_space = env.observation_space
        assert isinstance(obs_space, TupleSpace)
        num_states = len(obs_space.spaces)

        act_space = env.action_space
        assert isinstance(act_space, Discrete)
        num_actions = int(act_space.n)



        actor = Actor(num_states, num_actions, self.hidden_dim)
        actor.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True, map_location=torch.device('cpu')))
        actor.eval()

        # we don't need collect rollout here
        with torch.inference_mode():
            for i in itertools.count():
                state, _ = env.reset()

                done = False
                while not done:
                    action, _ = actor.select_action(state)
                    state, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

    def save_graph(self, mean_rewards):
        fig = plt.figure(1)
        plt.xlabel('Rollouts')
        plt.ylabel('Mean reward of last 100 eps')
        plt.plot(mean_rewards)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # Log helper function — for appending.
    def _log(self, msg):
        print(msg)
        with open(self.LOG_FILE, 'a') as f:
            f.write(msg + '\n')

# Make environment helper function for vectorized envs
def make_env(env_id, render=False, **env_make_params):
    def thunk():
        env = gym.make(env_id, render_mode="human" if render else None, **env_make_params)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test?")
    parser.add_argument('hyperparameters', help='Enter the name of the set of hyperparameters to test/train')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    grpo = GRPOAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        grpo.train() # python grpo_agent.py --train grpoblackjack
    else:
        grpo.run() # python grpo_agent.py grpoblackjack