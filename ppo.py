import torch
import torch.nn as nn
import copy
import os
import numpy as np
from tqdm import tqdm
from collections import deque

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # actor network - this takes in the state and outputs action logits
        # this is the main network used for selecting actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        # critic network - this takes in the state and outputs state value
        # the state value is essentially a prediction of future rewards from this state
        # if you follow the current policy, used to evaluate how good the state is
        # but also how much better taking a specific action is compared to the expected value at that state
        # it is used for advantage estimation during training
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def select_action(self, state):            
        action_probs = torch.softmax(self.actor(state), dim=-1) # get action probabilities
        dist = torch.distributions.Categorical(action_probs) # create categorical distribution
        action = dist.sample() # sample action from distribution
        # get log probability of sampled action
        # #used for training to compute loss and optimize policy
        action_logprob = dist.log_prob(action) 
        state_value = self.critic(state).squeeze(-1) # get state value from critic
        # return the action, logprob, and state value
        return action, action_logprob, state_value
    
    def evaluate_action(self, state, action):
        action_probs = torch.softmax(self.actor(state), dim=-1) # get action probabilities
        dist = torch.distributions.Categorical(action_probs) # create categorical distribution
        action_logprobs = dist.log_prob(action) # get log probability of given action
        dist_entropy = dist.entropy() # get entropy of the action distribution
        state_value = self.critic(state).squeeze(-1) # get state value from critic
        # return the log probabilities, state value, and entropy
        return action_logprobs, state_value, dist_entropy
    
class PPOTrainer:
    def __init__(
        self, agent, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, 
        clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01, max_grad_norm=0.5,
        epochs=10, batch_size=10000, norm_advantages=True, clip_value_loss=False, 
        target_kl=None, opponent_pool_size=20, save_opponent_interval=400000, save_agent_interval=20000000,
        swap_opponent_interval=200000, play_against_latest_prob=0.5, 
        win_rate_threshold=0.55, device="cpu", temp_pool_path="./opponent_checkpoints", checkpoint_path="./agent_checkpoints"
    ):
        self.agent = agent.to(device) # main agent being trained
        self.env = env # environment to train on
        self.device = device # device to run training on
        self.num_envs = env.num_envs # number of parallel environments

        self.gamma = gamma # discount factor for future rewards
        self.gae_lambda = gae_lambda # lambda for GAE

        self.clip_epsilon = clip_epsilon # PPO clipping epsilon
        self.value_coeff = value_coeff # coefficient for value loss
        self.entropy_coeff = entropy_coeff # coefficient for entropy bonus
        self.max_grad_norm = max_grad_norm # max gradient norm for clipping

        self.epochs = epochs # number of training epochs per update
        self.batch_size = batch_size # number of samples per training batch
        self.rollout_steps = self.env.max_steps # number of steps per rollout same as max updates

        self.norm_advantages = norm_advantages # whether to normalize advantages
        self.clip_value_loss = clip_value_loss # whether to clip value loss
        self.target_kl = target_kl # target KL divergence for early stopping

        self.opponent_pool_size = opponent_pool_size # size of opponent pool
        self.save_opponent_interval = save_opponent_interval # interval to save opponents
        self.swap_opponent_interval = swap_opponent_interval # interval to swap opponents
        self.save_agent_interval = save_agent_interval # interval to save main agent
        self.play_against_latest_prob = play_against_latest_prob # probability to play against latest opponent
        self.win_rate_threshold = win_rate_threshold # win rate threshold for opponent swapping

        self.next_opponent_swap = self.swap_opponent_interval # next timestep to swap opponent
        self.next_opponent_save = self.save_opponent_interval # next timestep to save opponent
        self.next_agent_save = self.save_agent_interval # next timestep to save main agent

        self.temp_pool_path = temp_pool_path # temporary path to save opponents
        os.makedirs(self.temp_pool_path, exist_ok=True) # create temp pool directory if not exists

        self.checkpoint_path = checkpoint_path # path to save checkpoints
        os.makedirs(self.checkpoint_path, exist_ok=True) # create checkpoint directory if not exists

        self.opponent_pool = deque(maxlen=opponent_pool_size) # queue to store opponent agents (FILO)
        self.current_opponent_idx = None # index of current opponent in pool

        self.opponent = copy.deepcopy(self.agent).to(device) # current opponent agent, initialized as a copy of main agent
        self.opponent.eval() # set opponent to eval mode

        self.total_timesteps = 0 # total timesteps trained
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate) # adam optimizer for training

        self.episode_wins = [] # list to track episode wins
        self.episode_agent_scores = [] # list to track agent scores
        self.episode_opponent_scores = [] # list to track opponent scores
        self.episode_agent_rewards = [] # list to track agent rewards
        self.episode_opponent_rewards = [] # list to track opponent rewards

        self.episode_rewards_p1 = torch.zeros(self.num_envs, device=device) # tensor to track rewards for player 1
        self.episode_rewards_p2 = torch.zeros(self.num_envs, device=device) # tensor to track rewards for player 2
        self.is_agent_p1 = torch.rand(self.num_envs, device=device) < 0.5 # random bool tensor to decide if agent is player 1 for each env

    def make_opponent_filename(self, step):
        # function to save agent model filename
        return os.path.join(self.temp_pool_path, f"agent_step{step}.pth")
    
    def make_opponent_filename(self, idx, step):
        # function to save opponent model filename
        return os.path.join(self.temp_pool_path, f"opponent_idx{idx}_step{step}.pth")
    
    def save_agent(self):
        # save the current agent model to checkpoint path
        filename = os.path.join(self.checkpoint_path, f"agent_step{self.total_timesteps}.pth")
        torch.save(self.agent.state_dict(), filename)
        print(f"Saved agent checkpoint to {filename}")
    
    def save_to_pool(self):
        state_dict = {k: v.cpu().clone() for k, v in self.agent.state_dict().items()} # get state dict of agent
        filename = self.make_opponent_filename(len(self.opponent_pool), self.total_timesteps) # make filename
        torch.save(state_dict, filename) # save state dict to file
        meta = {
            "state": state_dict, # state dict of the opponent
            "path": filename, # path to saved file
            "step": self.total_timesteps, # training step when saved
        }
        
        # remove oldest opponent if pool is full
        if len(self.opponent_pool) >= self.opponent_pool_size: 
            oldest = self.opponent_pool[0]
            # delete the file from disk
            if os.path.exists(oldest["path"]):
                os.remove(oldest["path"])

        self.opponent_pool.append(meta) # add new opponent to pool, automatically removes oldest if full
        print(f"Saved opponent (pool_size={len(self.opponent_pool)}) to {filename}")

    def delete_temp_pool(self):
        # delete all temporary opponent files from disk
        for meta in self.opponent_pool:
            if os.path.exists(meta["path"]):
                os.remove(meta["path"])
        os.rmdir(self.temp_pool_path) # remove the temp directory
        print(f"Deleted temporary opponent pool directory {self.temp_pool_path}")

    def sample_opponent(self):
        # sample an opponent from the pool or use latest agent based on probability
        if len(self.opponent_pool) == 0 or np.random.rand() < self.play_against_latest_prob:
            self.opponent.load_state_dict(self.agent.state_dict()) # use latest agent as opponent
            self.current_opponent_idx = -1 # indicate latest agent
            return
        
        idx = np.random.randint(0, len(self.opponent_pool)) # sample random index from pool
        meta = self.opponent_pool[idx] # get opponent metadata
        opponent_state = {k: v.to(self.device) for k, v in meta["state"].items()} # move state dict to device
        self.opponent.load_state_dict(opponent_state) # load state dict into opponent
        self.current_opponent_idx = idx # set current opponent index

    def augment_state(self, state, player_indicator):
        # augment state with player indicator (1 if agent is player 1, else 0) to help agent distinguish roles
        indicator = player_indicator.unsqueeze(1).float()
        return torch.cat([state, indicator], dim=1)
    
    def compute_gae(self, rewards, values, dones, next_values):
        # advantage estimation using Generalized Advantage Estimation (GAE)
        # advantage - how much better an action is compared to the expected value at that state
        # this essentially means the total expected future rewards minus the value estimate
        # returns - total expected future rewards, based on advantage and value estimates
        # GAE - reduces variance of advantage estimates while introducing some bias
        # formula: A_t = delta_t + (gamma * lambda) * (1 - done_{t+1}) * A_{t+1}
        # where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        # and A_{T-1} = delta_{T-1}, delta_{T-1} = r_{T-1} + gamma * V(s_T) - V(s_{T-1})
        # the returns are calculated as R_t = A_t + V(s_t)
        # we compute advantages backwards from the end of the trajectory
        # it works because it doesnt total returns (R_t - V(s_t)) directly
        # this is because total returns can have high variance, which means unstable training
        # but also not 1-step TD error (delta_t) since it has high bias
        # which can lead to suboptimal policies as it doesn't consider long-term rewards
        # instead it uses a weighted sum of n-step TD errors to balance bias and variance
        # it does this by using the lambda parameter to control the weighting of A_t+1 on A_t
        # this allows a balance between bias and variance in the advantage estimates
        # leading to more stable and efficient policy updates

        advantages = torch.zeros_like(rewards) # shape: (num_envs, 2)
        last_gae = torch.zeros(self.num_envs, device=self.device) # initialize last GAE to 0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = next_values # V(s_{T})
                next_done = torch.zeros(self.num_envs, device=self.device) # 0 so we can discard

            else:
                next_val = values[t + 1] # V(s_{t+1})
                next_done = dones[t + 1] # done_{t+1}

            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]

            # A_t = delta_t + (gamma * lambda) * (1 - done_{t+1}) * A_{t+1}
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae

        # R_t = A_t + V(s_t) 
        returns = advantages + values
        return advantages, returns
    
    def collect_rollout(self):
        # storage tensors for rollout data
        states = torch.zeros((self.rollout_steps, self.num_envs, self.env.get_state().shape[1] + 1), device=self.device, dtype=torch.float32) # states
        actions = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.long) # actions
        log_probs = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float32) # log probabilities of actions
        values = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float32) # value estimates
        rewards = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float32) # rewards
        dones = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float32) # done flags
        state = self.env.get_state() # initial state from environment, used to keep track of current state

        # rollout collection loop, step through the environment, and collect data
        for step in tqdm(range(self.rollout_steps), desc="Collecting Rollout"): 
            with torch.no_grad():
                agent_indicator = (~self.is_agent_p1).float() # indicator for agent being player 1 or 2
                agent_state = self.augment_state(state, agent_indicator) # augment state with player indicator

                # select action using current agent policy, and get log prob and value estimate
                action_agent, log_prob_agent, value_agent = self.agent.select_action(agent_state) 

                # select action for opponent, we only care about the action for the opponent
                opp_indicator = self.is_agent_p1.float()
                opp_state = self.augment_state(state, opp_indicator)
                action_opp, _, _ = self.opponent.select_action(opp_state)

            env_actions_p1 = torch.where(self.is_agent_p1, action_agent, action_opp) # actions for player 1, chosen based on whether the agent is player 1
            env_actions_p2 = torch.where(self.is_agent_p1, action_opp, action_agent) # actions for player 2, chosen based on whether the agent is player 2

            next_state, env_rewards, env_dones, info = self.env.step((env_actions_p1, env_actions_p2)) # step all the environments with the chosen actions
            agent_rewards = torch.where(self.is_agent_p1, env_rewards[:, 0], env_rewards[:, 1]) # get the rewards for the agent based on its role

            self.episode_rewards_p1 += env_rewards[:, 0] # accumulate rewards for player 1
            self.episode_rewards_p2 += env_rewards[:, 1] # accumulate rewards for player 2

            states[step] = agent_state # store augmented state
            actions[step] = action_agent # store agent action
            log_probs[step] = log_prob_agent # store log prob of agent action
            values[step] = value_agent # store value estimate
            rewards[step] = agent_rewards # store agent rewards
            dones[step] = env_dones.float() # store done flags

            done_envs = torch.where(env_dones)[0]  # get indices of done environments

            if done_envs.numel() > 0:
                # get player 1 and player 2 scores for done environments
                p1_scores = info["player_1_score"][done_envs] # tensor of scores
                p2_scores = info["player_2_score"][done_envs]

                # check if agent is player 1 in these envs
                is_p1 = self.is_agent_p1[done_envs] # boolean tensor

                # get agent and opponent scores based on role
                agent_scores = torch.where(is_p1, p1_scores, p2_scores)
                opp_scores = torch.where(is_p1, p2_scores, p1_scores)

                # get agent and opponent rewards based on role
                agent_rewards = torch.where(is_p1, self.episode_rewards_p1[done_envs], self.episode_rewards_p2[done_envs])
                opp_rewards = torch.where(is_p1, self.episode_rewards_p2[done_envs], self.episode_rewards_p1[done_envs])

                # determine win/loss/draw: 1.0 win, 0.0 loss, 0.5 draw
                wins = torch.where(agent_scores > opp_scores, 1.0, torch.where(agent_scores < opp_scores, 0.0, 0.5))

                # store results
                self.episode_wins.extend(wins.tolist()) # convert to list to append
                self.episode_agent_scores.extend(agent_scores.tolist())
                self.episode_opponent_scores.extend(opp_scores.tolist())
                self.episode_agent_rewards.extend(agent_rewards.tolist())
                self.episode_opponent_rewards.extend(opp_rewards.tolist())

                # reset rewards and randomly assign agent role only for done envs
                self.episode_rewards_p1[done_envs] = 0.0 # reset rewards for done envs
                self.episode_rewards_p2[done_envs] = 0.0
                self.is_agent_p1[done_envs] = torch.rand(len(done_envs), device=self.device) < 0.5 # reassign roles
                self.env.reset(done_envs) # reset done environments

            state = next_state # update current state
            self.total_timesteps += self.num_envs # increment total timesteps

            if self.total_timesteps >= self.next_opponent_swap:
                # if it's time to swap opponent, sample a new one
                self.sample_opponent()
                self.next_opponent_swap += self.swap_opponent_interval # schedule next swap time

            if self.total_timesteps >= self.next_opponent_save:
                # if it's time to save opponent, check win rate and save if criteria met
                win_rate = np.mean(self.episode_wins[-100:]) if len(self.episode_wins) >= 100 else 0.5
                # only save if win rate threshold met or pool is empty
                if win_rate >= self.win_rate_threshold or len(self.opponent_pool) == 0:
                    self.save_to_pool()
                self.next_opponent_save += self.save_opponent_interval # schedule next save time

            if self.total_timesteps >= self.next_agent_save:
                # if it's time to save the main agent, do so
                self.save_agent()
                self.next_agent_save += self.save_agent_interval # schedule next agent save time

        with torch.no_grad():
            agent_indicator = (~self.is_agent_p1).float() # indicator for agent being player 1 or 2
            next_augmented = self.augment_state(state, agent_indicator) # augment state with player indicator
            _, _, next_values = self.agent.select_action(next_augmented) # get value estimates for next states, so we can compute advantages with GAE

        return states, actions, log_probs, values, rewards, dones, next_values
    
    def update_policy(self, states, actions, old_log_probs, old_values, rewards, dones, next_values):
        # here is where the main PPO update happens
        # goals of PPO:
        # 1. actor - update the policy to maximize expected advantage 
        # while ensuring new policy doesn't deviate too much from old policy with clipping
        # 2. critic - update value function to minimize error between predicted values and actual returns
        # 3. add entropy bonus to encourage exploration and prevent premature convergence
        # formulas: ratio = pi_theta(a|s) / pi_theta_old(a|s), checks the ratio in policy change
        # we do: ratio = exp(log(pi_theta(a|s)) - log(pi_theta_old(a|s))) = pi_theta(a|s) / pi_theta_old(a|s)
        # this is done for numerical stability, as probabilities can be very small
        # actor loss: L_clip = -E[min(ratio * A, clip(ratio, 1-epsilon, 1+epsilon) * A)]
        # critic loss: L_value = 0.5 * E[(V(s) - R)^2], but with optional clipping:
        # V_clipped = V_old(s) + clip(V(s) - V_old(s), -epsilon, epsilon)
        # L_value = 0.5 * E[max((V_clipped - R)^2, (V(s) - R)^2)]
        # entropy loss: L_entropy = -E[entropy]
        # total loss: L_total = L_clip + c1 * L_value + c2 * L_entropy
        # where c1 and c2 are coefficients for value loss and entropy bonus
        # we optimize this total loss using mini-batch SGD over multiple epochs
        # so essentially we do:
        # 1. collect rollout data
        # 2. compute advantages and returns using GAE
        # 3. for multiple epochs, shuffle and create mini-batches
        # 4. compute losses and update policy using optimizer
        # 5. repeat until convergence (for a set number of updates, I used 5 updates of 20M steps each)

        with torch.no_grad():
            # compute advantages and returns using GAE
            advantages, returns = self.compute_gae(rewards, old_values, dones, next_values) 

        # flatten all tensors for batching
        states_flat = states.reshape(-1, states.shape[-1]) 
        actions_flat = actions.reshape(-1)
        old_log_probs_flat = old_log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        old_values_flat = old_values.reshape(-1)

        if self.norm_advantages:
            # normalize advantages to have mean 0 and std 1 for stable training
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        num_samples = states_flat.shape[0] # total number of samples
        # create progress bar for epochs and batches, with total number of updates calculated
        pbar = tqdm(range(self.epochs * (num_samples // self.batch_size + int(num_samples % self.batch_size > 0))), desc="Updating Policy")

        for _ in range(self.epochs):
            indices = torch.randperm(num_samples, device=self.device) # shuffle indices for batching

            for start in range(0, num_samples, self.batch_size):
                # prepare mini-batch
                end = min(start + self.batch_size, num_samples)
                batch_idx = indices[start:end]
                batch_states = states_flat[batch_idx]
                batch_actions = actions_flat[batch_idx]
                batch_old_log_probs = old_log_probs_flat[batch_idx]
                batch_advantages = advantages_flat[batch_idx]
                batch_returns = returns_flat[batch_idx]
                batch_old_values = old_values_flat[batch_idx]

                # evaluate actions with current policy, get new log probs, values, and entropy, for ratio calculation
                new_log_probs, new_values, entropy = self.agent.evaluate_action(batch_states, batch_actions)
                 
                # ratio = exp(log(pi_theta(a|s)) - log(pi_theta_old(a|s))) = pi_theta(a|s) / pi_theta_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs) 
                # s1 = ratio * A
                surr1 = ratio * batch_advantages
                # s2 = clip(ratio, 1-epsilon, 1+epsilon) * A
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                # actor loss = -E[min(s1, s2)]
                actor_loss = -torch.min(surr1, surr2).mean()

                # compute value loss
                if self.clip_value_loss:
                    # clipped value loss to prevent large updates
                    # V_clipped = V_old(s) + clip(V(s) - V_old(s), -epsilon, epsilon)
                    values_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -self.clip_epsilon,self.clip_epsilon)
                    # V_clipped - R)^2
                    value_loss_clipped = (values_clipped - batch_returns).pow(2)
                    # (V(s) - R)^2
                    value_loss_unclipped = (new_values - batch_returns).pow(2)
                    # L_value = 0.5 * E[max((V_clipped - R)^2, (V(s) - R)^2)]
                    critic_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    # standard value loss
                    # L_value = 0.5 * E[(V(s) - R)^2]
                    critic_loss = 0.5 * (new_values - batch_returns).pow(2).mean()

                # entropy loss to encourage exploration
                entropy_loss = -entropy.mean()
                # total loss is weighted sum of actor, critic, and entropy losses
                total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

                # backpropagation and optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm) # gradient clipping
                self.optimizer.step()

                pbar.update(1) # update progress bar

            # early stopping based on KL divergence, for stable updates
            # to add extra safety against large policy updates per update cycle
            # KL divergence: D_KL(pi_theta_old || pi_theta) = E[log(pi_theta_old(a|s)) - log(pi_theta(a|s))]
            if self.target_kl is not None:
                with torch.no_grad():
                    new_log_probs_all, _, _ = self.agent.evaluate_action(states_flat, actions_flat)
                    kl_div = (old_log_probs_flat - new_log_probs_all).mean().item()
                    if kl_div > self.target_kl:
                        break
        
        # close progress bar and return losses for logging
        pbar.close()
        return actor_loss.item(), critic_loss.item(), entropy_loss.item(), total_loss.item()
    
    def train(self, total_timesteps, log_interval=10):
        num_updates = total_timesteps // (self.rollout_steps * self.num_envs) # total number of updates to perform

        print(f"Starting Self-Play PPO Training")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Num environments: {self.num_envs}")
        print(f"Steps per rollout: {self.rollout_steps}")
        print(f"Timesteps per update: {self.rollout_steps * self.num_envs:,}")
        print(f"Total updates: {num_updates}")
        self.save_to_pool() # save initial agent to opponent pool

        for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
            # collect rollout data
            states, actions, log_probs, values, rewards, dones, next_values = self.collect_rollout()
            # update policy using collected data
            actor_loss, critic_loss, entropy_loss, total_loss = self.update_policy(states, actions, log_probs, values, rewards, dones, next_values)

            # log the win rate, scores, rewards, and losses at intervals
            if update % log_interval == 0:
                n_recent = min(100, len(self.episode_wins))
                win_rate = np.mean(self.episode_wins[-n_recent:]) if n_recent > 0 else 0.5
                avg_agent_score = np.mean(self.episode_agent_scores[-n_recent:]) if n_recent > 0 else 0.0
                avg_opp_score = np.mean(self.episode_opponent_scores[-n_recent:]) if n_recent > 0 else 0.0
                avg_agent_reward = np.mean(self.episode_agent_rewards[-n_recent:]) if n_recent > 0 else 0.0
                avg_opp_reward = np.mean(self.episode_opponent_rewards[-n_recent:]) if n_recent > 0 else 0.0
                
                print(f"Update {update}/{num_updates}, Steps: {self.total_timesteps:,}")
                print(f"Win Rate (last {n_recent}): {win_rate:.2%}")
                print(f"Agent: Score={avg_agent_score:.2f}, Reward={avg_agent_reward:.2f}")
                print(f"Opponent: Score={avg_opp_score:.2f}, Reward={avg_opp_reward:.2f}")
                print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}, Total={total_loss:.4f}")

        self.delete_temp_pool() # clean up temporary opponent files after training

    def save_model(self, filepath):
        torch.save(self.agent.state_dict(), filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device) # load checkpoint
        self.agent.load_state_dict(checkpoint) # load state dict into agent
        print(f"Loaded checkpoint from {filepath}")

if __name__ == "__main__":
    from game import SoccerGameEnv # import the soccer game environment
    
    # setup the correct device for training
    # I am using a Mac M4 Max with 36GB unified memory with MPS
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
        
    env = SoccerGameEnv(
        num_envs=10000, # 10000 parallel environments for faster data collection
        render=False,
        max_steps=2000, # max steps per episode
        device=device
    ) # based on this, thats 20 million steps per update

    # set up the agent
    state_dim = env.get_state().shape[1] + 1 # state dim is env state + 1 for player indicator
    action_dim = env.action_dim
    agent = ActorCriticNetwork(state_dim, action_dim).to(device)
    
    # set up the PPO trainer
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        device=device 
    )

    trainer.train(total_timesteps=200000000, log_interval=1) # train for 200 million steps, so 200M / 20M = 10 updates, and 200M / 2k = 100k games played!
    trainer.save_model('soccer_final.pth') # save final model, on my device training took just under 1 hour! 