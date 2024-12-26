import gym
from pathlib import Path
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from gym.spaces.utils import flatdim

from misc import get_action_split, soft_update, standardize_value
from models import (MLPValueCritic,MLPCategoricalActor,MLPMultiBinaryActor,MLPMultiCategoricalActor)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class MAPPO:
    def __init__(self,env,agent_idx,local_critic=False,clip_grads=True,actor_width=64,critic_width=128,activation_func="relu",
        actor_lr=3e-4,critic_lr=3e-4,gamma=0.90,v_coef=1.0,n_epochs=10,n_minibatches=4,normalize_adv=True,clip_value=False,
        clip_coef=0.2,entropy_coef=0.01,parameter_sharing=True,optimizer=None,d2rl=True,**kwargs):
        
        # Save Parameters:
        self.gamma = gamma
        self.v_coef = v_coef
        self.entropy_coef = entropy_coef
        self.clip_grads = clip_grads
        self.clip_value = clip_value
        self.clip_coef = clip_coef
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.normalize_adv = normalize_adv

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # Disable centralized critic
        self.local_critic = local_critic
        # Agent Observation space:
        num_states = flatdim(env.observation_space[agent_idx])
        # All agents observations:
        num_total_states = sum([flatdim(obsp) for obsp in env.observation_space])
        # Action space includes the action split
        agent_action_space = env.action_space[agent_idx]
        num_actions = flatdim(agent_action_space)
        action_split = get_action_split(agent_action_space)
        # All agents actions:
        num_total_actions = sum([flatdim(acsp) for acsp in env.action_space])
        # num_input_critic = num_total_states + num_total_actions

        # Independent or Centralized State:
        if self.local_critic:
            num_input_critic = num_states
        else:
            num_input_critic = num_total_states

        self.dict_args = {"num_states": num_states,"actor_width": actor_width,"num_actions": int(num_actions),
                          "action_split": action_split,"activation_func": activation_func,"d2rl": d2rl}

        # Networks
        if type(agent_action_space) == gym.spaces.MultiBinary:
            actor_class = MLPMultiBinaryActor
        else:
            #actor_class = MLPOneHotCategoricalActor
            actor_class = MLPMultiCategoricalActor #MLPCategoricalActor #

        self.actor = actor_class(num_states, actor_width, num_actions, action_split, activation_func, d2rl).to(DEVICE)
        self.critic = MLPValueCritic(num_input_critic, critic_width, activation_func, d2rl).to(DEVICE)

        # Optimizer:
        if optimizer == None:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            self.actor_optim = optimizer(self.actor.parameters(), lr=actor_lr)
            self.critic_optim = optimizer(self.critic.parameters(), lr=critic_lr)

    def reset(self):
        pass

    def act(self, state, explore=True):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(DEVICE)
            action = self.actor.act(state, explore=explore).to("cpu").numpy()
        return action

    def estimate_value(self, state):
        # Get value:
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(DEVICE)
            value = self.critic(state).to("cpu").numpy()
        return value

    def get_log_prob_entropy(self, state, action):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(DEVICE)
            log_probs, entropy = self.actor.get_log_prob_entropy(state, action)
        return log_probs, entropy

    def update(self, batch):
        # Get Batch:
        rewards = batch.rewards
        actions = batch.actions
        actions_n = batch.actions_n
        obs = batch.observations
        obs_n = batch.observations_n
        dones = batch.dones
        returns = batch.returns
        advantages = batch.advantages
        values = batch.values

        # Extract info from batch:
        batch_size = rewards.shape[0]
        minibatch_size = batch_size // self.n_minibatches

        # Get initial log probs and entropy of the policy that generated the samples:
        with torch.no_grad():
            old_log_probs, old_entropy = self.actor.get_log_prob_entropy(obs, actions)
        
        # Training Loop:
        for epoch in range(self.n_epochs):
            idxs = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                # Generate minibatch indexes:
                end = start + minibatch_size
                mb_idxs = idxs[start:end]  # minibatch indexes
                # Get the updated values for the minibatch and the updated log prob
                mb_state_critic = obs[mb_idxs] if self.local_critic else obs_n[mb_idxs]
                new_values = self.critic(mb_state_critic)
                new_log_probs, new_entropy = self.actor.get_log_prob_entropy(obs[mb_idxs], actions[mb_idxs])
                # Calculate the importance sampling ratio:
                log_ratio = new_log_probs - old_log_probs[0,mb_idxs]
                ratio = log_ratio.exp()

                # Calculate approximate KL-Divergence
                with torch.no_grad():
                    approx_kl = self._calculate_approx_kl(ratio, log_ratio)

                ##### Critic loss #####
                mb_returns = returns[mb_idxs]
                critic_error = new_values - mb_returns
                v_loss = critic_error.pow(2)
                if self.clip_value:
                    v_clipped = values[mb_idxs] + torch.clamp(new_values - values[mb_idxs], -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss, v_loss_clipped)
                    critic_loss = 0.5 * v_loss_max.mean() * self.v_coef
                else:
                    critic_loss = 0.5 * v_loss.mean() * self.v_coef

                # Update Critic:
                self.critic_optim.zero_grad()
                critic_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                # Actor Loss:
                # Get advantages:
                mb_advantages = advantages[mb_idxs]
                if self.normalize_adv:
                    mb_advantages = standardize_value(mb_advantages)

                # PPO Policy Gradient Loss:
                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                entropy_loss = new_entropy.nanmean()
                actor_loss = pg_loss - self.entropy_coef * entropy_loss
                assert not actor_loss.isnan().any(), "NaN detected"

                # update actor:
                self.actor_optim.zero_grad()
                actor_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

        return (critic_loss.detach().to("cpu").numpy(),actor_loss.detach().to("cpu").numpy())

    def get_model(self):
        return {"params": self.dict_args, "state_dict": self.actor.state_dict()}

    @staticmethod
    def _calculate_approx_kl(ratio, logratio):
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        return approx_kl

    def update_lr(self, fraction):
        self.actor_optim.param_groups[0]["lr"] = fraction * self.actor_lr
        self.critic_optim.param_groups[0]["lr"] = fraction * self.critic_lr

    def load_actor_state(self, state_dict):
        self.actor.load_state_dict(state_dict)
