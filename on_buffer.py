from collections import OrderedDict, namedtuple, deque
import torch
import numpy as np
import random

Transition = namedtuple("Transition",("observations","observations_n","actions","actions_n","rewards","next_observations",
        "next_observations_n","dones","values","returns","advantages"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class OnPolicyBuffer:
    def __init__(self,gamma=0.9,gae=False,gae_lmb=0.5,standardize_rewards=False,shuffle=False):
        
        self.gamma = gamma
        self.gae = gae
        self.gae_lmb = gae_lmb
        self.standardize_rewards = standardize_rewards
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.observations = []
        self.observations_n = []
        self.actions = []
        self.actions_n = []
        self.rewards = []
        self.next_observations = []
        self.next_observations_n = []
        self.dones = []
        self.values = []

    def push(self,observation,observation_n,action,action_n,reward,next_observation,next_observation_n,done,value):
        
        self.observations.append(observation)
        self.observations_n.append(np.concatenate(observation_n))
        self.actions.append(action)
        self.actions_n.append(np.concatenate(action_n))
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.next_observations_n.append(np.concatenate(next_observation_n))
        self.dones.append(done)
        self.values.append(value)

    def sample(self, next_value):
        next_value = torch.tensor(next_value, requires_grad=False, dtype=torch.float, device=DEVICE)
        # Get all info available in the buffer:
        observations = self._create_tensor(np.array(self.observations), 2)
        observations_n = self._create_tensor(np.array(self.observations_n), 2)
        actions = self._create_tensor(np.array(self.actions), 2)
        actions_n = self._create_tensor(np.array(self.actions_n), 2)
        # Standardize rewards? :
        if self.standardize_rewards:
            rewards = self._create_tensor(self._standardize_value(self.rewards), 2)
        else:
            rewards = self._create_tensor(np.array(self.rewards), 2)
        next_observations = self._create_tensor(np.array(self.next_observations), 2)
        next_observations_n = self._create_tensor(np.array(self.next_observations_n), 2)
        dones = self._create_tensor(np.array(self.dones), 2)
        values = self._create_tensor(np.array(self.values), 2)

        # Calculate the returns and advantages:
        values_cat = torch.cat([values.flatten(), next_value.flatten()])
        with torch.no_grad():
            if self.gae:
                advantages = self.calculate_advantages(values_cat, rewards.flatten(), dones.flatten())
                returns = advantages + values.flatten()
            else:
                returns = self.calculate_returns(values_cat, rewards.flatten(), dones.flatten())
                advantages = returns - values.flatten()
            returns, advantages = returns.unsqueeze(-1), advantages.unsqueeze(-1)

        # Create full batch:
        batch = Transition(observations,observations_n,actions,actions_n,rewards,next_observations,next_observations_n,
                           dones,values,returns,advantages)

        # Check if needs to shuffle and shuffle:
        if self.shuffle:
            batch_size = rewards.shape[0]
            idxs = torch.randperm(batch_size)
            batch = Transition(*self._shuffle_tensors(batch, idxs))
        return batch

    def __len__(self):
        return len(self.rewards)

    def calculate_returns(self, values, rewards, dones):
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1] + self.gamma * (1 - dones[-1]) * values[-1]
        for step in reversed(range(len(returns) - 1)):
            returns[step] = (rewards[step] + self.gamma * (1 - dones[step]) * returns[step + 1])
        return returns

    def calculate_advantages(self, values, rewards, dones):
        td_errors = rewards + self.gamma * (1 - dones) * values[1:] - values[:-1]
        adv = torch.zeros_like(td_errors)
        adv[-1] = td_errors[-1]
        for step in reversed(range(len(td_errors) - 1)):
            adv[step] = td_errors[step] + self.gamma * self.gae_lmb * adv[step + 1] * (1 - dones[step])
        return adv

    @staticmethod
    def _create_tensor(a, ensure_dim):
        tensor = torch.tensor(a, requires_grad=False, dtype=torch.float, device=DEVICE)
        if len(tensor.size()) < ensure_dim:
            tensor = tensor.unsqueeze(-1)
        return tensor

    @staticmethod
    def _shuffle_tensors(list_of_tensors, idxs):
        output = [tensor[idxs] for tensor in list_of_tensors]
        if len(output) == 1:
            output = output[0]
        return output

    @staticmethod
    def _standardize_value(vector):
        vec = np.array(vector)
        return (vec - vec.mean()) / vec.std()

    @staticmethod
    def concatenate_batches(batches):
        return Transition(*(torch.cat(list(zipped)) for zipped in zip(*batches)))
