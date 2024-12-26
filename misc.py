import gym
import numpy as np
import torch


def soft_update(model, target_model, tau):
    with torch.no_grad():
        # update target networks
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)


# def relaxed_to_one_hot(in_list):
#     out_list = []
#     for vector in in_list:
#         ar = np.zeros(vector.shape)
#         ar[np.argmax(vector)] = 1
#         out_list.append(ar)
#     return out_list


def get_action_split(space):
    if type(space) in [gym.spaces.Discrete, gym.spaces.MultiBinary]:
        action_split = [space.n]
    elif type(space) == gym.spaces.MultiDiscrete:
        action_split = space.nvec.tolist()
    elif type(space) == gym.spaces.Tuple:
        action_split = len(space)
    return action_split


def get_total_actions(space_n):
    total = 0
    for space in space_n:
        if isinstance(space, gym.spaces.Box):
            total += space.shape[0]
        elif isinstance(space, gym.spaces.Discrete):
            total += space.n
        elif isinstance(space, gym.spaces.MultiBinary):
            total += space.n
        elif isinstance(space, gym.spaces.MultiDiscrete):
            total += space.nvec.sum()
        else:
            raise RuntimeError("Unknown space type. Can't return shape.")
    return total


def get_space_dimension(space):
    if isinstance(space, gym.spaces.Box):
        dim = space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        dim = space.n
    elif isinstance(space, gym.spaces.MultiBinary):
        dim = space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        dim = space.nvec.sum()
    else:
        raise RuntimeError("Unknown space type. Can't return shape.")
    return dim


def standardize_value(vector):
    return (vector - vector.mean()) / (vector.std() + 1e-6)
# 1e-6 is added to avoid a floating point error because sometimes std is equal to zero and it gives nan. 

def process_oh_actions(env, actions):
    corrected_actions = []
    for action_space, action in zip(env.action_space, actions):
        # Get the number length of each action:
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            # Get the number of env actions and comm actions:
            sizes = action_space.nvec.tolist()
        # MultiBinary for the BS: [n_ues, voc_size_dl]:
        elif isinstance(action_space, gym.spaces.MultiBinary):
            sizes = [action_space.n[1]] * action_space.n[0]
        # Tuple of MultiBinary or Discrete for the UEs:
        # (MultiBinary(nActions), MultiBinary(voc_size_ul))
        elif isinstance(action_space, gym.spaces.Tuple):
            sizes = [a_sp.n for a_sp in action_space]
        elif isinstance(action_space, gym.spaces.Discrete):
            sizes = [action_space.n]
        # # First we need to pick the indexes to split the action:
        split_idxs = np.array([sum(sizes[:ii]) for ii in range(1, len(sizes))])
        # # Then we split:
        ag_actions = np.split(action, split_idxs)
        # Transform Relaxed categorical to one hot encoding:
        # oh_actions = relaxed_to_one_hot(ag_actions)
        ls_actions = [np.argmax(ag_a) for ag_a in ag_actions]
        if len(ls_actions) == 1:
            ls_actions = ls_actions[0]
        corrected_actions.append(ls_actions)
    return corrected_actions
