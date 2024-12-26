import gym.spaces as spaces
import numpy as np
from gym.utils import seeding

class BaseStation03:
    def __init__(self,n_ues,n_channels,hist_msg,hist_obs,n_voc_ul,n_actions):
        self.n_ues = n_ues
        self.n_channels = n_channels
        self.hist_msg = hist_msg
        self.hist_obs = hist_obs
        self.n_voc_ul = n_voc_ul
        self.n_actions = n_actions
        self.seed()

    def reset(self):
        self.env_action = 0
        self.comm_action = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.default_rng(seed)
        return [seed]

    def act(self, agent_state):
        return np.random.choice([item for item in range(0,self.n_actions)],self.n_channels,replace=True)

    def learn(self, state, action, reward, next_state):
        pass
