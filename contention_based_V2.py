import gym.spaces as spaces
import numpy as np
from gym.utils import seeding

class BaseStation02:
    def __init__(self,n_ues,n_channels,hist_msg,hist_obs,n_voc_ul):
        self.n_ues = n_ues
        self.n_channels = n_channels
        self.hist_msg = hist_msg
        self.hist_obs = hist_obs
        self.n_voc_ul = n_voc_ul
        self.seed()

    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.default_rng(seed)
        return [seed]

    def act(self, agent_state):
        """ "
        This policy outputs one message for each of the N UEs in the cell.
        Each message can take any of the following values:
            0 : DoNotTransmit
            1 : Transmit on channel 1
            2 : Transmit on channel 2
              :
              :
              ect.
           no. of channels + 1 : ACK

        In time steps, wherein the BS concurrently receives one
        SchedulingRequest and one UL SDU from a given UE, it will always
        respond first with an ACK to the UE and ignore the SchedulingRequest.

        Input parameters:
            :param obs: A vector contains the observation of channels from the environment,
                        where each channel can take one of the following values:
                0: Idle Channel
                1: Tx from UE 0
                2: Tx from UE 1
                3: Tx from UE 2
                .
                .
                |U|+1: Collision
            :param msg:  An array with N integers corresponding to the
                         messages receive from each and all UEs. The
                         entries in msg take the following values:
                0:  NullMessage
                1:  SchedulingRequest
            :param success computation: An array with N integers corresponding to the successful computation of the users.
        """
        # Unpack State:
        obs = agent_state[((self.hist_obs-1)*(self.n_channels*(self.n_ues+2))):
                          (self.hist_obs*self.n_channels*(self.n_ues+2))].reshape(self.n_channels,(self.n_ues+2)).astype(int)
  
        ul_msgs = (agent_state[self.n_channels*(self.n_ues+2)*self.hist_obs + ((self.hist_msg-1)*(self.n_ues*self.n_voc_ul)): 
                               self.n_channels*(self.n_ues+2)*self.hist_obs + (self.hist_msg*self.n_ues*self.n_voc_ul)]
                               .reshape(self.n_ues,self.n_voc_ul).argmax(axis=1).astype(int))
        
        sucess_comp = agent_state[-self.n_ues*self.n_ues:].reshape(self.n_ues,self.n_ues)
        
        # Transform obs:
        obs_ch = np.zeros(self.n_channels, dtype=np.int64)
        for ch in range(self.n_channels): 
            if np.any(obs[ch,:] == 1):
                if obs[ch,-1] == 1:
                    obs_ch[ch] = self.n_ues + 1  # Collision
                elif obs[ch,-2] == 1:
                    obs_ch[ch] = 0  # Idle
                else:
                    obs_ch[ch] = np.argmax(obs[ch,:]) + 1  # Idx of UE that transmitted sucessfully
            else:
                obs_ch[ch] = 0
                        
        # Check ul_msgs:
        if not isinstance(ul_msgs, np.ndarray):
            ul_msgs = np.array(ul_msgs)
        assert ((0 <= ul_msgs) & (ul_msgs <= 1)).all(), "Uplink messages from UEs out of range"
        
        # Memory allocation:
        dl_msgs = np.zeros((self.n_ues,), dtype=np.uint8)
        # Users who are requesting:
        requesters = np.where(ul_msgs == 1)[0]
        if requesters.size > 0:
            # Choose a set of UEs randomly to transmit from among those requesters.
            ue = self.rng.choice(requesters,size=min(2*self.n_channels,len(requesters)),replace=True)
            dl_msgs[ue] = self.rng.choice(np.arange(1,self.n_channels+1),size=len(ue),replace=True)
        else:
            ue = None  # Index of UE to be granted transmit rights

        # Send ACK to the successful computation of UEs.
        dl_msgs[np.where(sucess_comp == 1)[0]] = self.n_channels + 1

        return dl_msgs

    def learn(self, state, action, reward, next_state):
        pass


""" This policy the UE transmits and sends scheduling request with a certain probability. 
    If it will transmit, it choose random channel and ignore the assignment from the base station.
"""
class UEAgent02:
    def __init__(self,n_channels,hist_msg,hist_obs,n_voc_dl):
        self.n_channels = n_channels
        self.hist_msg = hist_msg
        self.hist_obs = hist_obs
        self.n_voc_dl = n_voc_dl    
        self.trans_prob = 0.2
        pass

    def reset(self):
        self.env_action = 0
        self.comm_action = 0

    def seed(self, seed):
        pass

    def act(self, agent_state):
        
        buffer_status = agent_state[self.hist_obs - 1]
        dl_msg = np.argmax(agent_state[(((self.hist_msg - 1)*self.n_voc_dl) + self.hist_obs): 
                                       (((self.hist_msg)*self.n_voc_dl) + self.hist_obs)]) 

        old_action = self.env_action
        # Environment action:
        if  self.trans_prob > np.random.uniform(0,1) and buffer_status > 0:
            action = np.random.randint(0,self.n_channels) 
        else:
            action = self.n_channels 
        # Communication Action:
        if buffer_status > 0:
            if self.trans_prob > np.random.uniform(0,1):
                ul_msg = 1
            else:
                ul_msg = 0
        else:
            ul_msg = 0

        self.env_action = action
        self.comm_action = ul_msg

        return np.array([action, ul_msg])

    def learn(self, state, action, reward, next_state):
        pass

