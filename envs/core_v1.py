# In this core version: I added the selected channel state to each user state space as one-hot encoding.

from collections import deque
from envs.utils import to_categorical  
import numpy as np
import gym

# properties and state of physical world entity
class UEObservation(object):
    def __init__(self):
        # Buffer Status:
        self.tx_buffer = None
        self.ch_buffer = None
        self.task_buffer = None
        
    def update(self, obs, ch_state, task_par):
        self.tx_buffer = obs
        self.ch_buffer = ch_state
        self.task_buffer = task_par
        
    def get(self):
        return self.tx_buffer,self.ch_buffer,self.task_buffer

    def write(self):
        return self.tx_buffer, self.ch_buffer,self.task_buffer


class BSObservation(object):
    def __init__(self):
        # BS observation is the states of channels --> one-hot encoding and users that their tasks successfully computed within the delay deadline --> one-hot encoding
        # [M,N+2] --> no. of channels * channel status which is no. of UEs + 2 
        # [ue_1,ue_2,.......,ue_n,free,collision,\ # ch_1 status
        #  ue_1,ue_2,.......,ue_n,free,collision,\ # ch_2 status
        #  ue_1,ue_2,.......,ue_n,free,collision,\ # ch_3 status
        #  ect.] 
        # The successful users computation --> [N,N] no. of UEs * no. of UEs
        self.channels_states = None 
        self.succ_comp = None
        
    def update(self, obs):
        self.channels_states = obs
        
    def get(self):
        return self.channels_states 

    def write(self):
        return self.channels_states


class Action(object):
    def __init__(self):
        self.env_action = None
        self.c_action = None

    def update_action(self,actions):
        self.env_action = actions[0]
        self.c_action = actions[1]


class Entity(object):
    def __init__(self,idx=0,name="",is_actor=False,silent=False,blind=False,c_noise=None):        
        # index among all entities (important to set for distance caching)
        self.idx = idx
        # name
        self.name = name
        # can the entity act:
        self.is_actor = is_actor
        # cannot send communication signals
        self.silent = silent
        # cannot observe the world
        self.blind = blind
        # communication noise amount
        self.c_noise = c_noise
        # action
        self.env_action = None
        # communication action
        self.comm_action = None

    @property
    def is_agent(self):
        if self.silent == True and self.is_actor == False:
            return False
        else:
            return True


class UE(Entity):        
    def __init__(self,idx=0,n_actions=0,hist_chs=0,hist_tasks=0,hist_act=0,hist_msg=0,hist_comm=0,hist_obs=0,n_voc_ul=0,n_voc_dl=0,
                      n_channels=0,silent=False,counter=False):  
        
        name = f"UE_{idx}"
        super(UE, self).__init__(name=name, idx=idx, is_actor=True, silent=silent)
        
        self.observation = None
        # Save Parameters:
        self.counter = counter
        # Buffer Lengths:
        self.hist_msg = hist_msg
        self.hist_obs = hist_obs
        self.hist_comm = hist_comm
        self.hist_act = hist_act
        self.hist_chs = hist_chs
        self.hist_tasks = hist_tasks
        self.n_channels = n_channels
        # Size of Actions:
        self.n_voc_ul = n_voc_ul
        self.n_voc_dl = n_voc_dl
        self.n_actions = n_actions
        # Dimensions of buffers:
        ac_dim = n_actions * hist_act
        rx_dim = n_voc_dl * hist_msg
        tx_dim = n_voc_ul * hist_comm
        ch_dim = 3 * hist_chs
        task_dim = 3 * hist_tasks
        obs_dim = 1 * hist_obs
        ch_mx_dim = hist_obs * self.n_channels * 1
        # Get spaces:
        self.state_dim = obs_dim + ac_dim + rx_dim + tx_dim + ch_dim # + task_dim + ch_mx_dim
        if self.counter:
            self.state_dim = self.state_dim + 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.state_dim,))

    def reset(self):
        """Reset the entity: Sets the values of the observations, actions and initial buffers."""
        # Generate fake previous obs and actions:
        self.observation = 0
        self.env_action = np.zeros(self.n_actions,dtype=np.int8)
        self.comm_action = np.zeros(self.n_voc_ul,dtype=np.int8)
        # Generate buffers
        # Observation buffer: 
        self.obs_buffer = deque([0] * self.hist_obs, maxlen=self.hist_obs)
        # Previous action buffer:
        self.ac_buffer = deque(np.zeros((self.hist_act, self.n_actions), dtype=np.int8),maxlen=self.hist_act)
        # Received msgs buffer (dl for UE, ul for BS):
        self.rx_buffer = deque(np.zeros((self.hist_msg, self.n_voc_dl), dtype=np.int8),maxlen=self.hist_msg)
        # Transmitted msgs buffer (ul for UE, dl for BS):
        self.tx_buffer = deque(np.zeros((self.hist_comm, self.n_voc_ul), dtype=np.int8),maxlen=self.hist_comm)
        # Selected Channel state buffer (ul channel for UE):
        self.ch_buffer = deque(np.zeros((self.hist_chs, 3), dtype=np.int8), maxlen=self.hist_chs)
        # Task parameters buffer:
        self.task_buffer = deque(np.zeros((self.hist_tasks, 3), dtype=np.float64), maxlen=self.hist_tasks)
        # Channel matrix buffer:
        self.ch_max_buffer = deque(np.zeros((self.hist_obs,1,self.n_channels),dtype=np.int8),maxlen=self.hist_obs)
        if self.counter:
            self.iter = 0

    def update_obs(self, obs, dl_msg=None, ch_state=None, task_parm=None,ch_gain=None):
        # Add the previous observation to buffer:
        self.obs_buffer.append(self.observation)
        if dl_msg is not None:
            self.rx_buffer.append(dl_msg)
        if ch_state is not None:
            self.ch_buffer.append(ch_state)
        if task_parm is not None:
            self.task_buffer.append(task_parm)
        if ch_gain is not None:
            self.ch_max_buffer.append(ch_gain)
        # Update:
        self.observation = obs

        
    def update_action(self, action):
        # Update history:
        self.ac_buffer.append(action[0])
        if not self.silent:
            self.tx_buffer.append(action[1])
        # Update Current action:
        self.env_action = action[0]
        if not self.silent:
            self.comm_action = action[1]
        if self.counter:
            self.iter += 1

    def get_state(self):
        # Current observation + previous ones in the buffer:
        state = np.concatenate((np.array(self.obs_buffer, dtype=np.int8).flatten(),
                                np.array(self.rx_buffer, dtype=np.int8).flatten(),
                                np.array(self.ac_buffer, dtype=np.int8).flatten(),
                                np.array(self.tx_buffer, dtype=np.int8).flatten(),
                                np.array(self.ch_buffer, dtype=np.float32).flatten()))
                                #np.array(self.task_buffer, dtype=np.int8).flatten()))
                                #np.array(self.ch_max_buffer, dtype=np.int8).flatten()
        
        if self.counter:
            state = np.concatenate((state, np.array([self.iter])))
        return state

    def get_action(self):
        return np.argmax(self.env_action)

    def get_msg(self):
        return np.argmax(self.comm_action, axis=1)

    def write_obs(self):
        return self.observation


class BS(Entity):
    def __init__(self,idx=0,n_ues=0,n_channels=0,hist_act=0,hist_msg=0,hist_comm=0,hist_obs=0,n_voc_ul=0,n_voc_dl=0,silent=False,\
                      counter=False):
        
        name = "BS"
        super(BS, self).__init__(name=name, idx=idx, silent=silent)
        self.n_ues = n_ues
        self.n_channels = n_channels
        self._observation = BSObservation()
        self.counter = counter
        # Parameters for the buffers:
        self.hist_msg = hist_msg   
        self.hist_obs = hist_obs
        self.hist_comm = hist_comm
        self.hist_act = hist_act
        # Parameters of the action/obs spaces:
        self.n_voc_ul = n_voc_ul if not silent else 0
        self.n_voc_dl = n_voc_dl if not silent else 0       
        # Obs: No. of channels * No. of UEs + 2
        self.obs_size = n_channels*(self.n_ues+2)
        # Get dimensions of the buffer (previous actions and obs stored):
        rx_dim = n_voc_ul * hist_msg * n_ues
        tx_dim = n_voc_dl * hist_comm * n_ues
        obs_dim = self.obs_size * hist_obs
        suss_dim = hist_obs * n_ues * n_ues
        # Generate Observation Space
        self.state_dim = obs_dim + rx_dim + tx_dim + suss_dim
        if self.counter:
            self.state_dim = self.state_dim + 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.state_dim,))

    @property
    def observation(self):
        return self._observation.get()

    @observation.setter
    def observation(self, obs):
        self._observation.update(obs)

    def reset(self):
        """Generate buffer to keep previous information."""
        # Initialize observation and actions:
        # Idle observation:
        self.observation = np.zeros((self.n_channels, (self.n_ues+2)))
        # Actions are zero:
        self.comm_action = np.zeros([self.n_ues, self.n_voc_dl])
        # Observation buffer:
        self.obs_buffer = deque(np.zeros((self.hist_obs, self.n_channels, (self.n_ues+2)), dtype=np.int8),maxlen=self.hist_obs)
        # Received msgs buffer (dl for UE, ul for BS):
        self.rx_buffer = deque(np.zeros((self.hist_msg, self.n_ues, self.n_voc_ul), dtype=np.int8),maxlen=self.hist_msg)
        # Transmitted msgs buffer (ul for UE, dl for BS):
        self.tx_buffer = deque(np.zeros((self.hist_comm, self.n_ues, self.n_voc_dl), dtype=np.int8),maxlen=self.hist_comm)
        # Successful computation buffer:
        self.succ_buffer = deque(np.zeros((self.hist_obs, self.n_ues,self.n_ues), dtype=np.int8),maxlen=self.hist_obs)
        if self.counter:
            self.iter = 0

    def update_obs(self, obs, ul_msgs=None, succ_comp=None):
        if ul_msgs is not None and not isinstance(ul_msgs, np.ndarray):
            ul_msgs = np.array(ul_msgs)
        if ul_msgs is not None and ul_msgs.shape != (self.n_ues, self.n_voc_ul):
            shape_1 = ul_msgs.shape
            shape_2 = (self.n_ues, self.n_voc_ul)
            raise ValueError(f"Wrong input shape: ul_msgs should be {shape_2}, but got {shape_1}")
        # Add the previous observation to buffer:
        self.obs_buffer.append(self.observation)
        # Add the successful computation to buffer:
        if succ_comp is not None:
            self.succ_buffer.append(succ_comp)
        # Add received msgs to buffer:
        if ul_msgs is not None:
            self.rx_buffer.append(ul_msgs)
        # Update:
        self.observation = obs

    def update_action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if action.shape != (self.n_ues, self.n_voc_dl):
            shape_1 = action.shape
            shape_2 = (self.n_ues, self.n_voc_dl)
            raise ValueError(f"Wrong input shape: action should be {shape_2}, but got {shape_1}")
        self.tx_buffer.append(action)
        # Update:
        self.comm_action = action
        if self.counter:
            self.iter += 1

    def get_state(self):
        state = np.concatenate((np.array(self.obs_buffer, dtype=np.int8).flatten(),
                                np.array(self.rx_buffer, dtype=np.int8).flatten(),                                
                                np.array(self.tx_buffer, dtype=np.int8).flatten(),
                                np.array(self.succ_buffer, dtype=np.int8).flatten()))

        if self.counter:
            state = np.concatenate((state, np.array([self.iter])))
        return state

    def get_msg(self):
        return np.argmax(self.comm_action, axis=1)

    def write_obs(self):
        return self._observation.write()

