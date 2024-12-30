# The objective function is to achieve an URLLC to the IIoT MDs. 
# The selected channels state history is added to each user state space as one-hot encoding.
# The first task parameters in the queue is added to each user state space. (excluded for now)
# The users that successfully computed their computation tasks is added to the BS state space.

from collections import deque
import numpy as np
import yaml
from gym import Env, spaces
from gym.utils import seeding
from envs.core_v1 import BS, UE
from envs.utils import get_len_space, relaxed_to_one_hot, to_categorical


class MECSCHEnvV1(Env):
    """ Observation space of each user is the status of the 
        tx_buffer: the number of computation tasks in the buffer.
        ch_buffer: the status of the selected channel. --> free/idle or successful transmission or collision as one-hot encoding.
        task_buffer: the parameters of the first task in the queue. (excluded for now)
        BS observation at each time step is the channels status --> no. of channels * channel status which is no. of UEs + 2 
        [ue_1,ue_2,.......,ue_n,free,collision,\ # ch_1 status
         ue_1,ue_2,.......,ue_n,free,collision,\ # ch_2 status
         ue_1,ue_2,.......,ue_n,free,collision,\ # ch_3 status
         ect.] 
        and the users that successfully computed their computation tasks within the delay deadline as one-hot encoding [no.of UEs,no.of UEs]

        *We consider three cases: 
        1- combined action mode (local and remote). 
        2- local computation mode only. 
        3- remote computation mode only.
        ## Combined action mode (local and remote):
        Environment action space of the MDs: [offloading decision,channel selection] 
           0 --> [0,0] Local computation and no channel access. 
           1 --> [1,1] Remote computation and channel 1 selected.
           2 --> [1,2] Remote computation and channel 2 selected.
           ect.
           no. channels + 1 --> Do nothing in case the buffer is empty or no resources.
        ## Local mode only:
           0 --> Local computation and no channel access. 
        ## Remote mode only:
           1 --> Remote computation and channel 1 selected.
           2 --> Remote computation and channel 2 selected.
           ect.
           no. channels  --> Do nothing in case the buffer is empty or no resources.
        * Communication action space is given by the vocabulary size of the uplink and downlink.
     """

    def __init__(self,n_ues=3,n_channels=2,UE_buffer_capacity=30,UE_CPU=1e9,BS_CPU=100e9,BS_BW=20e6,max_iters=30,n_voc_ul=2,n_voc_dl=4, 
                      hist_msg=3,hist_obs=3,hist_comm=3,hist_act=3,hist_chs=3,hist_tasks=3,arrival_prob=0.9,reward_com=1,penality=-1,
                      write_env=False,silent=False,counter=False,combined_mode=False,local_mode=False,remote_mode=True,NOMA_Scheme=True,
                      OMA_Scheme=False):                   
        
        super().__init__()
        self.seed()
        # Initialize Parameters
        # Locations of BS and mobile users in a single cell area 100*100
        self.UE_Locations = np.random.rand(n_ues,2)*300
        self.BS_Location = np.array([150,150])
        self.UE_Power = 0.08          
        self.task_size_min = 1e2
        self.task_size_max = 5e2
        self.task_cpu_min = 1e2
        self.task_cpu_max = 5e4
        self.task_delay_min = 1e-3
        self.task_delay_max = 3e-3
        self.uplink_th = np.random.uniform(15e6,30e6,n_ues) # np.linspace(5e6,20e6,n_ues)
        self.UE_power_levels = np.arange(0.01,0.1,0.02)
        # Save Parameters:
        self.BS_BW = BS_BW     # Base station channel bandwidth.        
        self.Noise_Power = np.power(10,(-174 + 10*np.log10(self.BS_BW))/10)/1000   # Noise power at the base station.        
        self.n_ues = n_ues  
        self.n_channels = n_channels
        self.Caching_Resources = np.random.uniform(5e2,5e2,self.n_channels) # np.linspace(3e2,6e2,self.n_channels) 
        self.Computing_Resources=  np.random.uniform(20e9,20e9,self.n_channels)  # np.linspace(50e9,100e9,self.n_channels)
        self.UE_buffer_capacity = UE_buffer_capacity
        self.UE_CPU = UE_CPU
        self.BS_CPU = BS_CPU
        self.arrival_prob = arrival_prob 
        self.reward_com = reward_com
        self.penality = penality
        self.max_iters = max_iters
        self.hist_msg = hist_msg if not silent else 0
        self.hist_obs = hist_obs
        self.hist_comm = hist_comm if not silent else 0
        self.hist_act = hist_act
        self.hist_chs = hist_chs
        self.hist_tasks = hist_tasks
        self.n_voc_dl = n_voc_dl if not silent else 0
        self.n_voc_ul = n_voc_ul if not silent else 0
        self.silent = silent
        self.write_env = write_env
        self.counter = counter
        self.combined_mode = combined_mode  
        self.local_mode = local_mode
        self.remote_mode = remote_mode
        self.NOMA_Scheme = NOMA_Scheme
        self.OMA_Scheme = OMA_Scheme
        # Environment action space available to the UEs [0, 1, 2, 3, 4,....,n_channels,Do_Nothing]
        if self.combined_mode:
            self.nA = self.n_channels + 2
        elif self.local_mode:
            self.nA =  1
        elif self.remote_mode:
            self.nA = self.n_channels  + 1
        
        # Initialize entities: adds self.ues, self.bs, self.entities
        entity_params = {"hist_act": self.hist_act,"hist_obs": self.hist_obs,"hist_msg": self.hist_msg,"hist_comm": self.hist_comm,
                         "n_voc_ul": self.n_voc_ul,"n_voc_dl": self.n_voc_dl,"silent": self.silent,"counter": self.counter}
        
        self.bs = BS(idx=0,n_ues=n_ues,n_channels=n_channels,**entity_params)
        self.ues = [UE(idx=ii,n_actions=self.nA,hist_chs=hist_chs,hist_tasks=hist_tasks,n_channels=n_channels,**entity_params) for ii in range(n_ues)]
        
        # Entities: [(bs, ue_1, ue_2,...)]
        self.entities = [self.bs,*self.ues] 
        self.agents = [a for a in self.entities if a.is_agent == True]
        # Number of agents:
        self.n_agents = len(self.agents)

        # Initialize channel matrix:
        self.Channel_Matrix = np.zeros((self.n_ues,self.n_channels),dtype=np.float32)
        self.Channel_Gain = np.zeros((self.n_ues,self.n_channels),dtype=np.float32)
        for n in range(self.n_ues):
            for m in range(self.n_channels):
                Path_loss = 128.1 + 37.6*np.log10(np.linalg.norm(self.BS_Location - self.UE_Locations[n,:])*1e-3)
                self.Channel_Matrix[n,m] = np.sqrt(np.power(10,-Path_loss/10)/2)*((np.random.randn(1) + 1.j*np.random.randn(1)).item())
                self.Channel_Gain[n,m] = np.linalg.norm(self.Channel_Matrix[n,m])**2

        # Define spaces:
        acsp = []
        obsp = []
        for ag in self.agents:
            if isinstance(ag, UE):
                # Action space is Multi Discrete: (Env_Actions, Comm_Actions) 
                if ag.silent:
                    acsp.append(spaces.MultiDiscrete([self.nA]))
                else:
                    acsp.append(spaces.MultiDiscrete([self.nA, n_voc_ul]))
            elif isinstance(ag, BS):
                acsp.append(spaces.MultiDiscrete([n_voc_dl] * n_ues))
            # Joint state space (bs, ue_1, ..., ue_n)
            obsp.append(ag.observation_space)
        self.action_space = spaces.Tuple(acsp)
        self.observation_space = spaces.Tuple(obsp)

    def seed(self, seed=None):
        if seed:
            seed = int(seed)
        self.np_random, self.seed_ = seeding.np_random(seed)
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):        
        # Initialize buffers:
        self.tx_buffer = self._init_buffer()
        self.rx_buffer = np.zeros((self.n_ues,self.max_iters))
        # Initialize parameters:
        self.iter = 0    
        self.reward = 0
        self.failed = 0
        self.n_successtasks = 0
        self.age_info = 0
        task_parm = np.zeros(3, dtype=np.float32)
        self.channels_access = np.zeros((self.n_channels,self.n_ues),dtype=np.int8)
        self.channels_state = np.zeros((self.n_channels,self.n_ues + 2),dtype=np.int8)
        self.ch_state_ues = np.zeros((self.n_ues,3),dtype=np.int8) 
        self.succ_comp = np.zeros((self.n_ues,self.n_ues),dtype=np.int8) 
        self.last_packets = np.zeros((self.n_ues,4))
        self.succ_tasks = np.zeros(self.n_ues,dtype=np.int8)
        # Reset and get state of entities:
        obs = []
        self._obs_render = []
        for ag in self.entities:
            ag.reset()
            if ag.is_actor:
                ag.update_obs(obs=len(self.tx_buffer[ag.idx]),dl_msg=None,ch_state=self.ch_state_ues[ag.idx,:],task_parm=task_parm,\
                              ch_gain=self.Channel_Gain[ag.idx,:])
            else:
                ag.update_obs(obs=self.channels_state,ul_msgs=None,succ_comp=self.succ_comp)
            if ag.is_agent:
                obs.append(ag.get_state())
            self._obs_render.append(ag.write_obs())
        return obs

    def step(self,actions_n):   
        self.iter += 1   
        self.reward = 0
        self.failed = 0
        self.n_successtasks = 0  
        self.sucess_rate = 0
        self.age_info = 0
        self.avg_delay = []
        self.goodput = []
        self.channels_access = np.zeros((self.n_channels,self.n_ues),dtype=np.int8)
        self.channels_state = np.zeros((self.n_channels,self.n_ues+2),dtype=np.int8)        
        self.delay = np.zeros(self.n_ues,dtype=np.float32) 
        self.ch_state_ues = np.zeros((self.n_ues,3),dtype=np.int8) 
        self.succ_comp = np.zeros((self.n_ues,self.n_ues),dtype=np.int8)         
        # Check action dimensions:
        self._check_action_input(actions_n)
        # Environment Actions:
        env_actions = np.zeros(self.n_ues,dtype=np.int8)
        # process action:
        for ii, ag in enumerate(self.agents):
            # Pass the actions to each entity:
            self._process_action(ag, actions_n[ii], self.action_space[ii])
            # Get all environment actions:
            if ag.is_actor:
                # Action is received in the Discrete form: 0,1,2,3,4
                env_actions[ag.idx] = ag.get_action()
        # get previous observation:
        self._obs_render = []
        for ii, ag in enumerate(self.entities):
            # Save current observation to render later:
            self._obs_render.append(ag.write_obs())
            
        # Actions in the right interval:
        assert np.logical_and(env_actions >= 0, env_actions < self.nA).all(), "Actions out of range"
        
        # Store the performance metrics
        info = {"No. of Success Tasks":0,"Channel Access Success Rate":0,"Channel Access Collision Rate":0,"Channel Idle Rate": 0,\
                "Packets Drop Rate":0,"Reward":0,"Age of Info":0,"Goodput":0,"Avg. Delay":0,"No. of Failed Tasks":0}
        
        # Get channels status      
        if self.combined_mode:
            for n in range(self.n_ues):
                tx_action = self._action_mapping(env_actions[n])
                if tx_action[0] == 1 and self.non_empty_tx_buffers[n]:
                    self.channels_access[tx_action[1],n] = 1      
            for ch in range(self.n_channels):            
                if np.sum(self.channels_access[ch,:]) == 0:
                    self.channels_state[ch,-2] = 1    # Idel channel
                elif np.sum(self.channels_access[ch,:]) == 1 or np.sum(self.channels_access[ch,:]) == 2 :
                    self.sucess_rate += 1
                    self.channels_state[ch,np.where(self.channels_access[ch,:] == 1)[0]] = 1    # success channel transmission
                elif np.sum(self.channels_access[ch,:]) > 2:
                    self.channels_state[ch,-1] = 1    # collision channel
        elif self.remote_mode:
             for ch in range(self.n_channels):  
                # number of transmission made on the channel at this time step:
                n_txs = (np.isin(env_actions, [ch]) * self.non_empty_tx_buffers).sum()
                if n_txs == 0:
                    self.channels_state[ch,-2] = 1
                elif n_txs == 1 or n_txs == 2:
                    self.sucess_rate += 1
                    self.channels_state[ch,np.where(env_actions == ch)[0]] = 1
                elif n_txs > 2:
                    self.channels_state[ch,-1] = 1 

        No_SCH = np.count_nonzero(np.array(self.channels_state[:,0:self.n_ues]) == 1) # No. of scheduled users.
        info["Channel Access Success Rate"] = self.sucess_rate/self.n_channels
        info["Channel Access Collision Rate"] = np.count_nonzero(np.array(self.channels_state[:,-1]) == 1)/self.n_channels
        info["Channel Idle Rate"] = np.count_nonzero(np.array(self.channels_state[:,-2]) == 1)/self.n_channels
        
        # Combined mode local computation + remote computation.
        if self.combined_mode:
            # Environment Step: take action:
            for ue, action in enumerate(env_actions):
                if action in np.arange(0,self.n_channels+1) and len(self.tx_buffer[ue]) > 0:     
                    action_ue = self._action_mapping(env_actions[ue])
                    offloading_action = action_ue[0]
                    up_channel_action = action_ue[1]           
                    if offloading_action == 0: # Local computation mode. 
                        self.ch_state_ues[ue,0] = 1
                        task_popped = self.tx_buffer[ue][-1] 
                        Local_delay = (1 - offloading_action)*task_popped[0]*task_popped[1]/self.UE_CPU
                        self.delay[ue] = task_popped[2] - Local_delay
                        if self.delay[ue] >= 0:
                            self.n_successtasks += 1
                            self.reward += self.reward_com
                            self.succ_comp[ue,ue] = 1
                            if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 1:
                                self.age_info += 1
                            self.tx_buffer[ue].pop()                                                             
                        elif self.delay[ue] < 0:
                            self.avg_delay.append(-1*self.dela[ue])
                            self.reward += self.penality
                            if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                                self.tx_buffer[ue].pop() 
                                info["Packets Drop Rate"] += 1
                    else: # Remote computation mode.
                        if 1 <= np.sum(self.channels_state[up_channel_action,0:self.n_ues]) <= 2:
                            self.ch_state_ues[ue,1] = 1
                            task_popped = self.tx_buffer[ue][-1]          
                            # Remote computation delay.
                            if np.sum(self.channels_state[up_channel_action,0:self.n_ues]) == 1:
                                transmit_delay = task_popped[0]/self._uplink_rate_OMA(ue,up_channel_action)
                                uplink_rate = self._uplink_rate_OMA(ue,up_channel_action)
                            elif np.sum(self.channels_state[up_channel_action,0:self.n_ues]) == 2:
                                ues = np.where(self.channels_state[action,0:self.n_ues] == 1)[0]
                                ind = np.where(ues == ue)[0]  
                                transmit_delay = task_popped[0]/self._uplink_rate_NOMA(ues,ind,up_channel_action)
                                uplink_rate = self._uplink_rate_NOMA(ues,ind,up_channel_action)
                            BS_comp_delay = offloading_action*task_popped[0]*task_popped[1]/(self.BS_CPU/No_SCH)
                            remote_delay =  BS_comp_delay + transmit_delay
                            self.delay[ue] = task_popped[2] - remote_delay
                            if self.delay[ue] >= 0 and uplink_rate >= self.uplink_th:
                                self.n_successtasks += 1
                                self.reward += self.reward_com
                                self.succ_comp[ue,ue] = 1
                                self.goodput.append(1)
                                self.succ_tasks[ue] += 1
                                if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 1:
                                    self.age_info += 1
                                self.tx_buffer[ue].pop()                                   
                            elif self.delay[ue] < 0 or uplink_rate < self.uplink_th:
                                self.avg_delay.append(-1*self.delay[ue])
                                self.reward += self.penality
                                if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                                    self.tx_buffer[ue].pop() 
                                    info["Packets Drop Rate"] += 1
                        elif self.channels_state[up_channel_action,-1] == 1:  # Collision Channel  
                            task_popped = self.tx_buffer[ue][-1]
                            self.ch_state_ues[ue,2] = 1
                            self.delay_reward[ue] = -task_popped[2]
                            self.avg_delay.append(-1*self.delay[ue])
                            self.reward += self.penality
                            if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                                self.tx_buffer[ue].pop() 
                                info["Packets Drop Rate"] += 1
        # Local computation mode
        elif self.local_mode:
            # Environment Step: take action:
            for ue, action in enumerate(env_actions):
                if action == 0 and len(self.tx_buffer[ue]) > 0:
                    self.ch_state_ues[ue,0] = 1
                    task_popped = self.tx_buffer[ue][-1]
                    Local_delay = task_popped[0]*task_popped[1]/self.UE_CPU
                    self.delay[ue] = task_popped[2] - Local_delay
                    if self.delay[ue] >= 0:
                        self.n_successtasks += 1
                        self.reward += self.reward_com
                        self.succ_comp[ue,ue] = 1
                        if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 1:
                            self.age_info += 1
                        self.tx_buffer[ue].pop()                             
                    elif self.delay[ue] < 0:
                        self.avg_delay.append(-1*self.delay[ue])
                        self.reward += self.penality
                        if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                            self.tx_buffer[ue].pop() 
                            info["Packets Drop Rate"] += 1                            
        # Remote computation mode        
        elif self.remote_mode:
            # Environment Step: take action:
            for ue, action in enumerate(env_actions):
                if action in np.arange(0,self.n_channels) and len(self.tx_buffer[ue]) > 0:  
                    if np.sum(self.channels_state[action,0:self.n_ues]) == 1 or np.sum(self.channels_state[action,0:self.n_ues]) == 2:       
                        self.ch_state_ues[ue,1] = 1
                        task_popped = self.tx_buffer[ue][-1]  
                        # Remote computation delay.
                        if np.sum(self.channels_state[action,0:self.n_ues]) == 1:
                            transmit_delay = task_popped[0]/self._uplink_rate_OMA(ue,action)
                            uplink_rate = self._uplink_rate_OMA(ue,action)
                        elif np.sum(self.channels_state[action,0:self.n_ues]) == 2:
                            ues = np.where(self.channels_state[action,0:self.n_ues] == 1)[0]
                            ind = np.where(ues == ue)[0]                   
                            transmit_delay = task_popped[0]/self._uplink_rate_NOMA(ues,ind,action)  
                            uplink_rate = self._uplink_rate_NOMA(ues,ind,action) 
                        BS_comp_delay = task_popped[0]*task_popped[1]/self.Computing_Resources[action] #(self.BS_CPU/No_SCH)
                        remote_delay =  BS_comp_delay + transmit_delay
                        self.delay[ue] = task_popped[2] - remote_delay 
                        self.storage = True if self.Caching_Resources[action] >= task_popped[0] else False
                        #if self.delay[ue] >= 0 and uplink_rate >= self.uplink_th:
                        if self.delay[ue] >= 0 and uplink_rate >= self.uplink_th[ue] and self.storage: 
                            self.n_successtasks += 1
                            self.reward += self.reward_com
                            self.succ_comp[ue,ue] = 1
                            self.goodput.append(1)
                            self.succ_tasks[ue] += 1
                            #if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 1:
                            self.age_info += 1
                            self.tx_buffer[ue].pop()                        
                        #elif self.delay[ue] < 0 or uplink_rate < self.uplink_th:
                        elif self.delay[ue] < 0 or uplink_rate < self.uplink_th[ue] or self.storage == False:
                            self.avg_delay.append(-1*self.delay[ue])
                            self.reward += self.penality
                            self.failed += 1
                            #if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                            #self.tx_buffer[ue].pop()
                            info["Packets Drop Rate"] += 1
                    elif self.channels_state[action,-1] == 1:  # Collision Channel  
                        task_popped = self.tx_buffer[ue][-1]
                        self.ch_state_ues[ue,2] = 1
                        self.delay[ue] = - task_popped[2]
                        self.avg_delay.append(-1*self.delay[ue])
                        self.reward += self.penality
                        self.failed += 1
                        #if (self.last_packets[ue,0:3] == self.tx_buffer[ue][-1]).all() and self.last_packets[ue,3] == self.iter - 2:
                        #self.tx_buffer[ue].pop()
                        info["Packets Drop Rate"] += 1
                            
                           
        if not len(self.avg_delay) == 0:         
            info["Avg. Delay"] += np.mean(self.avg_delay) 
        if not len(self.goodput) == 0:    
            info["Goodput"] += np.sum(self.goodput)
        info["No. of Success Tasks"] += self.n_successtasks/(2*self.n_channels) 
        info["Reward"] += self.reward
        info["Age of Info"] += self.age_info
        info["No. of Failed Tasks"] += self.failed/(2*self.n_channels)  
        
        # Transit:
        # Maybe generate traffic:
        self._gen_traffic()
        # Update observations of all entities:
        ul_msgs = [ue.comm_action for ue in self.ues] if not self.silent else None
        dl_msgs = self.bs.comm_action if not self.silent else [None] * self.n_ues
        for ii, ag in enumerate(self.entities):
            if isinstance(ag, UE):
                if len(self.tx_buffer[ag.idx]) > 0 :
                    task_parm = self.tx_buffer[ag.idx][-1]                
                    if (self.last_packets[ag.idx,0:3] != self.tx_buffer[ag.idx][-1]).all():                   
                        self.last_packets[ag.idx,0:3] = self.tx_buffer[ag.idx][-1]
                        self.last_packets[ag.idx,3] = self.iter
                else:
                    task_parm = np.zeros(3, dtype=np.float32)
                    self.last_packets[ag.idx,0:3] = np.zeros(3, dtype=np.float32)
                    self.last_packets[ag.idx,3] = self.iter
                ag.update_obs(obs=len(self.tx_buffer[ag.idx]),dl_msg=dl_msgs[ag.idx],ch_state=self.ch_state_ues[ag.idx,:],\
                              task_parm=task_parm,ch_gain=self.Channel_Gain[ag.idx,:])
            else:
                ag.update_obs(obs= self.channels_state,ul_msgs=ul_msgs,succ_comp=self.succ_comp)
                
        obs_buffer = []
        for agent in self.ues:
            obs_buffer.append(len(self.tx_buffer[agent.idx]))
        
        # Check if Final iteration:
        done = bool(self.iter > self.max_iters) or bool(obs_buffer == 0)
        dones = [done]*self.n_agents
        rewards = [self.reward]*self.n_agents 
        # Get next states
        obs = [ag.get_state() for ag in self.agents]
        return obs, rewards, dones, info


    @property
    def tx_buffer_sizes(self):
        return np.array([len(u) for u in self.tx_buffer.values()])

    @property
    def non_empty_tx_buffers(self):
        return self.tx_buffer_sizes > 0
    
    def render(self):
        # Environment Actions:
        env_actions = np.zeros(self.n_ues, dtype=np.int8)
        # Communication Actions:
        ul_msgs = np.zeros(self.n_ues, dtype=np.int8)
        # Obs:
        all_obs = self._obs_render
        # Get actions and messages:
        for ii, ag in enumerate(self.entities):
            # Get all environment actions:
            if ag.is_actor:
                # Action is received in the Discrete form: 0,1,2,3,4
                env_actions[ag.idx] = ag.get_action()
                ul_msgs[ag.idx] = ag.get_msg()
                # all_obs.append(ag.observation)
            else:
                dl_msgs = ag.get_msg()
                # all_obs.append(ag.write_obs())
        out = {f"O_bs": all_obs[0]}
        for ii in range(self.n_ues):
            out.update({f"O_{ii}": all_obs[ii + 1],f"AC_{ii}": self._transform_ac2str(env_actions[ii]),f"DL_{ii}": dl_msgs[ii],f"UL_{ii}":ul_msgs[ii]})
        return out

    def _gen_traffic(self):
        for u in range(self.n_ues):
            if self.rng.random() <= self.arrival_prob:
                if len(self.tx_buffer[u]) < self.tx_buffer[u].maxlen: 
                    # Add the tasks parameters: Task size, Required CPU, Task Deadline
                    self.tx_buffer[u].appendleft(np.array([np.random.uniform(self.task_size_min,self.task_size_max),
                                                           np.random.uniform(self.task_cpu_min,self.task_cpu_max),
                                                           np.random.uniform(self.task_delay_min,self.task_delay_max)]))

    def _init_buffer(self):
        """Initialize the buffers of the UEs:Each buffer is a deque and, if non-empty, it is of the form: [n, n-1, ..., 2, 1]"""        
        UE_buffer = {ue: deque(maxlen=self.UE_buffer_capacity) for ue in range(self.n_ues)} # Generate empty buffer:
        return UE_buffer


    def _process_action(self, agent, action, action_space):
        # Get the number length of each action:
        if isinstance(action_space, spaces.MultiDiscrete):
            # Get the number of env actions and comm actions:
            sizes = action_space.nvec.tolist()
        # MultiBinary for the BS: [n_ues, voc_size_dl]:
        elif isinstance(action_space, spaces.MultiBinary):
            sizes = [action_space.n[1]] * action_space.n[0]
        # Tuple of MultiBinary or Discrete for the UEs: (MultiBinary(nActions), MultiBinary(voc_size_ul))
        elif isinstance(action_space, spaces.Tuple):
            sizes = [a_sp.n for a_sp in action_space]
        # Separate communication action from env action:
        if self.discrete_input:
            oh_actions = [to_categorical(action[ii], sizes[ii]) for ii in range(len(sizes))]
        else:
            # First we need to pick the indexes to split the action:
            split_idxs = np.array([sum(sizes[:ii]) for ii in range(1, len(sizes))])
            # Then we split:
            ag_actions = np.split(action, split_idxs)
            # Transform Relaxed categorical to one hot encoding:
            oh_actions = relaxed_to_one_hot(ag_actions)
        # Add to the entity:
        agent.update_action(oh_actions)

    def _check_action_input(self, action_n):
        lens_in = [len(action) for action in action_n]
        lens_true_oh = [get_len_space(a_sp) for a_sp in self.action_space]
        lens_true = [len(a_sp) for a_sp in self.action_space]
        if lens_in == lens_true:
            self.discrete_input = True
        elif lens_in == lens_true_oh:
            self.discrete_input = False
        else:
            raise Exception(f"Wrong Action Input env.step(): received {lens_in},"+ f"but should be {lens_true} or {lens_true_oh}")


    def _uplink_rate_OMA(self,ue,selected_channel):  
        SNR = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ue,selected_channel])**2/(self.Noise_Power/self.n_channels) # SNR
        Up_Rate = (self.BS_BW/self.n_channels)*np.log10(1+SNR)     # Uplink rate. 
        return Up_Rate
    
    def _uplink_rate_NOMA(self,ues,ind,selected_channel):  
        SNR = np.zeros(2)
        Up_Rate = np.zeros(2)
        if np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2> np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2:
            SNR[0] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2/((self.Noise_Power/self.n_channels) + self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2) # SNR
            SNR[1] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2/(self.Noise_Power/self.n_channels) # SNR
        else: 
            SNR[1] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2/((self.Noise_Power/self.n_channels) + self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2) # SNR
            SNR[0] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2/(self.Noise_Power/self.n_channels) # SNR
        Up_Rate[0] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[0])     # Uplink rate. 
        Up_Rate[1] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[1])     # Uplink rate.
        return Up_Rate[ind]
    
    # Need to take joint action of offloading and channel selection.
    def _action_mapping(self,action):
        if self.combined_mode:
            if action == 0:
                map_action = np.array([0,0],dtype=np.int8)   # Local Computation.
            elif action in np.arange(1,self.n_channels+1):
                map_action = np.array([1,action-1],dtype=np.int8)  # Remote Computation.
            elif action == self.n_channels + 1:
                map_action = np.array([2,2],dtype=np.int8) # Do nothing.                    
        return map_action
