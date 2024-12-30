from collections import deque
import numpy as np
import linecache
import math
import yaml
import sys
import pickle
import ast
from gym import Env, spaces
from gym.utils import seeding
from envs.core_v2 import BS, UE
from itertools import permutations,product,combinations,compress
from envs.utils import get_len_space,relaxed_to_one_hot,to_categorical
sys.getsizeof(float())

class MECSCHEnvV2(Env):    
    def __init__(self,n_ues=12,n_channels=3,UE_buffer_capacity=30,UE_CPU=1e9,BS_CPU=100e9,BS_BW=20e6,max_iters=30,n_voc_ul=2,n_voc_dl=4,     
                     hist_msg=3,hist_obs=3,hist_comm=3,hist_act=3,arrival_prob=0.9,reward_com=1,penality=-1,write_env=False,silent=False,
                     counter=False,NOMA_Scheme=True,OMA_Scheme=False,Reduction=False,Round_robin=False,heuristic=False,semi_static=False):     
                         
        super().__init__()
        self.seed()
        # Initialize Parameters
        # Locations of BS and mobile users in a single cell area 300*300
        self.UE_Locations = np.random.rand(n_ues,2)*300
        self.BS_Location = np.array([150,150]) 
        self.UE_Power = 0.08         
        self.task_size_min = 1e2
        self.task_size_max = 5e2
        self.task_cpu_min = 1e2
        self.task_cpu_max = 5e4
        self.task_delay_min = 1e-3
        self.task_delay_max = 3e-3
        self.uplink_th = np.random.uniform(10e6,30e6,n_ues) # np.linspace(5e6,20e6,n_ues) 
        self.BS_BW = BS_BW             
        self.Noise_Power = np.power(10,(-174 + 10*np.log10(self.BS_BW))/10)/1000   # Noise power at the base station.        
        self.n_ues = n_ues  
        self.n_channels = n_channels
        self.Caching_Resources =  np.random.uniform(5e2,5e2,self.n_channels) # np.linspace(3e2,6e2,self.n_channels)
        self.Computing_Resources=  np.random.uniform(20e9,20e9,self.n_channels) # np.linspace(50e9,100e9,self.n_channels)
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
        self.n_voc_dl = n_voc_dl if not silent else 0
        self.n_voc_ul = n_voc_ul if not silent else 0
        self.silent = silent
        self.write_env = write_env
        self.Reduction = Reduction
        self.Round_robin = Round_robin
        self.heuristic = heuristic 
        self.semi_static = semi_static
        self.counter = counter        
        self.NOMA_Scheme = NOMA_Scheme
        self.OMA_Scheme = OMA_Scheme
        
        # Set the action list.
        distance = np.zeros(self.n_ues)
        for n in range(self.n_ues):
            distance[n] = np.linalg.norm(self.BS_Location - self.UE_Locations[n,:])
        sort_ues = np.argsort(distance)        
        self.Near_UEs = sort_ues[0:int(np.ceil(len(sort_ues)/2))]
        self.Far_UEs = sort_ues[int(np.ceil(len(sort_ues)/2)):]
        UEs = [self.Far_UEs,self.Near_UEs]
        # List of UEs with reduction.
        #self.List_UE = [list(p) for p in product(*UEs)]
        # List of UEs without reduction.
        #self.No_List_UEs = [[]]
        #ListUE = combinations(np.arange(0,n_ues,1),2)
        #for e in ListUE:
        #    self.No_List_UEs.append(list(e))
        # List of UEs with heuristics/semistatic.    
        #self.Heuristic_List_UEs = []    
        #H_ListUE = combinations(np.arange(0,self.n_ues,1),2)
        #for e in H_ListUE:
        #    self.Heuristic_List_UEs.append(list(e))
        # List of UEs with round robin:
        #self.RoundRobin_List_UEs = []
        #my_list = [item for item in range(0,n_ues)]*max_iters 
        #while my_list:
        #     chunk, my_list = my_list[:2], my_list[2:]
        #     self.RoundRobin_List_UEs.append(chunk)
            
        # Actions for different schemes.
        if self.OMA_Scheme:
            self.BSnA = self.n_ues
        elif self.NOMA_Scheme:                
            if self.Reduction:
                self.BSnA = self._list_generation_reduction() #len(self.List_UE)  
            elif self.heuristic or self.semi_static:        
                self.BSnA = self._list_generation()  #self._NOMA_action_set_ues() #len(self.Heuristic_List_UEs)
            elif self.Round_robin:
                self.BSnA = self._NOMA_Round_robin() #len(self.RoundRobin_List_UEs) 
            else:
                self.BSnA = self._list_generation_no_reduction() #len(self.No_List_UEs)

        # Initialize entities: adds self.ues, self.bs, self.entities
        entity_params = {"hist_act": self.hist_act,"hist_obs": self.hist_obs,"hist_msg": self.hist_msg,"hist_comm": self.hist_comm,
                         "n_voc_ul": self.n_voc_ul,"n_voc_dl": self.n_voc_dl,"silent": self.silent,"counter": self.counter}
        
        self.bs = BS(idx=0,n_ues=n_ues,n_channels=n_channels,n_actions=self.BSnA,heuristic=self.heuristic,**entity_params)
        self.ues = [UE(idx=ii,**entity_params) for ii in range(n_ues)]
        
        # Entities: [(bs, ue_1, ue_2,...)]
        self.entities = [self.bs,*self.ues] 
        self.agents = [a for a in self.entities if a.is_agent == True]
        self.n_agents = len(self.agents)

        # Initialize channel matrix:
        self.Channel_Matrix = np.zeros((self.n_ues,self.n_channels),dtype=np.complex) 
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
            if isinstance(ag, BS):
                # Action space is Multi Discrete: (Env_Actions, Comm_Actions) 
                if ag.silent:
                    acsp.append(spaces.MultiDiscrete([self.BSnA]*n_channels))
                else:
                    acsp.append(spaces.MultiDiscrete([self.BSnA]*n_channels+([n_voc_dl]*n_ues)))
            elif isinstance(ag, UE):
                acsp.append(spaces.MultiDiscrete([n_voc_ul]))
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
        # Initialize buffers and parameters:
        self.tx_buffer = self._init_buffer()
        self.rx_buffer = np.zeros((self.n_ues,self.max_iters),dtype=np.int8)
        self.iteration = 0    
        self.reward = 0
        self.n_failedtasks = 0
        self.n_successtasks = 0
        self.success_allocation = []
        self.storage = np.empty(self.n_ues, dtype=bool)
        self.channels_state = np.zeros((self.n_channels,self.n_ues + 2),dtype=np.int8)
        self.succ_comp = np.zeros((self.n_ues,self.n_ues),dtype=np.int8) 
        self.succ_tasks = np.zeros(self.n_ues,dtype=np.int8)
        self.chs_state = np.zeros((self.n_channels,2),dtype=np.int8)        
        # Reset and get state of entities:
        obs = []
        obs_buffer = []
        deadline_buffer = []
        self._obs_render = []  
        for ag in self.entities:
            ag.reset()
            if ag.is_actor:
                for agent in self.ues:
                    obs_buffer.append(len(self.tx_buffer[agent.idx]))
                    if len(self.tx_buffer[agent.idx]) > 0:
                       Task_popped = self.tx_buffer[agent.idx][-1]
                       deadline_buffer.append(Task_popped[2])
                    else:
                       deadline_buffer.append(0.0)
                if self.heuristic:                
                    ag.update_obs(obs=obs_buffer,succ_comp=self.succ_comp,buffer_state=self.tx_buffer,deadlines=deadline_buffer)
                else:                    
                    ag.update_obs(obs=obs_buffer,ul_msgs=None,succ_comp=None,ch_status=self.channels_state,ch_mx=self.Channel_Gain,deadlines=deadline_buffer)                    
            if ag.is_agent:
                obs.append(ag.get_state())
            self._obs_render.append(ag.write_obs())
        return obs

    def step(self,actions_n):
        self.iteration += 1   
        self.reward = 0 
        self.n_failedtasks = 0
        self.n_successtasks = 0  
        self.goodput = []
        self.success_allocation = []
        self.storage = np.empty(self.n_ues,dtype=bool)
        self.delay = np.zeros(self.n_ues,dtype=np.float32) 
        self.uplink_rate =  np.zeros(self.n_ues,dtype=np.float32) 
        self.channels_access = np.zeros((self.n_channels,self.n_ues),dtype=np.int8)
        self.channels_state = np.zeros((self.n_channels,self.n_ues+2),dtype=np.int8)         
        self.succ_comp = np.zeros((self.n_ues,self.n_ues),dtype=np.int8) 
               
        # Check action dimensions:
        if self.Round_robin or self.semi_static or self.heuristic:
            self._check_action_input(actions_n)
        else:
            self._check_action_input(actions_n[0])
        for ii, ag in enumerate(self.agents):
            # Pass the actions to each entity:
            if self.Round_robin or self.semi_static or self.heuristic:
                self._process_action(ag, actions_n[ii], self.action_space[ii])
            else:
                self._process_action(ag, actions_n[0][ii], self.action_space[ii])
                
            # Get all environment actions:
            if ag.is_actor:
                env_action = ag.get_action()
            
        # get previous observation:
        self._obs_render = []
        for ii, ag in enumerate(self.entities):
            # Save current observation to render later:
            self._obs_render.append(ag.write_obs())            
        # Actions in the right interval:
        assert np.logical_and(env_action >= 0, env_action < self.BSnA).all(), "Actions out of range"
        
        # Store the performance metrics
        info = {"No. of Success Tasks":0,"Channel Access Success Rate":0,"Channel Access Collision Rate":0,
                "Channel Idle Rate":0,"Packets Drop Rate":0,"Reward":0,"Goodput":0,"No. of Failed Tasks":0}   
        
        if self.OMA_Scheme:
            # Environment Step: take action:
            for ch, ue in enumerate(env_action):
                if self.non_empty_tx_buffers[ue]:
                    Task_popped = self.tx_buffer[ue][-1] 
                    # Remote computation delay.
                    transmit_delay = Task_popped[0]/self._uplink_rate_OMA(ue,ch)
                    BS_comp_delay = Task_popped[0]*Task_popped[1]/self.Computing_Resources[ch] 
                    remote_delay =  transmit_delay  + BS_comp_delay 
                    self.delay[ue] = Task_popped[2] - remote_delay
                    self.storage[ue] = True if self.Caching_Resources[ch] >= Task_popped[0] else False
                    if self.delay[ue] >= 0 and self._uplink_rate_OMA(ue,ch) >= self.uplink_th[ue] and self.storage[ue]:
                        self.n_successtasks += 1
                        self.channels_state[ch,ue] = 1
                        self.chs_state[ch,0] = 1
                        self.reward += self.reward_com
                        self.succ_comp[ue,ue] = 1
                        self.goodput.append(1)
                        self.succ_tasks[ue] += 1
                        self.tx_buffer[ue].pop() 
                        info["Channel Access Success Rate"] += 1
                    elif self.delay[ue] < 0 or self._uplink_rate_OMA(ue,ch) < self.uplink_th[ue] or self.storage[ue] == False:
                        self.reward += self.penality
                        self.n_failedtasks += 1 
                        #self.tx_buffer[ue].pop()
                        self.channels_state[ch,-2] = 1
                        self.chs_state[ch,1] = 1
                        info["Packets Drop Rate"] += 1 
                        info["Channel Idle Rate"] += 1
                else:
                    self.reward += self.penality
        
        elif self.NOMA_Scheme:
            # Map env_actions to the list of users:
            if self.Reduction:
               env_actions = self._access_UEs_list_reduction(env_action) # np.array(self.List_UE)[env_action]
            elif self.Round_robin:
               env_actions = self._NOMA_Round_robin_access(iteration = self.iteration)
               #env_actions = np.array(self.RoundRobin_List_UEs)[[item for item in range((self.iteration - 1)*self.n_channels, self.iteration*self.n_channels)]]  
            elif self.semi_static:
               #env_actions = self._semistatic()
               #env_actions = np.array(self.Heuristic_List_UEs)[env_action] 
               #env_actions = self._NOMA_action_set_ues_access(action_index = actions_n[0][0])
               env_actions = self._access_UEs_list(env_action)
            elif self.heuristic:
               #env_actions = np.array(self.Heuristic_List_UEs)[env_action]
               env_actions = self._access_UEs_list(env_action) 
            else:
               env_actions = self._access_UEs_list_no_reduction(env_action) # np.array(self.No_List_UEs)[env_action]
            
            # Environment Step: Take Action:
            for ch, ues in enumerate(env_actions): 
                self.succ_comp = np.zeros((self.n_ues,self.n_ues),dtype=np.int8) 
                if len(ues) > 0 : #and self.non_empty_tx_buffers[ues].all():
                    for ind,ue in enumerate(ues):
                        if self.non_empty_tx_buffers[ue]:
                           Task_popped = self.tx_buffer[ue][-1]
                           self.uplink_rate[ue] = self._uplink_rate_NOMA(ues,ind,ch)
                           transmit_delay = Task_popped[0]/self.uplink_rate[ue]
                           BS_comp_delay = Task_popped[0]*Task_popped[1]/self.Computing_Resources[ch]
                           remote_delay =   transmit_delay + BS_comp_delay 
                           self.delay[ue] = Task_popped[2] - remote_delay
                           self.storage[ue] = True if self.Caching_Resources[ch] >= Task_popped[0] else False
                           if self.delay[ue] >= 0 and self.uplink_rate[ue] >= self.uplink_th[ue] and self.storage[ue]: 
                              self.n_successtasks += 1 
                              self.succ_comp[ue,ue] = 1
                              self.goodput.append(1)
                              self.succ_tasks[ue] += 1                            
                              self.tx_buffer[ue].pop() 
                              #self.reward += self.reward_com
                           elif self.delay[ue] < 0 or self.uplink_rate[ue] < self.uplink_th[ue] or self.storage[ue] == False:
                              #self.tx_buffer[ue].pop()  
                              self.n_failedtasks += 1                            
                              info["Packets Drop Rate"] += 1
                              #self.reward +=  self.penality 
                        #else:
                        #    self.reward += self.penality 
                else:
                    #self.reward += self.penality 
                    self.channels_state[ch,-2] = 1
                    
                if np.sum(self.succ_comp[ues,ues]) >= 2:
                    info["Channel Access Success Rate"] += 1
                    self.chs_state[ch,0] = 1
                    self.channels_state[ch,ues] = 1
                else:
                    info["Channel Idle Rate"] += 1
                    self.chs_state[ch,1] = 1
                    self.channels_state[ch,-2] = 1
        
        self.reward +=  - (2*self.n_channels - self.n_successtasks) #- info["Channel Idle Rate"]  #- self.n_failedtasks  # #
        
        #if (2*self.n_channels - self.n_successtasks) == 0:
        #    self.reward += 1
        
        #self.reward = self.reward/(2*self.n_channels)
        self.No_SCH = np.count_nonzero(np.array(self.channels_state[:,0:self.n_ues]) == 1) # No. of scheduled users.  
        info["Channel Access Collision Rate"] = np.count_nonzero(np.array(self.channels_state[:,-1]) == 1)/self.n_channels
        if len(self.goodput) > 0:    
            info["Goodput"] += np.sum(self.goodput)
        info["No. of Success Tasks"] += self.n_successtasks/(2*self.n_channels) 
        info["Reward"] += self.reward 
        info["No. of Failed Tasks"] += self.n_failedtasks/(2*self.n_channels) 
        info["Channel Idle Rate"] = info["Channel Idle Rate"]/self.n_channels
        info["Channel Access Success Rate"] = info["Channel Access Success Rate"]/self.n_channels
        
        # Transit:
        # Maybe generate traffic:
        self._gen_traffic()
        # Update observations of all entities:
        ul_msgs = [ue.comm_action for ue in self.ues] if not self.silent else None      
        dl_msgs = self.bs.comm_action if not self.silent else [None]*self.n_ues
        for ii, ag in enumerate(self.entities):
            if isinstance(ag,UE):
                ag.update_obs(obs=len(self.tx_buffer[ag.idx]),dl_msg=dl_msgs[ag.idx])
            else:
                obs_buffer = []
                deadline_buffer = []
                for agent in self.ues:
                    obs_buffer.append(len(self.tx_buffer[agent.idx]))
                    if len(self.tx_buffer[agent.idx]) > 0:
                       Task_popped = self.tx_buffer[agent.idx][-1]
                       deadline_buffer.append(Task_popped[2])
                    else:
                       deadline_buffer.append(0.0)
                if self.heuristic:                
                    ag.update_obs(obs=obs_buffer,succ_comp=self.succ_comp,buffer_state=self.tx_buffer)
                else:
                    ag.update_obs(obs=obs_buffer,ul_msgs=ul_msgs,succ_comp=self.succ_comp,ch_status=self.channels_state,ch_mx=self.Channel_Gain,deadlines=deadline_buffer)
        
        # Check if Final iteration:
        done = bool(self.iteration > self.max_iters) or bool(obs_buffer == 0)
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
    
    def _env_action_sizes(self,actions):
        return all(np.array([len(a) for a in actions])) > 0
    
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
        SNR = self.UE_Power*self.Channel_Gain[ue,selected_channel]/(self.Noise_Power/self.n_channels) # SNR
        Up_Rate = (self.BS_BW/self.n_channels)*np.log10(1+SNR)     # Uplink rate. 
        return Up_Rate
    
    def _uplink_rate_NOMA(self,ues,ind,selected_channel):  
        SNR = np.zeros(2)
        Up_Rate = np.zeros(2)
        if self.Channel_Gain[ues[0],selected_channel] > self.Channel_Gain[ues[1],selected_channel]:
            SNR[0] = self.UE_Power*self.Channel_Gain[ues[0],selected_channel]/((self.Noise_Power/self.n_channels) + self.UE_Power*self.Channel_Gain[ues[1],selected_channel]) # SNR
            SNR[1] = self.UE_Power*self.Channel_Gain[ues[1],selected_channel]/(self.Noise_Power/self.n_channels) # SNR
        else: 
            SNR[1] = self.UE_Power*self.Channel_Gain[ues[1],selected_channel]/((self.Noise_Power/self.n_channels) + self.UE_Power*self.Channel_Gain[ues[0],selected_channel]) # SNR
            SNR[0] = self.UE_Power*self.Channel_Gain[ues[0],selected_channel]/(self.Noise_Power/self.n_channels) # SNR
        Up_Rate[0] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[0])     # Uplink rate. 
        Up_Rate[1] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[1])     # Uplink rate.
        return Up_Rate[ind]
    

    def _semistatic(self):
        action = [[]]*self.n_channels 
        chosen_ues = []
        possible_ues = [item for item in range(0,self.n_ues)] 
        env_action = [[]]*self.n_channels 
        for ch in range(self.n_channels):    
            action[ch] = np.random.choice(possible_ues,2,replace=False)
            chosen_ues.append(action[ch].tolist())
            for ue in action[ch].tolist():
                possible_ues.remove(ue)
        for ch in range(self.n_channels): 
            env_action[ch] = action[ch].tolist()
        return env_action
        
    
    # Action set for round robin
    def _NOMA_Round_robin(self):
        with open("roundrobin.txt", "w") as file:
            no_actions = 0
            arr = [item for item in range(0,self.n_ues)] 
            my_list = arr*self.max_iters
            while my_list:
                chunk, my_list = my_list[:2], my_list[2:]
                file.write("{}\n".format(chunk))
                no_actions += 1
            file.close() 
        return no_actions 
    
    # Access action set for round robin
    def _NOMA_Round_robin_access(self,iteration):
        with open("roundrobin.txt", 'r') as file:
            action = []
            for x, line in enumerate(file):
                if x in range((iteration - 1)*self.n_channels,iteration*self.n_channels):
                    content = line.rstrip()
                    action.append(ast.literal_eval(f'{content}'))
        return np.array(action)   
        
        
    # Action set for heuristics/semi-static
    def _NOMA_action_set_ues(self):
        with open("heuristic.txt", "w") as file:
            no_actions = 0
            List_UEs = []
            List_UE = combinations(np.arange(0,self.n_ues,1),2)
            for e in List_UE:
                List_UEs.append(list(e))
            comb = permutations(List_UEs,self.n_channels)
            for i in comb:
                unique_elements, counts= np.unique(list(i), return_counts=True)
                if all(counts == 1):
                    no_actions += 1
                    file.write("{}\n".format(list(i)))
            file.close() 
        return no_actions 
    
    # Access action set for heuristics/semi-static
    def _NOMA_action_set_ues_access(self,action_index): 
        content = linecache.getline("heuristic.txt", action_index + 1)
        action = ast.literal_eval(f'{content}')
        return np.array(action)
        
        
    # Generate a list of combination of users
    def _list_generation(self):
        with open("UEs_list.txt", "w") as file:
            no_actions = 0
            for a,i in  enumerate(combinations(np.arange(0,self.n_ues,1),2)):
                    no_actions += 1
                    file.write("{}\n".format(list(i)))
            file.close() 
        return no_actions 
        
    def _access_UEs_list(self,env_action):
        action = []
        for a in env_action:
            content = linecache.getline("UEs_list.txt", a + 1)
            action_a = ast.literal_eval(f'{content}')
            action.append(action_a)
        return np.array(action)
        
        
        
    # Generate a list of combination of users for no-reduction:
    def _list_generation_no_reduction(self):
        with open("UEs_list_no_red.txt", "w") as file:
            no_actions = 0
            file.write("{}\n".format(list([])))
            for a,i in  enumerate(combinations(np.arange(0,self.n_ues,1),2)):
                    no_actions += 1
                    file.write("{}\n".format(list(i)))
            file.close() 
        return no_actions 
        
    def _access_UEs_list_no_reduction(self,env_action):
        action = []
        for a in env_action:
            content = linecache.getline("UEs_list_no_red.txt", a + 1)
            action_a = ast.literal_eval(f'{content}')
            action.append(action_a)
        return np.array(action)
        
    
    # Generate a list of combination of users for reduction:
    def _list_generation_reduction(self):
        with open("UEs_list_red.txt", "w") as file:
            no_actions = 0
            UEs = [self.Far_UEs,self.Near_UEs]
            for a,i in  enumerate(product(*UEs)):
                    no_actions += 1
                    file.write("{}\n".format(list(i)))
            file.close() 
        return no_actions 
        
    def _access_UEs_list_reduction(self,env_action):
        action = []
        for a in env_action:
            content = linecache.getline("UEs_list_red.txt", a + 1)
            action_a = ast.literal_eval(f'{content}')
            action.append(action_a)
        return np.array(action)