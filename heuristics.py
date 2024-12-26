import gym.spaces as spaces
import numpy as np
import random
import ast
from gym.utils import seeding
from itertools import permutations,product,combinations
import linecache
import re

class BaseStation05:
    def __init__(self,n_ues,n_channels,n_actions,Channel_Matrix,UE_Power,Noise_Power,BS_BW,BS_CPU,uplink_th,Caching_Resources,Computing_Resources):
        self.n_ues = n_ues
        self.n_channels = n_channels
        self.n_actions = n_actions
        self.Channel_Matrix = Channel_Matrix
        self.UE_Power = UE_Power
        self.Noise_Power = Noise_Power 
        self.BS_BW = BS_BW
        self.BS_CPU = BS_CPU
        self.uplink_th = uplink_th
        self.Computing_Resources = Computing_Resources
        self.Caching_Resources = Caching_Resources
        #self.List_UEs = List_UEs
        self.seed()

            
    def reset(self):
        self.env_action = 0
        self.comm_action = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.rng = np.random.default_rng(seed)
        return [seed]

    def act(self, agent_state): 
        indexs = []
        if not agent_state == [None,None]:
            self.tx_buffer = agent_state[0]
            hyperedges = self._calculate_edges_weight()
            env_actions = self._heuristic_selection(hyperedges) 
            for ues in env_actions:
                #indexs.append(self.List_UEs.index(ues))
                with open("UEs_list.txt") as search:
                     for ind,line in enumerate(search):
                         line = line.rstrip()  # remove '\n' at end of line
                         temp = re.findall(r'\d+', line)
                         res = list(map(int, temp))
                         if ues == res:
                            indexs.append(ind)
                            break
        return indexs

    def learn(self, state, action, reward, next_state):
        pass


    # Weight calculation to each hyperedge
    def _calculate_edges_weight(self):        
        self.No_SCH = 2*self.n_channels
        hyperedges = np.array([[0,0,0,0]])        
        action_list = [] 
        #for ues in self.List_UEs:
        for a in range(self.n_actions):
            ues_s = linecache.getline("UEs_list.txt", a + 1)
            ues_s = ues_s.rstrip('\n')
            temp = re.findall(r'\d+',  ues_s)
            ues = list(map(int, temp))
            for ch in np.arange(0,self.n_channels,1):
                self.succ_tasks = np.zeros(self.n_ues,dtype=np.int8)
                self.delay = np.zeros(self.n_ues,dtype=np.float32)
                hyperedge = np.zeros((4),dtype=np.int8)
                for ind,ue in enumerate(ues):
                        hyperedge[ind] = ue
                        hyperedge[2] = ch
                        if len(self.tx_buffer[ue]) > 0:  
                            Task_popped = self.tx_buffer[ue][-1]
                            # Remote computation delay.
                            transmit_delay = Task_popped[0]/self._uplink_rate_NOMA(ues,ind,ch)
                            BS_comp_delay = Task_popped[0]*Task_popped[1]/self.Computing_Resources[ch] #(self.BS_CPU/self.No_SCH)
                            remote_delay =  transmit_delay  + BS_comp_delay 
                            self.delay[ue] = Task_popped[2] - remote_delay
                            self.storage = True if self.Caching_Resources[ch] >= Task_popped[0] else False
                            if self.delay[ue] >= 0 and self._uplink_rate_NOMA(ues,ind,ch) >= self.uplink_th[ue] and self.storage:
                                self.succ_tasks[ue] += 1 
                hyperedge[3] = np.sum(self.succ_tasks[ues])
                hyperedges = np.vstack((hyperedges,hyperedge)) 
        hyperedges = np.delete(hyperedges,0,0)
        return hyperedges

    # Heuristic selection
    def _heuristic_selection(self,hyperedges):        
        action = []       
        for ch in np.arange(0,self.n_channels,1): 
            if len(hyperedges) > 0: 
                high_weg = np.where(hyperedges[:,3] == hyperedges[np.argmax(hyperedges[:,3]),3])
                sel = np.random.choice(high_weg[0])
                chosen_ues = hyperedges[sel,0:2]
                action.append(hyperedges[sel,0:2].tolist())
                value = np.where(hyperedges[:,2] == ch)
                hyperedges = np.delete(hyperedges,value,0)
                #index_1 = np.where(hyperedges[:,0:2] == chosen_ues[0])              
                #hyperedges = np.delete(hyperedges,index_1,0)
                #index_2 = np.where(hyperedges[:,0:2] == chosen_ues[1])
                #hyperedges = np.delete(hyperedges,index_2,0)
        return action
    
    # Calculate the uplink rate of each user.
    def _uplink_rate_NOMA(self,ues,ind,selected_channel):  
        SNR = np.zeros(2)
        Up_Rate = np.zeros(2)
        if np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2> np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2:
            SNR[0] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2/((self.Noise_Power/self.n_channels) + self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2) # SNR
            SNR[1] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2/(self.Noise_Power/self.n_channels) # SNR
        else: 
            SNR[1] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[1],selected_channel])**2/((self.Noise_Power/self.n_channels) + self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2) # SNR
            SNR[0] = self.UE_Power*np.linalg.norm(self.Channel_Matrix[ues[0],selected_channel])**2/(self.Noise_Power/self.n_channels) # SNR
        Up_Rate[0] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[0])  # Uplink rate. 
        Up_Rate[1] = (self.BS_BW/self.n_channels)*np.log10(1+SNR[1])  # Uplink rate.
        return Up_Rate[ind]
    