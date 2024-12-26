import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scienceplots
import matplotlib
import matplotlib.ticker as ticker
import matplotlib as mpl
from plot_utils import window_mean
import csv
import copy
import ast
import torch
import random
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
from misc import process_oh_actions
from envs import MECSCHEnvV1
from envs import MECSCHEnvV2
from tqdm.notebook import tqdm
from agent_wrapper import OnPolicyWrapper
from models import MLPCategoricalActor, MLPRelaxedCategoricalActor, BaseMLPNet, BaseMLPActor
from plot_functions import set_fonts, set_style, draw_boxplot, draw_brace
from gym.envs import register
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from colorsys import rgb_to_hls
from copy import deepcopy

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def register_envs():  
    # 'MECSCH-v1': Mobile edge computing full computation offloading no communication NOMA with reduction
    gym.envs.register(id='MECSCH-v1',entry_point=MECSCHEnvV2,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':True,'NOMA_Scheme':True,'OMA_Scheme':False,'Reduction':True,
                 'Round_robin':False,'semi_static':False,'heuristic':False})
    
    # 'MECSCH-v2': Mobile edge computing full computation offloading no communication NOMA without reduction
    gym.envs.register(id='MECSCH-v2',entry_point=MECSCHEnvV2,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':True,'NOMA_Scheme':True,'OMA_Scheme':False,'Reduction':False,
                 'Round_robin':False,'semi_static':False,'heuristic':False})
    
    # 'MECSCH-v3': Mobile edge computing full computation offloading semi-static NOMA 
    gym.envs.register(id='MECSCH-v3',entry_point=MECSCHEnvV2,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':True,'NOMA_Scheme':True,'OMA_Scheme':False,'Reduction':False,
                 'Round_robin':False,'semi_static':True,'heuristic':False})
        
    # 'MECSCH-v4': Mobile edge computing full computation offloading round-robin NOMA 
    gym.envs.register(id='MECSCH-v4',entry_point=MECSCHEnvV2,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':True,'NOMA_Scheme':True,'OMA_Scheme':False,'Reduction':False,
                 'Round_robin':True,'semi_static':False,'heuristic':False})
     
    # 'MECSCH-v5': Mobile edge computing full computation offloading heuristic NOMA 
    gym.envs.register(id='MECSCH-v5',entry_point=MECSCHEnvV2,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':True,'NOMA_Scheme':True,'OMA_Scheme':False,'Reduction':False,
                 'Round_robin':False,'semi_static':False,'heuristic':True})
       
    # 'MECSCH-v6': Mobile edge computing full computation offloading contention-free/contention-based NOMA 
    gym.envs.register(id='MECSCH-v6',entry_point=MECSCHEnvV1,
         kwargs={'n_ues':26,'n_channels':3,'UE_buffer_capacity':200,'UE_CPU':1e9,'BS_CPU':120e9,'BS_BW':30e6,'hist_msg':1,
                 'hist_comm':1,'hist_obs':1,'hist_act':1,'n_voc_ul':2,'n_voc_dl':5,'arrival_prob':0.8,'max_iters':15,
                 'reward_com':0,'penality':-2,'silent':False,'NOMA_Scheme':True,'OMA_Scheme':False})   

from contention_free_V2 import BaseStation01, UEAgent01
def base_runner(env_id,n_episodes=1000,n_eval=10,eval_every=10,n_eval_episodes=10,max_ep_len=30,seed=1024):    
    register_envs()
    env = gym.make(env_id)
    env.seed(seed)
    seed_everything(seed)
    
    n_ues = env.n_ues
    n_channels = env.n_channels
    hist_msg = env.hist_msg
    hist_obs = env.hist_obs
    n_voc_ul = env.n_voc_ul
    n_voc_dl = env.n_voc_dl
        
    agents = [BaseStation01(n_ues,n_channels,hist_msg,hist_obs,n_voc_ul)]
    for ii in range(n_ues):
        agents.append(UEAgent01(n_channels,hist_msg,hist_obs,n_voc_dl))

    for ii, ag in enumerate(agents):
        ag.seed(seed + ii)
        
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    train_value = []
    values = 0
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in tqdm(range(n_episodes)):
        done, terminal = False, False
        ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0     
        obs = env.reset()
        for ag in agents:
            ag.reset()
    
        while not (done or terminal):
            ep_len += 1
            actions = [ag.act(obs[ii]) for ii, ag in enumerate(agents)]
            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones)
            terminal = ep_len > env.max_iters
            obs = next_obs          
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
        
        train_value.append(values)
        train_actor_loss.append(policy_loss[0])
        train_critic_loss.append(critic_loss[0])
        train_rewards.append(ep_reward)
        train_success_tasks.append(ep_success_tasks/ep_len)
        train_channel_success.append(ep_channel_success/ep_len)
        train_channel_collision.append(ep_channel_collision/ep_len)
        train_channel_idle.append(ep_channel_idle/ep_len)
        train_goodput.append(ep_goodput/ep_len)
        train_droprate.append(ep_droprate)
        train_failed.append(ep_failed/ep_len)
     
    eval_rewards = np.mean(np.array(train_rewards).reshape(-1,n_eval), axis=1).tolist()
    eval_success_tasks = np.mean(np.array(train_success_tasks).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_success = np.mean(np.array(train_channel_success).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_collision = np.mean(np.array(train_channel_collision).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_idle = np.mean(np.array(train_channel_idle).reshape(-1,n_eval), axis=1).tolist()
    eval_goodput = np.mean(np.array(train_goodput).reshape(-1,n_eval), axis=1).tolist()
    eval_droprate = np.mean(np.array(train_droprate).reshape(-1,n_eval), axis=1).tolist()
    eval_failed = np.mean(np.array(train_failed).reshape(-1,n_eval), axis=1).tolist()
   
    return train_rewards,train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,\
           train_goodput,train_droprate,train_failed,eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,\
           eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value            
           
           
from contention_based_V2 import BaseStation02, UEAgent02
def contention_base_runner(env_id,n_episodes=1000,n_eval=10,eval_every=10,n_eval_episodes=10,max_ep_len=30,seed=1024):    
    register_envs()
    env = gym.make(env_id)
    env.seed(seed)
    seed_everything(seed)
    
    n_ues = env.n_ues
    n_channels = env.n_channels
    hist_msg = env.hist_msg
    hist_obs = env.hist_obs
    n_voc_ul = env.n_voc_ul
    n_voc_dl = env.n_voc_dl
        
    agents = [BaseStation02(n_ues,n_channels,hist_msg,hist_obs,n_voc_ul)]
    for ii in range(n_ues):
        agents.append(UEAgent02(n_channels,hist_msg,hist_obs,n_voc_dl))

    for ii, ag in enumerate(agents):
        ag.seed(seed + ii)
            
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    train_value = []
    values = 0
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in tqdm(range(n_episodes)):
        done, terminal = False, False
        ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0     
        obs = env.reset()
        for ag in agents:
            ag.reset()
    
        while not (done or terminal):
            ep_len += 1
            actions = [ag.act(obs[ii]) for ii, ag in enumerate(agents)]
            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones)
            terminal = ep_len > env.max_iters
            obs = next_obs
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
        
        train_value.append(values)
        train_actor_loss.append(policy_loss[0])
        train_critic_loss.append(critic_loss[0]) 
        train_rewards.append(ep_reward)
        train_success_tasks.append(ep_success_tasks/ep_len)
        train_channel_success.append(ep_channel_success/ep_len)
        train_channel_collision.append(ep_channel_collision/ep_len)
        train_channel_idle.append(ep_channel_idle/ep_len)
        train_goodput.append(ep_goodput/ep_len)
        train_droprate.append(ep_droprate)
        train_failed.append(ep_failed/ep_len)
    
    eval_rewards = np.mean(np.array(train_rewards).reshape(-1,n_eval), axis=1).tolist()
    eval_success_tasks = np.mean(np.array(train_success_tasks).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_success = np.mean(np.array(train_channel_success).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_collision = np.mean(np.array(train_channel_collision).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_idle = np.mean(np.array(train_channel_idle).reshape(-1,n_eval), axis=1).tolist()
    eval_goodput = np.mean(np.array(train_goodput).reshape(-1,n_eval), axis=1).tolist()
    eval_droprate = np.mean(np.array(train_droprate).reshape(-1,n_eval), axis=1).tolist()
    eval_failed = np.mean(np.array(train_failed).reshape(-1,n_eval), axis=1).tolist()
    
    return train_rewards,train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,train_goodput,\
           train_droprate,train_failed,eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,\
           eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value 
           
from semi_static import BaseStation03
def semi_static_runner(env_id,n_episodes=1000,n_eval = 10,eval_every=10,n_eval_episodes=10,max_ep_len=30,seed=1024):    
    register_envs()
    env = gym.make(env_id)
    env.seed(seed)
    seed_everything(seed)
    
    n_ues = env.n_ues
    n_channels = env.n_channels
    hist_msg = env.hist_msg
    hist_obs = env.hist_obs
    n_voc_ul = env.n_voc_ul
    n_voc_dl = env.n_voc_dl
    n_actions = env.BSnA
        
    agents = [BaseStation03(n_ues,n_channels,hist_msg,hist_obs,n_voc_ul,n_actions)]

    for ii, ag in enumerate(agents):
        ag.seed(seed + ii)
            
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    train_value = []
    values = 0
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in tqdm(range(n_episodes)):
        done, terminal = False, False
        ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0      
        obs = env.reset()
        for ag in agents:
            ag.reset()
    
        while not (done or terminal):
            ep_len += 1
            actions = [ag.act(obs[ii]) for ii, ag in enumerate(agents)]
            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones)
            terminal = ep_len > env.max_iters
            obs = next_obs
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
        
        train_value.append(values)
        train_actor_loss.append(policy_loss[0])
        train_critic_loss.append(critic_loss[0])
        train_rewards.append(ep_reward)
        train_success_tasks.append(ep_success_tasks/ep_len)
        train_channel_success.append(ep_channel_success/ep_len)
        train_channel_collision.append(ep_channel_collision/ep_len)
        train_channel_idle.append(ep_channel_idle/ep_len)
        train_goodput.append(ep_goodput/ep_len)
        train_droprate.append(ep_droprate)
        train_failed.append(ep_failed/ep_len)
    
    eval_rewards = np.mean(np.array(train_rewards).reshape(-1,n_eval), axis=1).tolist()
    eval_success_tasks = np.mean(np.array(train_success_tasks).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_success = np.mean(np.array(train_channel_success).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_collision = np.mean(np.array(train_channel_collision).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_idle = np.mean(np.array(train_channel_idle).reshape(-1,n_eval), axis=1).tolist()
    eval_goodput = np.mean(np.array(train_goodput).reshape(-1,n_eval), axis=1).tolist()
    eval_droprate = np.mean(np.array(train_droprate).reshape(-1,n_eval), axis=1).tolist()
    eval_failed = np.mean(np.array(train_failed).reshape(-1,n_eval), axis=1).tolist()

    
    return train_rewards, train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,\
           train_goodput,train_droprate,train_failed,eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,\
           eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value 
           
from Round_robin import BaseStation04
def Round_robin_runner(env_id,n_episodes=1000,n_eval=10,eval_every=10,n_eval_episodes=10,max_ep_len=30,seed=1024):    
    register_envs()
    env = gym.make(env_id)
    env.seed(seed)
    seed_everything(seed)
    
    n_ues = env.n_ues
    n_channels = env.n_channels
    hist_msg = env.hist_msg
    hist_obs = env.hist_obs
    n_voc_ul = env.n_voc_ul
        
    agents = [BaseStation04(n_ues,n_channels,hist_msg,hist_obs,n_voc_ul)]

    for ii, ag in enumerate(agents):
        ag.seed(seed + ii)
            
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    train_value = []
    values = 0
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in tqdm(range(n_episodes)):
        done, terminal = False, False
        ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0   
        obs = env.reset()
        for ag in agents:
            ag.reset()
    
        while not (done or terminal):
            ep_len += 1
            actions = [ag.act(obs[ii],ep_len) for ii, ag in enumerate(agents)]
            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones)
            terminal = ep_len > env.max_iters
            obs = next_obs
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
        
        train_value.append(values)
        train_actor_loss.append(policy_loss[0])
        train_critic_loss.append(critic_loss[0])
        train_rewards.append(ep_reward)
        train_success_tasks.append(ep_success_tasks/ep_len)
        train_channel_success.append(ep_channel_success/ep_len)
        train_channel_collision.append(ep_channel_collision/ep_len)
        train_channel_idle.append(ep_channel_idle/ep_len)
        train_goodput.append(ep_goodput/ep_len)
        train_droprate.append(ep_droprate)
        train_failed.append(ep_failed/ep_len)
    
    eval_rewards = np.mean(np.array(train_rewards).reshape(-1,n_eval), axis=1).tolist()
    eval_success_tasks = np.mean(np.array(train_success_tasks).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_success = np.mean(np.array(train_channel_success).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_collision = np.mean(np.array(train_channel_collision).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_idle = np.mean(np.array(train_channel_idle).reshape(-1,n_eval), axis=1).tolist()
    eval_goodput = np.mean(np.array(train_goodput).reshape(-1,n_eval), axis=1).tolist()
    eval_droprate = np.mean(np.array(train_droprate).reshape(-1,n_eval), axis=1).tolist()
    eval_failed = np.mean(np.array(train_failed).reshape(-1,n_eval), axis=1).tolist()
    
    return train_rewards,train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,\
           train_goodput,train_droprate,train_failed,eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,\
           eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value 



from heuristics import BaseStation05
def heuristics_runner(env_id,n_episodes=1000,n_eval=10,eval_every=10,n_eval_episodes=10,max_ep_len=30,seed=1024):    
    register_envs()
    env = gym.make(env_id)
    env.seed(seed)
    seed_everything(seed)
    
    n_ues = env.n_ues
    n_channels = env.n_channels    
    n_actions = env.BSnA
    Channel_Matrix = env.Channel_Matrix
    UE_Power = env.UE_Power
    Noise_Power = env.Noise_Power
    BS_BW = env.BS_BW
    BS_CPU = env.BS_CPU
    uplink_th = env.uplink_th
    Caching_Resources = env.Caching_Resources
    Computing_Resources = env.Computing_Resources
    #List_UEs = env.Heuristic_List_UEs
    
    #agents = [BaseStation05(n_ues,n_channels,n_actions,Channel_Matrix,UE_Power,Noise_Power,BS_BW,BS_CPU,uplink_th,Caching_Resources,Computing_Resources,List_UEs)]
    agents = [BaseStation05(n_ues,n_channels,n_actions,Channel_Matrix,UE_Power,Noise_Power,BS_BW,BS_CPU,uplink_th,Caching_Resources,Computing_Resources)]

    for ii, ag in enumerate(agents):
        ag.seed(seed + ii)
            
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    train_value = []
    values = 0
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in tqdm(range(n_episodes)):
        done, terminal = False, False
        ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0      
        obs = env.reset()
        for ag in agents:
            ag.reset()
    
        while not (done or terminal):
            ep_len += 1
            actions = [ag.act(obs[ii]) for ii, ag in enumerate(agents)]
            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones)
            terminal = ep_len > env.max_iters
            obs = next_obs
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
        
        train_value.append(values)
        train_actor_loss.append(policy_loss[0])
        train_critic_loss.append(critic_loss[0])
        train_rewards.append(ep_reward)
        train_success_tasks.append(ep_success_tasks/ep_len)
        train_channel_success.append(ep_channel_success/ep_len)
        train_channel_collision.append(ep_channel_collision/ep_len)
        train_channel_idle.append(ep_channel_idle/ep_len)
        train_goodput.append(ep_goodput/ep_len)
        train_droprate.append(ep_droprate)
        train_failed.append(ep_failed/ep_len)
    
    eval_rewards = np.mean(np.array(train_rewards).reshape(-1,n_eval), axis=1).tolist()
    eval_success_tasks = np.mean(np.array(train_success_tasks).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_success = np.mean(np.array(train_channel_success).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_collision = np.mean(np.array(train_channel_collision).reshape(-1,n_eval), axis=1).tolist()
    eval_channel_idle = np.mean(np.array(train_channel_idle).reshape(-1,n_eval), axis=1).tolist()
    eval_goodput = np.mean(np.array(train_goodput).reshape(-1,n_eval), axis=1).tolist()
    eval_droprate = np.mean(np.array(train_droprate).reshape(-1,n_eval), axis=1).tolist()
    eval_failed = np.mean(np.array(train_failed).reshape(-1,n_eval), axis=1).tolist()
    #print("eval_rewards",len(eval_rewards))
    return train_rewards,train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,\
           train_goodput,train_droprate,train_failed,eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,\
           eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value 

def test_agents(agents,env_id="MECSCH-v0",n_episodes=10,max_ep_len=30):
    register_envs()
    env = gym.make(env_id)
    env.seed(1024)
       
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_len, ep_droprate, ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0  
        done, terminal = False, False
        while not (done or terminal):
            ep_len += 1
            actions = agents.act(obs,explore=False)
            next_obs, rewards, dones, info = env.step(actions)
            terminal = ep_len > max_ep_len
            done = all(dones)
            obs = next_obs
            ep_reward += np.mean(rewards)
            ep_success_tasks += info["No. of Success Tasks"]
            ep_channel_success += info["Channel Access Success Rate"]
            ep_channel_collision += info["Channel Access Collision Rate"]
            ep_channel_idle += info["Channel Idle Rate"]
            ep_goodput += info["Goodput"]
            ep_droprate += info["Packets Drop Rate"]
            ep_failed += info["No. of Failed Tasks"]
            
        eval_rewards.append(ep_reward)
        eval_success_tasks.append(ep_success_tasks/ep_len)
        eval_channel_success.append(ep_channel_success/ep_len)
        eval_channel_collision.append(ep_channel_collision/ep_len)
        eval_channel_idle.append(ep_channel_idle/ep_len)
        eval_goodput.append(ep_goodput/ep_len)
        eval_droprate.append(ep_droprate)
        eval_failed.append(ep_failed/ep_len)
        
    return eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,eval_channel_idle,eval_goodput,eval_droprate,eval_failed


def run_sim_on(env_id="MECSCH-v0",n_episodes=1000,max_ep_len=30,parameter_sharing=False,seed=1024,disable_tqdm=False,
               eval_every=10,n_eval_episodes=10,n_feval_episodes=100,**kwargs):
    
    register_envs()
    env = gym.make(env_id)
    agents = OnPolicyWrapper(env,parameter_sharing=parameter_sharing,**kwargs)
    env.seed(seed)
    seed_everything(seed)    
    batch_size = kwargs.get("batch_size",32)
    previous_best_ST = 0
    best_agents = deepcopy(agents)
    
    ep_point = []
    train_rewards = []    
    train_success_tasks = []
    train_channel_success = []
    train_channel_collision = []
    train_channel_idle = []
    train_goodput = []
    train_droprate = []
    train_failed = []
    train_actor_loss = []
    train_critic_loss = []
    train_value = []
    
    evals_rewards = []
    evals_success_tasks = []
    evals_channel_success = []
    evals_channel_collision = []
    evals_channel_idle = []
    evals_goodput = []
    evals_droprate = []
    evals_failed = []
    total_count = 0
    critic_loss = {0: np.array(0.0)}
    policy_loss = {0: np.array(0.0)}
    
    eval_rewards = []
    eval_success_tasks = []
    eval_channel_success = []
    eval_channel_collision = []
    eval_channel_idle = []
    eval_goodput = []
    eval_droprate = []
    eval_failed = []
    
    with tqdm(total=n_episodes,desc="Training",disable=disable_tqdm) as pbar:
        for ep in range(n_episodes):
            obs = env.reset()
            ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0     
            done, terminal = False, False
            while not (done or terminal):
                ep_len += 1
                total_count += 1
                actions = agents.act(obs)
                values = agents.estimate_value(obs)
                next_obs, rewards, dones, info = env.step(actions)               
                terminal = ep_len > max_ep_len
                agents.experience(ep, obs, actions, rewards, next_obs, dones, values)
                done = all(dones)
                if total_count >= batch_size:
                    next_values = agents.estimate_value(next_obs)
                    critic_loss, policy_loss = agents.update(next_values)
                    total_count = 0
                obs = next_obs
                ep_reward += np.mean(rewards)
                ep_success_tasks += info["No. of Success Tasks"]
                ep_channel_success += info["Channel Access Success Rate"]
                ep_channel_collision += info["Channel Access Collision Rate"]
                ep_channel_idle += info["Channel Idle Rate"]
                ep_goodput += info["Goodput"]
                ep_droprate += info["Packets Drop Rate"]
                ep_failed += info["No. of Failed Tasks"]
                
            pbar.set_postfix({"episode": ep+1,"Training reward": np.round(ep_reward, decimals=2)})
            pbar.update(1)
            
            train_value.append(values[0])
            train_actor_loss.append(policy_loss[0])
            train_critic_loss.append(critic_loss[0])
            train_rewards.append(ep_reward)
            train_success_tasks.append(ep_success_tasks/ep_len)
            train_channel_success.append(ep_channel_success/ep_len)
            train_channel_collision.append(ep_channel_collision/ep_len)
            train_channel_idle.append(ep_channel_idle/ep_len)
            train_goodput.append(ep_goodput/ep_len)
            train_droprate.append(ep_droprate)
            train_failed.append(ep_failed/ep_len)
            
            if ep == n_episodes-1:
                for ep in range(n_eval_episodes):
                    obs = env.reset()
                    ep_reward,ep_len,ep_droprate,ep_failed,ep_success_tasks,ep_channel_success,ep_channel_collision,ep_channel_idle,ep_goodput = 0,0,0,0,0,0,0,0,0     
                    done, terminal = False, False
                    while not (done or terminal):
                        ep_len += 1
                        actions = agents.act(obs,explore=False)
                        next_obs, rewards, dones, info = env.step(actions)               
                        terminal = ep_len > max_ep_len
                        done = all(dones)
                        obs = next_obs
                        ep_reward += np.mean(rewards)
                        ep_success_tasks += info["No. of Success Tasks"]
                        ep_channel_success += info["Channel Access Success Rate"]
                        ep_channel_collision += info["Channel Access Collision Rate"]
                        ep_channel_idle += info["Channel Idle Rate"]
                        ep_goodput += info["Goodput"]
                        ep_droprate += info["Packets Drop Rate"]
                        ep_failed += info["No. of Failed Tasks"]
                
                    eval_rewards.append(ep_reward)
                    eval_success_tasks.append(ep_success_tasks/ep_len)
                    eval_channel_success.append(ep_channel_success/ep_len)
                    eval_channel_collision.append(ep_channel_collision/ep_len)
                    eval_channel_idle.append(ep_channel_idle/ep_len)
                    eval_goodput.append(ep_goodput/ep_len)
                    eval_droprate.append(ep_droprate)
                    eval_failed.append(ep_failed/ep_len)
                    
            #if ep % eval_every == 0:                
            #    eval_reward,eval_success_tasks,eval_channel_success,eval_channel_collision,eval_channel_idle,eval_goodput,eval_droprate,eval_failed = test_agents(best_agents, env_id, n_episodes=n_eval_episodes, max_ep_len=max_ep_len)
            #    ep_point.append(ep)
            #    evals_rewards.append(np.mean(eval_reward))
            #    evals_success_tasks.append(np.mean(eval_success_tasks))
            #    evals_channel_success.append(np.mean(eval_channel_success))
            #    evals_channel_collision.append(np.mean(eval_channel_collision))
            #    evals_channel_idle.append(np.mean(eval_channel_idle))
            #    evals_goodput.append(np.mean(eval_goodput))
            #    evals_droprate.append(np.mean(eval_droprate))
            #    evals_failed.append(np.mean(eval_failed))
            #    if np.mean(eval_success_tasks) > previous_best_ST:
            #       del best_agents
            #       best_agents = deepcopy(agents)
            #       previous_best_ST = np.mean(eval_success_tasks)
            #    print("previous_best_ST",previous_best_ST)
                 
    #feval_reward,feval_success_tasks,feval_channel_success,feval_channel_collision,feval_channel_idle,feval_goodput,feval_droprate,feval_failed = test_agents(best_agents, env_id, n_episodes=n_feval_episodes, max_ep_len=max_ep_len)
    #print("feval_success_tasks",feval_success_tasks)
    return train_rewards,train_success_tasks,train_channel_success,train_channel_collision,train_channel_idle,train_goodput,train_droprate,train_failed,\
           eval_rewards,eval_success_tasks,eval_channel_success,eval_channel_collision,eval_channel_idle,eval_goodput,eval_droprate,eval_failed,train_actor_loss,train_critic_loss,train_value

# get the start time
st = time.time()

envs = ["MECSCH-v1","MECSCH-v2"]
labels = ["Proposed with Reduction","Proposed No-Reduction"]
results_mappo = {}

for env,label in zip(envs,labels):
    env_id = env
    n_episodes = 6000
    max_ep_len = 15
    n_seeds = 8
    eval_every = 50
    n_eval_episodes = 2000
    n_feval_episodes = 2000
    n_eval = np.int(n_episodes/n_feval_episodes)
    parameter_sharing = False
    disable_tqdm = False

    mappo_params = {"actor_lr":1e-2,"critic_lr":1e-4,"gamma": 0.99,"gae": True,"gae_lmb":0.95,"shuffle": False,"model":"MAPPO"}
    list_seeds = 1024 * np.arange(1, n_seeds+1)
    # Results for MAPPO 
    mappo_params["local_critic"] = True
    results_mappo[label] = Parallel(n_jobs=-1,verbose=10)(delayed(run_sim_on)(env_id,n_episodes,max_ep_len,parameter_sharing,seed, disable_tqdm, eval_every, n_eval_episodes,n_feval_episodes,**mappo_params) for seed in list_seeds.tolist())

results_mappo["Heuristics"] = Parallel(n_jobs=-1,verbose=10)(delayed(heuristics_runner)("MECSCH-v5",n_episodes=n_episodes,n_eval=n_eval,eval_every=eval_every,n_eval_episodes=n_eval_episodes,max_ep_len=max_ep_len,seed=seed) for seed in list_seeds.tolist())
results_mappo["Contention-free"] = Parallel(n_jobs=-1,verbose=10)(delayed(base_runner)("MECSCH-v6",n_episodes=n_episodes,n_eval=n_eval,eval_every=eval_every,n_eval_episodes=n_eval_episodes, max_ep_len=max_ep_len,seed=seed) for seed in list_seeds.tolist())
results_mappo["Contention-based"] = Parallel(n_jobs=-1,verbose=10)(delayed(contention_base_runner)("MECSCH-v6",n_episodes=n_episodes,n_eval=n_eval,eval_every=eval_every,n_eval_episodes=n_eval_episodes, max_ep_len=max_ep_len,seed=seed) for seed in list_seeds.tolist())
results_mappo["Round-robin"] = Parallel(n_jobs=-1,verbose=10)(delayed(Round_robin_runner)("MECSCH-v4",n_episodes=n_episodes,n_eval=n_eval,eval_every=eval_every,n_eval_episodes=n_eval_episodes,max_ep_len=max_ep_len,seed=seed) for seed in list_seeds.tolist())
results_mappo["Semi-static"] = Parallel(n_jobs=-1,verbose=10)(delayed(semi_static_runner)("MECSCH-v3",n_episodes=n_episodes,n_eval=n_eval,eval_every=eval_every,n_eval_episodes=n_eval_episodes,max_ep_len=max_ep_len,seed=seed) for seed in list_seeds.tolist())

results = [results_mappo["Proposed with Reduction"],results_mappo["Proposed No-Reduction"],results_mappo["Contention-free"],results_mappo["Contention-based"],results_mappo["Semi-static"],results_mappo["Round-robin"],results_mappo["Heuristics"]]
schemes = ["Proposed with Reduction","Proposed No-Reduction","Contention-free","Contention-based","Semi-static","Round-robin","Heuristics"]

eval_reward_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]} 
eval_success_tasks_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_channel_success_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_channel_collision_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_channel_idle_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_goodput_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_URLLC_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_droprate_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}
eval_failed_results = {"Proposed with Reduction":[],"Proposed No-Reduction":[],"Contention-free":[],"Contention-based":[],"Semi-static":[],"Round-robin":[],"Heuristics":[]}

data_0 = {"episodes":np.arange(0,n_episodes,1)} #,"Upper Bound":np.array([max_ep_len*2*n_channels]*n_episodes)


for number, result in enumerate(list_seeds):    
    Seed_0_reward = pd.DataFrame(data=data_0)
    Seed_0_Success_Task = pd.DataFrame(data=data_0)
    Seed_0_channel_success = pd.DataFrame(data=data_0) 
    Seed_0_channel_collision = pd.DataFrame(data=data_0)
    Seed_0_channel_idle = pd.DataFrame(data=data_0) 
    Seed_0_goodput = pd.DataFrame(data=data_0)
    Seed_0_droprate = pd.DataFrame(data=data_0)
    Seed_0_failed = pd.DataFrame(data=data_0)
    Seed_0_actor_loss = pd.DataFrame(data=data_0)
    Seed_0_critic_loss = pd.DataFrame(data=data_0)
    Seed_0_value = pd.DataFrame(data=data_0)
    
    Seed_1_reward = pd.DataFrame(data=data_0)
    Seed_1_Success_Task = pd.DataFrame(data=data_0)
    Seed_1_channel_success = pd.DataFrame(data=data_0) 
    Seed_1_channel_collision = pd.DataFrame(data=data_0) 
    Seed_1_channel_idle = pd.DataFrame(data=data_0) 
    Seed_1_goodput = pd.DataFrame(data=data_0)
    Seed_1_droprate = pd.DataFrame(data=data_0)
    Seed_1_failed = pd.DataFrame(data=data_0)
    Seed_1_actor_loss = pd.DataFrame(data=data_0)
    Seed_1_critic_loss = pd.DataFrame(data=data_0)
    Seed_1_value = pd.DataFrame(data=data_0)
    
    Seed_2_reward = pd.DataFrame(data=data_0)
    Seed_2_Success_Task = pd.DataFrame(data=data_0)
    Seed_2_channel_success = pd.DataFrame(data=data_0) 
    Seed_2_channel_collision = pd.DataFrame(data=data_0) 
    Seed_2_channel_idle = pd.DataFrame(data=data_0) 
    Seed_2_goodput = pd.DataFrame(data=data_0)
    Seed_2_droprate = pd.DataFrame(data=data_0)
    Seed_2_failed = pd.DataFrame(data=data_0)
    Seed_2_actor_loss = pd.DataFrame(data=data_0)
    Seed_2_critic_loss = pd.DataFrame(data=data_0)
    Seed_2_value = pd.DataFrame(data=data_0)
    
    Seed_3_reward = pd.DataFrame(data=data_0)
    Seed_3_Success_Task = pd.DataFrame(data=data_0)
    Seed_3_channel_success = pd.DataFrame(data=data_0) 
    Seed_3_channel_collision = pd.DataFrame(data=data_0) 
    Seed_3_channel_idle = pd.DataFrame(data=data_0) 
    Seed_3_goodput = pd.DataFrame(data=data_0)
    Seed_3_droprate = pd.DataFrame(data=data_0)
    Seed_3_failed = pd.DataFrame(data=data_0)
    Seed_3_actor_loss = pd.DataFrame(data=data_0)
    Seed_3_critic_loss = pd.DataFrame(data=data_0)
    Seed_3_value = pd.DataFrame(data=data_0)
    
    Seed_4_reward = pd.DataFrame(data=data_0)
    Seed_4_Success_Task = pd.DataFrame(data=data_0)
    Seed_4_channel_success = pd.DataFrame(data=data_0) 
    Seed_4_channel_collision = pd.DataFrame(data=data_0) 
    Seed_4_channel_idle = pd.DataFrame(data=data_0) 
    Seed_4_goodput = pd.DataFrame(data=data_0)
    Seed_4_droprate = pd.DataFrame(data=data_0)
    Seed_4_failed = pd.DataFrame(data=data_0)
    Seed_4_actor_loss = pd.DataFrame(data=data_0)
    Seed_4_critic_loss = pd.DataFrame(data=data_0)
    Seed_4_value = pd.DataFrame(data=data_0)
    
    Seed_5_reward = pd.DataFrame(data=data_0)
    Seed_5_Success_Task = pd.DataFrame(data=data_0)
    Seed_5_channel_success = pd.DataFrame(data=data_0) 
    Seed_5_channel_collision = pd.DataFrame(data=data_0) 
    Seed_5_channel_idle = pd.DataFrame(data=data_0) 
    Seed_5_goodput = pd.DataFrame(data=data_0)
    Seed_5_droprate = pd.DataFrame(data=data_0)
    Seed_5_failed = pd.DataFrame(data=data_0)
    Seed_5_actor_loss = pd.DataFrame(data=data_0)
    Seed_5_critic_loss = pd.DataFrame(data=data_0)
    Seed_5_value = pd.DataFrame(data=data_0)
    
    Seed_6_reward = pd.DataFrame(data=data_0)
    Seed_6_Success_Task = pd.DataFrame(data=data_0)
    Seed_6_channel_success = pd.DataFrame(data=data_0) 
    Seed_6_channel_collision = pd.DataFrame(data=data_0) 
    Seed_6_channel_idle = pd.DataFrame(data=data_0) 
    Seed_6_goodput = pd.DataFrame(data=data_0)
    Seed_6_droprate = pd.DataFrame(data=data_0)
    Seed_6_failed = pd.DataFrame(data=data_0)
    Seed_6_actor_loss = pd.DataFrame(data=data_0)
    Seed_6_critic_loss = pd.DataFrame(data=data_0)
    Seed_6_value = pd.DataFrame(data=data_0)
    
    Seed_7_reward = pd.DataFrame(data=data_0)
    Seed_7_Success_Task = pd.DataFrame(data=data_0)
    Seed_7_channel_success = pd.DataFrame(data=data_0) 
    Seed_7_channel_collision = pd.DataFrame(data=data_0) 
    Seed_7_channel_idle = pd.DataFrame(data=data_0) 
    Seed_7_goodput = pd.DataFrame(data=data_0)
    Seed_7_droprate = pd.DataFrame(data=data_0)
    Seed_7_failed = pd.DataFrame(data=data_0)
    Seed_7_actor_loss = pd.DataFrame(data=data_0)
    Seed_7_critic_loss = pd.DataFrame(data=data_0)
    Seed_7_value = pd.DataFrame(data=data_0)
    

for key, results_ind in zip(schemes, results):
    eval_reward_results[key] = np.mean([np.array(result[8]) for result in results_ind],axis=0) 
    eval_success_tasks_results[key] = np.mean([np.array(result[9]) for result in results_ind],axis=0)  
    eval_channel_success_results[key] = np.mean([np.array(result[10]) for result in results_ind],axis=0)      
    eval_channel_collision_results[key] = np.mean([np.array(result[11]) for result in results_ind],axis=0)  
    eval_channel_idle_results[key] = np.mean([np.array(result[12]) for result in results_ind],axis=0)  
    eval_goodput_results[key] = np.mean([np.array(result[13]) for result in results_ind],axis=0)
    eval_droprate_results[key] = np.mean([np.array(result[14]) for result in results_ind],axis=0)
    eval_failed_results[key] = np.mean([np.array(result[15]) for result in results_ind],axis=0)
    
    for number, result in enumerate(results_ind):
        if number == 0:
            Seed_0_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_0_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_0_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_0_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_0_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_0_goodput[key] = window_mean(np.array(result[5]),40) 
            Seed_0_droprate[key] = window_mean(np.array(result[6]),40) 
            Seed_0_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_0_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_0_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_0_value[key] = window_mean(np.array(result[18]),1) 
        elif number == 1:
            Seed_1_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_1_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_1_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_1_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_1_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_1_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_1_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_1_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_1_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_1_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_1_value[key] = window_mean(np.array(result[18]),1) 
        elif number == 2:
            Seed_2_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_2_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_2_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_2_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_2_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_2_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_2_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_2_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_2_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_2_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_2_value[key] = window_mean(np.array(result[18]),1) 
        elif number == 3:
            Seed_3_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_3_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_3_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_3_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_3_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_3_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_3_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_3_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_3_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_3_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_3_value[key] = window_mean(np.array(result[18]),1) 
        elif number == 4:
            Seed_4_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_4_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_4_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_4_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_4_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_4_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_4_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_4_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_4_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_4_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_4_value[key] = window_mean(np.array(result[18]),1) 
        elif number == 5:
            Seed_5_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_5_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_5_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_5_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_5_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_5_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_5_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_5_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_5_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_5_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_5_value[key] = window_mean(np.array(result[18]),1)
        elif number == 6:
            Seed_6_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_6_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_6_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_6_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_6_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_6_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_6_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_6_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_6_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_6_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_6_value[key] = window_mean(np.array(result[18]),1)
        elif number == 7:
            Seed_7_reward[key] = window_mean(np.array(result[0]),40) 
            Seed_7_Success_Task[key] = window_mean(np.array(result[1]),40) 
            Seed_7_channel_success[key] = window_mean(np.array(result[2]),40) 
            Seed_7_channel_collision[key] = window_mean(np.array(result[3]),40) 
            Seed_7_channel_idle[key] = window_mean(np.array(result[4]),40)    
            Seed_7_goodput[key] = window_mean(np.array(result[5]),40)
            Seed_7_droprate[key] = window_mean(np.array(result[6]),40)
            Seed_7_failed[key] = window_mean(np.array(result[7]),40) 
            Seed_7_actor_loss[key] = window_mean(np.array(result[16]),20) 
            Seed_7_critic_loss[key] = window_mean(np.array(result[17]),20) 
            Seed_7_value[key] = window_mean(np.array(result[18]),1)
 
       
train_reward = pd.concat([Seed_0_reward,Seed_1_reward,Seed_2_reward,Seed_3_reward,Seed_4_reward,Seed_5_reward,Seed_6_reward,Seed_7_reward],axis=0)
train_reward.to_csv('train_Reward_Data.csv',index=False)

train_success_tasks = pd.concat([Seed_0_Success_Task,Seed_1_Success_Task,Seed_2_Success_Task,Seed_3_Success_Task,Seed_4_Success_Task,Seed_5_Success_Task,Seed_6_Success_Task,Seed_7_Success_Task],axis=0)
train_success_tasks.to_csv('train_success_tasks_Data.csv',index=False)

train_channel_success = pd.concat([Seed_0_channel_success,Seed_1_channel_success,Seed_2_channel_success,Seed_3_channel_success,Seed_4_channel_success,Seed_5_channel_success,Seed_6_channel_success,Seed_7_channel_success],axis=0)
train_channel_success.to_csv('train_channel_success_Data.csv',index=False)

train_channel_collision = pd.concat([Seed_0_channel_collision,Seed_1_channel_collision,Seed_2_channel_collision,Seed_3_channel_collision,Seed_4_channel_collision,Seed_5_channel_collision,Seed_6_channel_collision,Seed_7_channel_collision],axis=0)
train_channel_collision.to_csv('train_channel_collision_Data.csv',index=False)

train_channel_idle = pd.concat([Seed_0_channel_idle,Seed_1_channel_idle,Seed_2_channel_idle,Seed_3_channel_idle,Seed_4_channel_idle,Seed_5_channel_idle,Seed_6_channel_idle,Seed_7_channel_idle],axis=0)
train_channel_idle.to_csv('train_channel_idle_Data.csv',index=False)

train_goodput = pd.concat([Seed_0_goodput,Seed_1_goodput,Seed_2_goodput,Seed_3_goodput,Seed_4_goodput,Seed_5_goodput,Seed_6_goodput,Seed_7_goodput],axis=0)
train_goodput.to_csv('train_goodput_Data.csv',index=False)

train_droprate = pd.concat([Seed_0_droprate,Seed_1_droprate,Seed_2_droprate,Seed_3_droprate,Seed_4_droprate,Seed_5_droprate,Seed_6_droprate,Seed_7_droprate],axis=0)
train_droprate.to_csv('train_droprate_Data.csv',index=False)

train_failed = pd.concat([Seed_0_failed,Seed_1_failed,Seed_2_failed,Seed_3_failed,Seed_4_failed,Seed_5_failed,Seed_6_failed,Seed_7_failed],axis=0)
train_failed.to_csv('train_failed_Data.csv',index=False)

train_actor_loss = pd.concat([Seed_0_actor_loss,Seed_1_actor_loss,Seed_2_actor_loss,Seed_3_actor_loss,Seed_4_actor_loss,Seed_5_actor_loss,Seed_6_actor_loss,Seed_7_actor_loss],axis=0)
train_actor_loss.to_csv('train_actor_loss_Data.csv',index=False)

train_critic_loss = pd.concat([Seed_0_critic_loss,Seed_1_critic_loss,Seed_2_critic_loss,Seed_3_critic_loss,Seed_4_critic_loss,Seed_5_critic_loss,Seed_6_critic_loss,Seed_7_critic_loss],axis=0)
train_critic_loss.to_csv('train_critic_loss_Data.csv',index=False)

train_value = pd.concat([Seed_0_value,Seed_1_value,Seed_2_value,Seed_3_value,Seed_4_value,Seed_5_value,Seed_6_value,Seed_7_value],axis=0)
train_value.to_csv('train_value_Data.csv',index=False)

eval_reward = pd.DataFrame.from_dict(eval_reward_results)
eval_reward.to_csv('eval_Reward_Data.csv',index=False)

eval_success_tasks = pd.DataFrame.from_dict(eval_success_tasks_results)
eval_success_tasks.to_csv('eval_success_tasks_Data.csv',index=False)

eval_channel_success = pd.DataFrame.from_dict(eval_channel_success_results)
eval_channel_success.to_csv('eval_channel_success_Data.csv',index=False)

eval_channel_collision = pd.DataFrame.from_dict(eval_channel_collision_results)
eval_channel_collision.to_csv('eval_channel_collision_Data.csv',index=False)

eval_channel_idle = pd.DataFrame.from_dict(eval_channel_idle_results)
eval_channel_idle.to_csv('eval_channel_idle_Data.csv',index=False)

eval_goodput = pd.DataFrame.from_dict(eval_goodput_results)
eval_goodput.to_csv('eval_goodput_Data.csv',index=False)

eval_droprate = pd.DataFrame.from_dict(eval_droprate_results)
eval_droprate.to_csv('eval_droprate_Data.csv',index=False)

eval_failed = pd.DataFrame.from_dict(eval_failed_results)
eval_failed.to_csv('eval_failed_Data.csv',index=False)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
# Plot Style:
colors = sns.color_palette(n_colors= 15).as_hex()
mpl.rcParams.update({"font.size": 15, "axes.labelsize": 15, "lines.markersize": 10})
sns.set(rc={'figure.figsize': (8,6)})
sns.set_context("notebook")
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
linestyles = ["--", "-.",":","-","--","-.",":","-","--", "-.",":","-","--","-.",":","-"]
markers = ["s","o","d","v","P","<","X","s","o","d","v","P","<","X"]
set_fonts()
#colors = set_style()
# Define the size and style of the circle to show the mean of the box plots:
meanpointprops = dict(marker='o', markeredgecolor='black',markerfacecolor='lightgray', markersize=3)


train_reward_results = pd.read_csv('train_Reward_Data.csv')
train_success_tasks_results = pd.read_csv('train_success_tasks_Data.csv')
train_channel_success_results = pd.read_csv('train_channel_success_Data.csv')
train_channel_collision_results = pd.read_csv('train_channel_collision_Data.csv')
train_channel_idle_results = pd.read_csv('train_channel_idle_Data.csv')
train_goodput_results = pd.read_csv('train_goodput_Data.csv')
train_droprate_results = pd.read_csv('train_droprate_Data.csv')
train_failed_results = pd.read_csv('train_failed_Data.csv')
train_actor_loss_results = pd.read_csv('train_actor_loss_Data.csv')
train_critic_loss_results = pd.read_csv('train_critic_loss_Data.csv')
train_value_results = pd.read_csv('train_value_Data.csv')

eval_reward_results = pd.read_csv('eval_Reward_Data.csv')
eval_success_tasks_results = pd.read_csv('eval_success_tasks_Data.csv')
eval_channel_success_results = pd.read_csv('eval_channel_success_Data.csv')
eval_channel_collision_results = pd.read_csv('eval_channel_collision_Data.csv')
eval_channel_idle_results = pd.read_csv('eval_channel_idle_Data.csv')
eval_goodput_results = pd.read_csv('eval_goodput_Data.csv')
eval_droprate_results = pd.read_csv('eval_droprate_Data.csv')
eval_failed_results = pd.read_csv('eval_failed_Data.csv')


X_axis = np.arange(0,n_episodes+1,1) 
step_plot = int(0.05 * n_episodes)
width = int(0.03*n_episodes) # Width of the box plots
n_eval_episodes = 1000

fig1 = plt.figure("Figure 1")  
data_success_tasks = []
y_max_ls =[]
i = 0
for key, value in train_reward_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_reward_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_reward_results.items():
    data_success_tasks.append(value)
for ii, test_data in enumerate(data_success_tasks):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_success_tasks) + 2
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-100, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Reward")
plt.savefig("reward.jpg",dpi=300,bbox_inches='tight')
plt.savefig("reward.eps",dpi=300,bbox_inches='tight')
plt.savefig("reward.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig2 = plt.figure("Figure 2")  
data_success_tasks = []
y_max_ls =[]
i = 0
for key, value in train_success_tasks_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_success_tasks_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_success_tasks_results.items():
    data_success_tasks.append(value)
for ii, test_data in enumerate(data_success_tasks):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_success_tasks) + 2
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Normalized Successful Tasks")
plt.savefig("task.jpg",dpi=300,bbox_inches='tight')
plt.savefig("task.eps",dpi=300,bbox_inches='tight')
plt.savefig("task.pdf",dpi=300,bbox_inches='tight')
plt.show()

fig3 = plt.figure("Figure 3")
data_channel_success = []
y_max_ls =[]
i = 0
for key, value in train_channel_success_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_channel_success_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_channel_success_results.items():
    data_channel_success.append(value)
for ii, test_data in enumerate(data_channel_success):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_success_tasks) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Channels Access Success Rate")
plt.savefig("channel_success.jpg",dpi=300,bbox_inches='tight')
plt.savefig("channel_success.eps",dpi=300,bbox_inches='tight')
plt.savefig("channel_success.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig4 = plt.figure("Figure 4")  
data_channel_collision = []
y_max_ls =[]
i = 0
for key, value in train_channel_collision_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_channel_collision_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_channel_collision_results.items():
    data_channel_collision.append(value)
for ii, test_data in enumerate(data_channel_collision):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_success_tasks) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Channels Access Collision Rate")
plt.savefig("channel_collision.jpg",dpi=300,bbox_inches='tight')
plt.savefig("channel_collision.eps",dpi=300,bbox_inches='tight')
plt.savefig("channel_collision.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig5 = plt.figure("Figure 5")  
data_channel_idle = []
y_max_ls =[]
i = 0
for key, value in train_channel_idle_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_channel_idle_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_channel_idle_results.items():
    data_channel_idle.append(value)
for ii, test_data in enumerate(data_channel_idle):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_success_tasks) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Channels Idle Rate")
plt.savefig("channel_idle.jpg",dpi=300,bbox_inches='tight')
plt.savefig("channel_idle.eps",dpi=300,bbox_inches='tight')
plt.savefig("channel_idle.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig6 = plt.figure("Figure 6")  
data_goodput = []
y_max_ls =[]
i = 0
for key, value in train_goodput_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_goodput_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_goodput_results.items():
    data_goodput.append(value)
for ii, test_data in enumerate(data_goodput):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_goodput) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Goodput")
plt.savefig("goodput.jpg",dpi=300,bbox_inches='tight')
plt.savefig("goodput.eps",dpi=300,bbox_inches='tight')
plt.savefig("goodput.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig7 = plt.figure("Figure 7")  
data_droprate = []
y_max_ls =[]
i = 0
for key, value in train_droprate_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_droprate_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_droprate_results.items():
    data_droprate.append(value)
for ii, test_data in enumerate(data_droprate):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_droprate) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("No. of Dropped Packets")
plt.savefig("drop.jpg",dpi=300,bbox_inches='tight')
plt.savefig("drop.eps",dpi=300,bbox_inches='tight')
plt.savefig("drop.pdf",dpi=300,bbox_inches='tight')
plt.show()


fig8 = plt.figure("Figure 8")  
data_droprate = []
y_max_ls =[]
i = 0
for key, value in train_failed_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_failed_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
for key, value in eval_failed_results.items():
    data_droprate.append(value)
for ii, test_data in enumerate(data_droprate):
    ymax = draw_boxplot(test_data, color=colors[ii], positions=[n_episodes + (1.5+ii)*step_plot],
                        widths=width, showfliers=False, meanprops=meanpointprops, showmeans=True, whis=1.0)
    y_max_ls.append(ymax)
end_tick = len(data_droprate) + 4
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = max(y_max_ls)
plt.ylim(-0.02, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Normalized Failed Tasks")
plt.savefig("failed.jpg",dpi=300,bbox_inches='tight')
plt.savefig("failed.eps",dpi=300,bbox_inches='tight')
plt.savefig("failed.pdf",dpi=300,bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
values = []
for key, value in eval_droprate_results.items():
    values.append(np.mean(value))    
fig , ax = plt.subplots()
ax.bar(schemes,values, color=colors)
plt.xticks(rotation=-70)
ax.set_xlabel('Schemes')
ax.set_ylabel("No. of Dropped Packets")
plt.savefig("drop.pdf",dpi=1500,bbox_inches='tight')
plt.show()

values = []
for key, value in eval_channel_idle_results.items():
    values.append(np.mean(value))    
fig , ax = plt.subplots()
ax.bar(schemes,values, color=colors)
plt.xticks(rotation=-70)
ax.set_xlabel('Schemes')
ax.set_ylabel("Channels Idle Rate")
plt.savefig("idle.pdf",dpi=300,bbox_inches='tight')
plt.savefig("idle.eps",dpi=300,bbox_inches='tight')
plt.show()

values = []
for key, value in eval_channel_collision_results.items():
    values.append(np.mean(value))    
fig , ax = plt.subplots()
ax.bar(schemes,values, color=colors)
plt.xticks(rotation=-70)
ax.set_xlabel('Schemes')
ax.set_ylabel("Channels Access Collision Rate")
plt.savefig("collision.pdf",dpi=300,bbox_inches='tight')
plt.savefig("collision.eps",dpi=300,bbox_inches='tight')
plt.savefig("collision.jpg",dpi=300,bbox_inches='tight')
plt.show()



fig1 = plt.figure("Figure 9")  
data_success_tasks = []
y_max_ls =[]
i = 0
for key, value in train_actor_loss_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_actor_loss_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = 0
plt.ylim(-1, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Actor Loss")
plt.savefig("actor_loss.jpg",dpi=300,bbox_inches='tight')
plt.show()


fig1 = plt.figure("Figure 10")  
data_success_tasks = []
y_max_ls =[]
i = 0
for key, value in train_critic_loss_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_critic_loss_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = 500
plt.ylim(-100, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Critic Loss")
plt.savefig("critic_loss.jpg",dpi=300,bbox_inches='tight')
plt.show()



fig1 = plt.figure("Figure 11")  
data_success_tasks = []
y_max_ls =[]
i = 0
for key, value in train_value_results.iloc[:,1:10].items():
    g = sns.lineplot(data=train_value_results,x='episodes',y=key,label=key,linestyle=linestyles[i],color=colors[i],marker=markers[i],
                     markevery=n_eval_episodes, markersize=5)
    i +=1    
g.axes.xaxis.set_major_formatter(ticker.EngFormatter())
plt.axvline(n_episodes + 0.7*step_plot, color='k', linestyle="--")
plt.draw()
locs, labels = plt.xticks() 
plt.xticks(locs[1:-1], labels[1:-1])
ylim_ = 100
plt.ylim(-100, ylim_)
draw_brace(g.axes, [0, n_episodes], ylim_, "Train")
draw_brace(g.axes, [n_episodes + 0.8*step_plot, n_episodes + end_tick*step_plot], ylim_, "Test")
x_min, _ = g.axes.get_xlim()
plt.xlim(x_min, n_episodes + end_tick*step_plot)
plt.legend(ncol=2, frameon=True, shadow=True,bbox_to_anchor=(0.95,1.45))
plt.xlabel('Episodes')
plt.ylabel("Action Value")
plt.savefig("action_value.jpg",dpi=300,bbox_inches='tight')
plt.show()