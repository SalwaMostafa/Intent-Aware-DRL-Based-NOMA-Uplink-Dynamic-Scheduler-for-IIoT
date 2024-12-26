from maa2c import MAA2C
from mappo import MAPPO
from envs.mecsch_v1 import MECSCHEnvV1
from envs.mecsch_v2 import MECSCHEnvV2
from on_buffer import OnPolicyBuffer
import numpy as np

class OnPolicyWrapper:
    def __init__(self,env,model="MAA2C",**kwargs):
        if model == "MAA2C":
            algo = MAA2C
        elif model == "MAPPO":
            algo = MAPPO
        else:
            raise RuntimeError("Unknown or Unspecified RL algorithm.")

        # Buffer Params:
        gae = kwargs.get("gae", False)
        standardize_rewards = kwargs.get("standardize_rewards", False)
        shuffle = kwargs.get("shuffle", False)
        gae_lmb = kwargs.get("gae_lmb", 0.5)
        gamma = kwargs.get("gamma", 0.90)

        # Algo Params and internal params
        parameter_sharing = kwargs.get("parameter_sharing", True)
        self.n_agents = env.n_agents
        self.parameter_sharing = parameter_sharing
        self.local_critic = kwargs.get("local_critic", False)
        self.recurrent = False
        self.mecsch = "MECSCH" in env.spec.id

        # Generate Agents:
        if parameter_sharing and self.mecsch:
            agents = []
            agent_bs = algo(env, 0, **kwargs)
            agent_ue = algo(env, 1, **kwargs)
            agents = [agent_bs, agent_ue]
        else:
            agents = [algo(env, idx, **kwargs) for idx in range(self.n_agents)]
        self.agents = agents

        # Build Replay buffers:
        self.memories = self._build_memory(gamma, gae, gae_lmb, standardize_rewards, shuffle)

    def _build_memory(self, gamma, gae, gae_lmb, standardize_rewards, shuffle):
        memories = [OnPolicyBuffer(gamma=gamma,gae=gae,gae_lmb=gae_lmb,standardize_rewards=standardize_rewards,
                                   shuffle=shuffle) for _ in range(self.n_agents)]
        return memories

    def reset(self):
        pass

    def act(self, state, explore=True):
        if self.parameter_sharing and self.mecsch:
            actions = []
            actions.append(self.agents[0].act(np.array(state[0]), explore))
            actions.extend(self.agents[1].act(np.array(state[1:]), explore))
        else:
            actions = [agent.act(state[ii], explore) for ii, agent in enumerate(self.agents)]
        return actions

    def estimate_value(self, state_n):
        value_n = []
        if self.local_critic:
            s = state_n
        else:
            s = [np.concatenate(state_n)] * self.n_agents

        if self.parameter_sharing and self.mecsch:
            value_n.append(self.agents[0].estimate_value(s[0]))
            value_n.extend(self.agents[1].estimate_value(np.array(s[1:])))
        else:
            for ii, ag in enumerate(self.agents):
                value_n.append(ag.estimate_value(s[ii]))

        return value_n

    def experience(self, episode_count, obs, actions, rewards, new_obs, dones, values):
        for ii, memory in enumerate(self.memories):
            memory.push(observation=obs[ii],
                        observation_n=obs,
                        action=actions[ii],
                        action_n=actions,
                        reward=rewards[ii],
                        next_observation=new_obs[ii],
                        next_observation_n=new_obs,
                        done=dones[ii],
                        value=values[ii])

    def update(self, next_values):
        v_losses = {}
        pi_losses = {}

        # Get batches from all agents:
        batches = [memory.sample(next_values[ii]) for ii, memory in enumerate(self.memories)]
        # If sharing parameters, need to concat the batches:
        for ii, agent in enumerate(self.agents):
            if self.parameter_sharing and self.mecsch:
                if ii == 0:
                    batch = batches[ii]
                else:
                    batch = self.memories[1].concatenate_batches(batches[1:])
                v_losses[ii], pi_losses[ii] = self.agents[ii].update(batch)
            else:
                batch = batches[ii]
                v_losses[ii], pi_losses[ii] = agent.update(batch)

        # Empty buffer
        self.reset_memories()

        return v_losses, pi_losses

    def reset_memories(self):
        for memory in self.memories:
            memory.reset()

    def clear_memory(self):
        del self.memory
        self.memory = None

    def load_state_dict(self, ls_dicts):
        for ii, ag in enumerate(self.agents):
            ag.load_actor_state(ls_dicts[ii]["state_dict"])

    def update_lr(self, fraction):
        for ii, ag in enumerate(self.agents):
            ag.update_lr(fraction)
