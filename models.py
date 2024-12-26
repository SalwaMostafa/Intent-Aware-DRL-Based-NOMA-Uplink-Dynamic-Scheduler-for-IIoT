import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


#####################
# #### BASE NET ######
# ####################


class BaseMLPNet(nn.Module):
    """
    Base Class for simple feedforward Network Modules (MLP).
    The Other classes need to implement the forward and other necessary methods.

    It implements D2RL (Skip Connections).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func="relu",
        d2rl=False,
    ):
        super().__init__()
        # Save important parameters:
        self.d2rl = d2rl
        self.activation_func = torch.tanh if activation_func == "tanh" else F.relu

        # D2RL or 2 hidden layers with Recurrent layer being the second:
        if self.d2rl:
            hidden_in = input_size + hidden_size
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_in, hidden_size)
            self.linear3 = nn.Linear(hidden_in, hidden_size)
            self.linear4 = nn.Linear(hidden_in, hidden_size)
            self.linearOut = nn.Linear(hidden_size, output_size)
        else:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linearOut = nn.Linear(hidden_size, output_size)

    def _get_output(self, x_in):
        """
        Param state is torch tensors
        """
        x = self.activation_func(self.linear1(x_in))
        if self.d2rl:
            # Hidden layer 2:
            x = torch.cat([x, x_in], dim=-1)
            x = self.activation_func(self.linear2(x))
            # Hidden Layer 3:
            x = torch.cat([x, x_in], dim=-1)
            x = self.activation_func(self.linear3(x))
            # Hidden Layer 4:
            x = torch.cat([x, x_in], dim=-1)
            x = self.activation_func(self.linear4(x))
        else:
            x = self.activation_func(self.linear2(x))

        x = self.linearOut(x)
        return x


######################
# ##### CRITICS #######
# #####################


class MLPValueCritic(BaseMLPNet):
    """
    State-Value function estimator. Aka, Value-Function estimator.
    Example: V(s) = 29.12
    """

    def __init__(self, input_size, hidden_size, activation_func="relu", d2rl=False):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            activation_func=activation_func,
            d2rl=d2rl,
        )

    def forward(self, state):
        value = self._get_output(state)
        return value


class MLPActionCritic(BaseMLPNet):
    """
    Q-Value estimator. Models the Action-Value Function
    Example: Q(s,a) = 30.12
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        activation_func="relu",
        d2rl=False,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            activation_func=activation_func,
            d2rl=d2rl,
        )

    def forward(self, state, action):
        # Concat in data dimension:
        x_in = torch.cat([state, action], 1)
        value = self._get_output(x_in)
        return value


########################################
# ############## ACTORS #################
# #######################################


class BaseMLPActor(BaseMLPNet):
    """
    Base class for MLP actors.
    Child classes should define the distribution function and other details
    It defaults to a single distribution action space
    Multidistributions such as MultiDiscrete action spaces will rewrite some methods
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dist_function,
        activation_func="relu",
        d2rl=False,
        masking=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation_func=activation_func,
            d2rl=d2rl,
        )
        self.dist_function = dist_function
        self.masking = masking

    def forward(self, state, action_mask=None):
        logits = self._get_output(state)
        if self.masking:
            logits[action_mask == 0] = -1e10
        return logits

    def _get_distributions(self, state, action_mask=None):
        logits = self.forward(state, action_mask)
        dists = self.dist_function(logits=logits)
        return dists

    def act(self, state, action_mask=None, explore=False, **kwargs):
        dist = self._get_distributions(state, action_mask)
        if explore:
            action = dist.sample()
        else:
            action = dist.probs.argmax()
        return action

    def get_log_prob_entropy(self, state, action, action_mask=None):
        # Get distribution:
        dist = self._get_distributions(state, action_mask)
        # Get Log Prob and Entropy
        entropy = dist.entropy()
        log_prob = dist.log_prob(action)

        return log_prob, entropy


class MLPCategoricalOneHotActor(BaseMLPActor):
    """
    Actor which outputs a OneHotCategorical Action:
    act(obs) = [0, 0, 1, 0]

    If an index is needed, use argmax.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func="relu",
        masking=False,
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.OneHotCategorical,
            activation_func=activation_func,
            d2rl=d2rl,
            masking=masking,
        )

    def act(self, state, action_mask=None, explore=False, **kwargs):
        dist = self._get_distributions(state, action_mask)
        if explore:
            action = dist.sample()
        else:
            action = dist.probs
        return action


class MLPCategoricalActor(BaseMLPActor):
    """
    Actor which outputs a Categorical Action:
    act(obs) = 4
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func="relu",
        masking=False,
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.Categorical,
            activation_func=activation_func,
            d2rl=d2rl,
            masking=masking,
        )

    def get_log_prob_entropy(self, state, action, action_mask=None):
        # Get distribution:
        dist = self._get_distributions(state, action_mask)
        # Get Log Prob and Entropy
        entropy = dist.entropy()
        log_prob = dist.log_prob(action.reshape(-1))

        return log_prob, entropy


class MLPMultiCategoricalOneHotActor(BaseMLPActor):
    """
    Actor which outputs a MultiDiscrete One Hotted Action:
    act(obs) = [[0, 1], [0, 0, 1, 0]]
    If an index is needed, use argmax.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        action_split,
        activation_func="relu",
        masking=False,
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.OneHotCategorical,
            activation_func=activation_func,
            d2rl=d2rl,
            masking=masking,
        )
        self.action_split = action_split

    def get_log_prob_entropy(self, state, action, action_mask=None):
        # Get distributions of each action group:
        split_actions = torch.split(action, self.action_split, dim=-1)
        dists = self._get_distributions(state, action_mask)
        # Get Log Prob and Entropy
        entropy = torch.stack([dist.entropy() for dist in dists]).sum(0).unsqueeze(-1)
        log_prob = (
            torch.stack(
                [dist.log_prob(action) for dist, action in zip(dists, split_actions)]
            )
            .sum(0)
            .unsqueeze(-1)
        )

        return log_prob, entropy

    def _get_distributions(self, state, action_mask=None):
        logits = self.forward(state, action_mask)

        # Check if it came flattened or a batch
        if logits.ndim < 2:
            logits = logits.unsqueeze(0)  # add batch dim

        split_logits = torch.split(logits, self.action_split, dim=-1)
        dists = [
            self.dist_function(probs=torch.softmax(l, dim=1)) for l in split_logits
        ]
        return dists

    def act(self, state, action_mask=None, explore=False, **kwargs):
        flatten_flag = False
        if state.ndim < 2:
            flatten_flag = True
        dists = self._get_distributions(state, action_mask)
        if explore:
            actions = [d.sample() for d in dists]
        else:
            actions = [d.probs for d in dists]
        output = torch.cat(actions, dim=-1)
        if flatten_flag:
            output = output.flatten()
        return output


class MLPMultiCategoricalActor(BaseMLPActor):
    """
    Actor which outputs a Categorical Action: [4].
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        action_split,
        activation_func="relu",
        d2rl=False,
        masking=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.Categorical,
            activation_func=activation_func,
            d2rl=d2rl,
            masking=masking,
        )
        self.action_split = action_split
        print(input_size,hidden_size,output_size)

    def _get_distributions(self, state, action_mask=None):
        logits = self.forward(state, action_mask)

        # Check if it came flattened or a batch
        if logits.ndim < 2:
            logits = logits.unsqueeze(0)  # add batch dim

        split_logits = torch.split(logits, self.action_split, dim=-1)
        dists = [
            self.dist_function(probs=torch.softmax(l, dim=1)) for l in split_logits
        ]
        return dists

    def get_log_prob_entropy(self, state, action, action_mask=None):
        # Get distributions of each action group:
        # split_actions = torch.split(action, self.action_split, dim=-1)
        dists = self._get_distributions(state, action_mask)
        # Get Log Prob and Entropy
        entropy = torch.stack([dist.entropy() for dist in dists]).sum(0).unsqueeze(-1)
        log_prob = (torch.stack([dist.log_prob(ac) for dist, ac in zip(dists, action.T)]).sum(0).unsqueeze(-1))
        return log_prob, entropy

    def act(self, state, action_mask=None, explore=False, **kwargs):
        dists = self._get_distributions(state, action_mask)
        if explore:
            actions = torch.stack([d.sample() for d in dists]).T
        else:
            actions = torch.stack([torch.argmax(d.probs, -1) for d in dists]).T
        # return torch.cat(actions, dim=-1)
        return actions


class MLPMultiBinaryActor(BaseMLPActor):
    """
    Get a MultiBinary Actor.
    Each distribution produces an action such as: [0, 1, 0, 0, 1, 1]
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        action_split,
        activation_func="relu",
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.Bernoulli,
            activation_func=activation_func,
            d2rl=d2rl,
        )
        self.action_split = action_split

class MLPRelaxedMultiCategoricalActor(BaseMLPActor):
    """
    Get a Gumbel-Softmax distribution.
    We rewrite the act and get_dist due to the temperature parameter.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        action_split,
        activation_func="relu",
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.RelaxedOneHotCategorical,
            activation_func=activation_func,
            d2rl=d2rl,
        )
        self.action_split = action_split

    def _get_distributions(self, state, action_mask=None, temp=1.0):
        logits = self.forward(state, action_mask)
        split_logits = torch.split(logits, self.action_split, dim=-1)
        dists = [self.dist_function(temp, logits=l) for l in split_logits]
        return dists

    def act(self, state, action_mask=None, explore=False, temp=1.0, **kwargs):
        dists = self._get_distributions(state, action_mask=action_mask, temp=temp)
        if explore:
            actions = [d.sample() for d in dists]
        else:
            actions = [d.probs for d in dists]
        return torch.cat(actions, dim=-1)

class MLPRelaxedCategoricalActor(BaseMLPActor):
    """
    Get a Gumbel-Softmax distribution.
    We rewrite the act and get_dist due to the temperature parameter.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func="relu",
        d2rl=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dist_function=torch.distributions.RelaxedOneHotCategorical,
            activation_func=activation_func,
            d2rl=d2rl,
        )

    def _get_distributions(self, state, action_mask=None, temp=1.0):
        logits = self.forward(state, action_mask)
        dists = self.dist_function(temp, logits=logits)
        return dists

    def act(self, state, action_mask=None, explore=False, temp=1.0, **kwargs):
        dist = self._get_distributions(state, action_mask=action_mask, temp=temp)
        if explore:
            actions = dist.sample()
        else:
            actions = dist.probs
        return actions

class MLPDPGActor(BaseMLPNet):
    """
    Actor which outputs a continuous action directly, without sampling from a
    distribution.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func="relu",
        d2rl=False,
        masking=False,
        action_limit=1,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation_func=activation_func,
            d2rl=d2rl,
        )
        self.masking = masking
        self.action_limit = action_limit

    def forward(self, state, action_mask=None):
        linear_out = self._get_output(state)
        output = torch.tanh(linear_out) * self.action_limit
        return output

    def act(self, state, action_mask=None, explore=False, **kwargs):
        action = self.forward(state, action_mask=action_mask)
        return action
