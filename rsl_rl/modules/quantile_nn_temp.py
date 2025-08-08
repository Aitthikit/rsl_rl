# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import warnings
import torch
import torch.nn as nn
from rsl_rl.networks import Memory
from torch.distributions import Normal
from typing import List,Union,Tuple

from rsl_rl.utils import resolve_nn_activation

eps = torch.finfo(torch.float32).eps

def squeeze_preserve_batch(tensor):
    """Squeezes a tensor, but preserves the batch dimension"""
    single_batch = tensor.shape[0] == 1

    squeezed_tensor = tensor.squeeze()

    if single_batch:
        squeezed_tensor = squeezed_tensor.unsqueeze(0)

    return squeezed_tensor



def reshape_measure_parameters(
    qn: QuantileCritic, *params: Union[torch.Tensor, float]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Reshapes the parameters of a measure function to match the shape of the quantile network.

    Args:
        qn (Network): The quantile network.
        *params (Union[torch.Tensor, float]): The parameters of the measure function.
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The reshaped parameters.
    """
    if not params:
        return qn._tau.to(qn.device), *params

    assert len([*set([torch.is_tensor(p) for p in params])]) == 1, "All parameters must be either tensors or scalars."

    if torch.is_tensor(params[0]):
        assert all([p.dim() == 1 for p in params]), "All parameters must have dimensionality 1."
        assert len([*set([p.shape[0] for p in params])]) == 1, "All parameters must have the same size."

        reshaped_params = [p.reshape(-1, 1).to(qn.device) for p in params]
        tau = qn._tau.expand(params[0].shape[0], -1).to(qn.device)
    else:
        reshaped_params = params
        tau = qn._tau.to(qn.device)

    return tau, *reshaped_params


def make_distorted_measure(distorted_tau: torch.Tensor) -> Callable:
    """Creates a measure function for the distorted expectation under some distortion function.

    The distorted expectation for some distortion function g(tau) is given by the integral w.r.t. tau
    "int_0^1 g'(tau) * F_Z^{-1}(tau) dtau" where g'(tau) is the derivative of g w.r.t. tau and F_Z^{-1} is the inverse
    cumulative distribution function of the value distribution.
    See https://arxiv.org/pdf/2004.14547.pdf and https://arxiv.org/pdf/1806.06923.pdf for details.
    """
    distorted_tau = distorted_tau.reshape(-1, distorted_tau.shape[-1])
    distortion = (distorted_tau[:, 1:] - distorted_tau[:, :-1]).squeeze(0)

    def distorted_measure(quantiles):
        sorted_quantiles, _ = quantiles.sort(-1)
        sorted_quantiles = sorted_quantiles.reshape(-1, sorted_quantiles.shape[-1])

        # dtau = tau[1:] - tau[:-1] cancels the denominator of g'(tau) = g(tau)[1:] - g(tau)[:-1] / dtau.
        values = squeeze_preserve_batch((distortion.to(sorted_quantiles.device) * sorted_quantiles).sum(-1))

        return values

    return distorted_measure

#TODO = "Need to fix beta parameter handling" 

def risk_measure_wang(qn: QuantileCritic, beta: Union[float, torch.Tensor] = 0.0) -> Callable:
    """Wang's risk measure.

    The risk measure computes the distorted expectation under Wang's risk distortion function
    g(tau) = Phi(Phi^-1(tau) + beta) where Phi and Phi^-1 are the standard normal CDF and its inverse.
    See https://arxiv.org/pdf/2004.14547.pdf for details.

    Args:
        qn (QuantileNetwork): Quantile network to compute the risk measure for.
        beta (float): Parameter of the risk distortion function.
    Returns:
        A risk measure function.
    """
    tau, beta = reshape_measure_parameters(qn, beta)

    distorted_tau = Normal(0, 1).cdf(Normal(0, 1).icdf(tau) + beta)

    return make_distorted_measure(distorted_tau)


def energy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes sample energy loss between predictions and targets.

    The energy loss is computed as 2*E[||X - Y||_2] - E[||X - X'||_2] - E[||Y - Y'||_2], where X, X' and Y, Y' are
    random variables and ||.||_2 is the L2-norm. X, X' are the predictions and Y, Y' are the targets.

    Args:
        predictions (torch.Tensor): Predictions to compute loss from.
        targets (torch.Tensor): Targets to compare predictions against.
    Returns:
        A torch.Tensor of shape (1,) containing the loss.
    """
    dims = [-1 for _ in range(predictions.dim())]
    prediction_mat = predictions.unsqueeze(-1).expand(*dims, predictions.shape[-1])
    target_mat = targets.unsqueeze(-1).expand(*dims, predictions.shape[-1])

    delta_xx = (prediction_mat - prediction_mat.transpose(-1, -2)).abs().mean()
    delta_yy = (target_mat - target_mat.transpose(-1, -2)).abs().mean()
    delta_xy = (prediction_mat - target_mat.transpose(-1, -2)).abs().mean()

    loss = 2 * delta_xy - delta_xx - delta_yy

    return loss

class QuantileCritic(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims=[256, 256, 256],
            quantile_count=200,
            recurrent_layers=1,
            activations=[nn.ReLU, nn.ReLU, nn.ReLU],
            init_fade=False,
            init_gain=0.5,
            measure_kwargs={},
            ):
        
        super().__init__()
        # self.num_quantiles = num_quantiles
        # layers = [nn.Linear(input_dim, hidden_dims[0]), activation()]
        # for i in range(len(hidden_dims) - 1):
        #     layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        #     layers.append(activation())
        # layers.append(nn.Linear(hidden_dims[-1], num_quantiles))
        # self.net = nn.Sequential(*layers)
        self._normalization = nn.Identity()

        dims = [input_dim] + hidden_dims + [output_dim]
        self._recurrent = True
        self.hidden_state = None
        self._last_hidden_state = None
        recurrent_kwargs = dict()
        recurrent_kwargs["hidden_size"] = dims[1]
        recurrent_kwargs["input_size"] = dims[0]
        recurrent_kwargs["num_layers"] = recurrent_layers

        rnn = nn.LSTM(**recurrent_kwargs)
        activation = activations[0]
        dims = dims[1:]
        activations = activations[1:]

        self._features = nn.Sequential(rnn, activation)
        # else:
        #     self._features = nn.Identity()

        layers = []
        for i in range(len(activations)):
            layer = nn.Linear(dims[i], dims[i + 1])
            activation = activations[i]
            layers.append(layer)
            layers.append(activation)
        print(f"QuantileCritic MLP: {layers}")

        self._layers = nn.Sequential(*layers)

        if len(layers) > 0:
            self._init(self._layers, fade=init_fade, gain=init_gain)
        
        self._quantile_count = quantile_count
        self._tau = torch.arange(self._quantile_count + 1) / self._quantile_count
        self._tau_hat = torch.tensor([(self._tau[i] + self._tau[i + 1]) / 2 for i in range(self._quantile_count)])
        self._tau_hat_mat = torch.empty((0,))

        self._quantile_layers = nn.ModuleList([nn.Linear(hidden_dims[-1], quantile_count) for _ in range(output_dim)])

        self._init(self._quantile_layers, fade=init_fade, gain=init_gain)

        measure_func = risk_measure_wang 
        self._measure_func = measure_func
        self._measure = measure_func(self, **measure_kwargs)

        self._last_quantiles = None

    @property
    def device(self):
        """Returns the device of the network."""
        return next(self.parameters()).device
    
    @property
    def last_quantiles(self) -> torch.Tensor:
        return self._last_quantiles
    
    def make_diracs(self, values: torch.Tensor) -> torch.Tensor:
        """Generates value distributions that have a single spike at the given values.

        Args:
            values (torch.Tensor): Values to generate dirac distributions for.
        Returns:
            A torch.Tensor of shape (*values.shape, quantile_count) containing the dirac distributions.
        """
        dirac = values.unsqueeze(-1).expand(*[-1 for _ in range(values.dim())], self._quantile_count)

        return dirac
    
    @property
    def quantile_count(self) -> int:
        return self._quantile_count
    


    #TODO = "need to fix hidden state handling"
    # def forward(self, x):
    #     return self.net(x)  # output shape: (batch_size, num_quantiles)
    def forward(self, x: torch.Tensor,hidden_state=None, distribution: bool = False, measure_args: list = [], **kwargs) -> torch.Tensor:
    # def forward(self, x: torch.Tensor, hidden_state=None) -> torch.Tensor:
        assert hidden_state is None or self._recurrent, "Cannot pass hidden state to non-recurrent network."

        input = self._normalization(x.to(self.device))

        if self._recurrent:
            current_hidden_state = self.hidden_state if hidden_state is None else hidden_state
            current_hidden_state = (current_hidden_state[0].to(self.device), current_hidden_state[1].to(self.device))

            input = input.unsqueeze(0) if len(input.shape) == 2 else input
            input, next_hidden_state = self._features[0](input, current_hidden_state)
            input = self._features[1](input).squeeze(0)

            if hidden_state is None:
                self.hidden_state = next_hidden_state
            self._last_hidden_state = next_hidden_state

        features = squeeze_preserve_batch(self._layers(input))
        quantiles = squeeze_preserve_batch(torch.stack([layer(features) for layer in self._quantile_layers], dim=1))

        self._last_quantiles = quantiles

        if distribution:
            return quantiles

        values = self.quantiles_to_values(quantiles, *measure_args)

        return values
    
    #TODO = "need to fix measure_args handling"
    
    def quantiles_to_values(self, quantiles: torch.Tensor, *measure_args) -> torch.Tensor:
        """Computes values from quantiles.

        Args:
            quantiles (torch.Tensor): Quantiles to compute values from.
            measure_kwargs (dict): Keyword arguments to pass to the risk measure function instead of the arguments
                passed when creating the network.
        Returns:
            A torch.Tensor of shape (1,) containing the values.
        """
        if measure_args:
            values = self._measure_func(self, *[squeeze_preserve_batch(m) for m in measure_args])(quantiles)
        else:
            values = self._measure(quantiles)

        return values
    
    #TODO = "need to fix reset hidden state to move it in the memory class"

    def reset_hidden_state(self, indices: torch.Tensor) -> None:
        """Resets the hidden state of the neural network.

        Throws an error if the network is not recurrent.

        Args:
            indices (torch.Tensor): A 1-dimensional int tensor containing the indices of the terminated
                environments.
        """
        assert self._recurrent

        self.hidden_state[0][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)
        self.hidden_state[1][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)

    def reset_full_hidden_state(self, batch_size=None) -> None:
        """Resets the hidden state of the neural network.

        Args:
            batch_size (int): The batch size of the hidden state. If None, the hidden state is reset to None.
        """
        assert self._recurrent

        if batch_size is None:
            self.hidden_state = None
        else:
            layer_count, hidden_size = self._features[0].num_layers, self._features[0].hidden_size
            self.hidden_state = (
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
            )

    def _init(self, layers: List[nn.Module], fade: bool = True, gain: float = 1.0) -> List[nn.Module]:
        """Initializes neural network layers."""
        last_layer_idx = len(layers) - 1 - next(i for i, l in enumerate(reversed(layers)) if isinstance(l, nn.Linear))

        for idx, layer in enumerate(layers):
            if not isinstance(layer, nn.Linear):
                continue

            current_gain = gain / 100.0 if fade and idx == last_layer_idx else gain
            nn.init.xavier_normal_(layer.weight, gain=current_gain)

        return layers


class Quantile_NN(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        measure_kwargs: dict = {},
        quantile_count = 200,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # num_actor_obs=rnn_hidden_dim
        # num_critic_obs=rnn_hidden_dim
        # num_actions=num_actions
        # actor_hidden_dims=actor_hidden_dims
        # critic_hidden_dims=critic_hidden_dims
        # activation=activation
        # init_noise_std=init_noise_std
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = rnn_hidden_dim
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        ##############################################################################
        # critic_layers = []
        # critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        # critic_layers.append(activation)
        # for layer_index in range(len(critic_hidden_dims)):
        #     if layer_index == len(critic_hidden_dims) - 1:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
        #     else:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
        #         critic_layers.append(activation)
        # self.critic = nn.Sequential(*critic_layers)
        # self.critic = QuantileCritic(
        #     input_dim=mlp_input_dim_c,
        #     hidden_dims=critic_hidden_dims,
        #     num_quantiles=quantile_count,
        #     activation = nn.ELU
        #     )
        self.critic = QuantileCritic(
            input_dim=mlp_input_dim_c,
            output_dim=1, # output_dim is the number of quantiles??
            hidden_dims=critic_hidden_dims,
            quantile_count=quantile_count,
            recurrent_layers=rnn_num_layers,
            activations=[activation] * len(critic_hidden_dims),
            init_fade=False,
            init_gain=0.5,
            measure_kwargs=measure_kwargs,
        )
        ###############################################################################
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        print(f"Actor RNN: {self.memory_a}")

        # self._quantile_count = quantile_count
        # self._tau = torch.arange(self._quantile_count+1)/self._quantile_count
        # self._tau_hat = torch.tensor([(self._tau[i] + self._tau[i + 1]) / 2 for i in range(self._quantile_count)])
        # self._tau_hat_mat = torch.empty((0,))

        # measure_func = risk_measure_wang
        # self._measure_function = measure_func
        # self._measure = measure_func(self, **measure_kwargs)

        # self._last_quantiles = None

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        # pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act_b(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return self.act_b(input_a.squeeze(0))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference_b(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return self.act_inference_b(input_a.squeeze(0))


    def evaluate(self, critic_observations, hidden_state=None, **kwargs):
        value = self.critic(critic_observations,hidden_state)
        # print(f"Critic MLP output: {value.shape}")
        return value

    # def evaluate(self, critic_observations, **kwargs):
    #     quantiles = self.critic(critic_observations)  # shape: (B, N)
    #     # self._last_quantiles = quantiles
    #     return self._measure(quantiles)  # scalar value using distorted expectation

    def evaluate_quantiles(self, critic_observations):
        return self.critic(critic_observations,distribution=True)  # shape: (B, N, quantile_count)


    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states,None#self.critic.hidden_state
