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
from typing import List, Union, Tuple, Callable


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
    qn, *params: Union[torch.Tensor, float]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Reshapes the parameters of a measure function to match the shape of the quantile network."""
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
    """Creates a measure function for the distorted expectation under some distortion function."""
    distorted_tau = distorted_tau.reshape(-1, distorted_tau.shape[-1])
    distortion = (distorted_tau[:, 1:] - distorted_tau[:, :-1]).squeeze(0)

    def distorted_measure(quantiles):
        sorted_quantiles, _ = quantiles.sort(-1)
        # print(f"Sorted quantiles: {sorted_quantiles}")
        # sorted_quantiles = sorted_quantiles.reshape(-1, sorted_quantiles.shape[-1])
        # print(f"Sort_quantile: {sorted_quantiles.shape}")
        values = squeeze_preserve_batch((distortion.to(sorted_quantiles.device) * sorted_quantiles).sum(-1))
        # print(f"Distorted measure values: {values.shape}")
        return values.unsqueeze(-1)

    return distorted_measure

def risk_measure_wang(qn, beta: Union[float, torch.Tensor] = 0.0) -> Callable:
    """Wang's risk measure."""
    tau, beta = reshape_measure_parameters(qn, beta)
    distorted_tau = Normal(0, 1).cdf(Normal(0, 1).icdf(tau) + beta)
    return make_distorted_measure(distorted_tau)

def energy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes sample energy loss between predictions and targets."""
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
            output_dim=1,  # This should be 1 for value function
            hidden_dims=[256, 256, 256],
            quantile_count=200,
            recurrent_layers=1,
            activations=[nn.ReLU, nn.ReLU, nn.ReLU, nn.Tanh],
            init_fade=False,
            init_gain=0.5,
            measure_kwargs={},
    ):
        super().__init__()
        
        self._normalization = nn.Identity()
        self._recurrent = recurrent_layers > 0
        self.hidden_state = None
        
        if self._recurrent:
            # Use Memory class for RNN handling
            self.memory = Memory(input_dim, type="lstm", num_layers=recurrent_layers, hidden_size=hidden_dims[0])
            mlp_input_dim = hidden_dims[0]
        else:
            self.memory = None
            mlp_input_dim = input_dim
        
        # Build MLP layers after RNN
        layers = []
        dims = [mlp_input_dim] + hidden_dims
        # for i in range(len(dims) - 1):
        #     layers.append(nn.Linear(dims[i], dims[i + 1]))
        #     if i < len(activations):
        #         layers.append(activations[i])
        for i in range(len(activations)):
            layer = nn.Linear(dims[i], dims[i + 1])
            activation = activations[i]
            layers.append(layer)
            layers.append(activation)
        
        self._layers = nn.Sequential(*layers)
        
        # Quantile output layers - one for each output dimension
        self._quantile_layers = nn.ModuleList([
            nn.Linear(hidden_dims[-1], quantile_count) for _ in range(output_dim)
        ])

        
        if len(layers) > 0:
            self._init(self._layers, fade=init_fade, gain=init_gain)
        self._init(self._quantile_layers, fade=init_fade, gain=init_gain)
        
        self._quantile_count = quantile_count
        self._tau = torch.arange(self._quantile_count + 1) / self._quantile_count
        self._tau_hat = torch.tensor([(self._tau[i] + self._tau[i + 1]) / 2 for i in range(self._quantile_count)])
        
        # Risk measure
        measure_func = risk_measure_wang 
        self._measure_func = measure_func
        self._measure = measure_func(self, **measure_kwargs) #use function in make_distortes_measure
        
        self._last_quantiles = None

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def last_quantiles(self) -> torch.Tensor:
        return self._last_quantiles
    
    @property
    def quantile_count(self) -> int:
        return self._quantile_count

    def forward(self, x: torch.Tensor, masks=None, hidden_states=None, distribution: bool = False, measure_args: list = [], **kwargs) -> torch.Tensor:
        input = self._normalization(x.to(self.device)) # TODO: IS this need?
        if self._recurrent and self.memory is not None:
            # Use Memory class for RNN processing
            features = self.memory(input, masks, hidden_states)
            if features.dim() == 3:  # Remove sequence dimension for inference
                features = features.squeeze(0)
        else:
            features = input

        features = squeeze_preserve_batch(self._layers(features))

        
        # Generate quantiles for each output dimension
        quantiles = torch.stack([layer(features) for layer in self._quantile_layers], dim=1)
        quantiles = squeeze_preserve_batch(quantiles)
        self._last_quantiles = quantiles
        # print(f"quantile shape: {quantiles.shape}")
        if distribution:
            return quantiles

        # Convert quantiles to values using risk measure
        values = self.quantiles_to_values(quantiles, *measure_args)
        return values
    
    def quantiles_to_values(self, quantiles: torch.Tensor, *measure_args) -> torch.Tensor:
        """Computes values from quantiles."""
        if measure_args:
            values = self._measure_func(self, *[squeeze_preserve_batch(m) for m in measure_args])(quantiles)
        else:
            values = self._measure(quantiles)
        return values
    
    def reset(self, dones=None):
        """Reset hidden states."""
        if self.memory is not None:
            self.memory.reset(dones)
    
    def get_hidden_states(self):
        """Get current hidden states."""
        if self.memory is not None:
            return self.memory.hidden_states
        return None

    def _init(self, layers: Union[List[nn.Module], nn.ModuleList], fade: bool = True, gain: float = 1.0):
        """Initializes neural network layers."""
        if isinstance(layers, nn.ModuleList):
            layers = list(layers)
        elif isinstance(layers, nn.Sequential):
            layers = list(layers)
            
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        if not linear_layers:
            return
            
        last_layer_idx = len(linear_layers) - 1
        
        for idx, layer in enumerate(linear_layers):
            current_gain = gain / 100.0 if fade and idx == last_layer_idx else gain
            nn.init.xavier_normal_(layer.weight, gain=current_gain)


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
        quantile_count=200,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # Handle deprecated arguments
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        
        if kwargs:
            print("Quantile_NN.__init__ got unexpected arguments, which will be ignored:", list(kwargs.keys()))
        
        super().__init__()
        
        activation = resolve_nn_activation(activation)
        mlp_input_dim_a = rnn_hidden_dim
        mlp_input_dim_c = num_critic_obs

        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)


        # Critic network - QuantileCritic
        self.critic = QuantileCritic(
            input_dim=mlp_input_dim_c,
            output_dim=1,  # Value function outputs single value
            hidden_dims=critic_hidden_dims,
            quantile_count=quantile_count,
            recurrent_layers=1,  # Critic is not recurrent
            activations=[activation] * len(critic_hidden_dims),
            init_fade=False,
            init_gain=0.5,
            measure_kwargs=measure_kwargs,
        )

        # Actor memory
        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic Network: {self.critic}")
        print(f"Actor RNN: {self.memory_a}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.critic.reset(dones)

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
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
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
        return self.actor(observations)

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return self.act_inference_b(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        """Evaluate critic - returns scalar values from quantile distribution."""
        return self.critic(critic_observations, masks=masks, hidden_states=hidden_states, distribution=False)

    def evaluate_quantiles(self, critic_observations, masks=None, hidden_states=None):
        """Get full quantile distribution."""
        return self.critic(critic_observations, masks=masks, hidden_states=hidden_states, distribution=True)

    def get_last_quantiles(self):
        """Get last quantiles from critic."""
        return self.critic.last_quantiles

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.critic.get_hidden_states()

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True