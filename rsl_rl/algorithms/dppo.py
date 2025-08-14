from __future__ import annotations
import torch
from torch import nn
from typing import Dict, List, Tuple, Type, Union
from itertools import chain

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.quantile_nn import Quantile_NN, energy_loss
from rsl_rl.env import VecEnv
from rsl_rl.utils.recurrency import trajectories_to_transitions, transitions_to_trajectories
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

import torch.optim as optim

class DPPO:
    """Distributional Proximal Policy Optimization algorithm."""

    policy: Quantile_NN

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cuda",
        normalize_advantage_per_mini_batch=False,
        # Distributional parameters
        distributional_loss_type="mse",  # "mse", "huber", "energy"
        huber_delta=1.0,
        quantile_loss_coef=1.0,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            learning_rate_rnd = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=learning_rate_rnd)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # Policy components
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Storage
        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Distributional parameters
        self.distributional_loss_type = distributional_loss_type
        self.huber_delta = huber_delta
        self.quantile_loss_coef = quantile_loss_coef

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
            print(f"Hidden states: {len(self.transition.hidden_states)}")
        
        # Compute actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute intrinsic rewards if RND is used
        if self.rnd:
            rnd_state = infos["observations"]["rnd_state"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards += self.intrinsic_rewards
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def compute_distributional_loss(self, predicted_quantiles, target_values):
        """Compute distributional loss between quantiles and target values."""
        batch_size = predicted_quantiles.shape[0]
        quantile_count = predicted_quantiles.shape[-1]
        
        # Expand target values to match quantile dimensions
        target_expanded = target_values
        
        if self.distributional_loss_type == "mse":
            # Simple MSE loss between quantiles and targets
            loss = nn.functional.mse_loss(predicted_quantiles, target_expanded)
        
        elif self.distributional_loss_type == "huber":
            # Huber loss between quantiles and targets
            loss = nn.functional.huber_loss(predicted_quantiles, target_expanded, delta=self.huber_delta)
        
        elif self.distributional_loss_type == "energy":
            # Energy loss between distributions
            loss = energy_loss(predicted_quantiles.reshape(-1, quantile_count), 
                             target_expanded.reshape(-1, quantile_count))
        
        else:
            raise ValueError(f"Unknown distributional loss type: {self.distributional_loss_type}")
        
        return loss


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_distributional_loss = 0
        
        # RND and symmetry losses
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        # Generator for mini batches
        if self.policy.is_recurrent:
            print("Using recurrent mini batch generator")
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            old_actions_log_prob_batch = old_actions_log_prob_batch.detach()
            target_values_batch        = target_values_batch.detach()
            advantages_batch           = advantages_batch.detach()
            returns_batch              = returns_batch.detach()
            old_mu_batch               = old_mu_batch.detach()
            old_sigma_batch            = old_sigma_batch.detach()
            hid_states_batch           = hid_states_batch

            print("HIDDDDDDDDDDDDDDDD",hid_states_batch)

            print("HEEEEEEEEEEEEEEEEE",hid_states_batch[1])

            # Make sure hidden states are detached from previous graphs
            # if hid_states_batch is not None:
            #     if isinstance(hid_states_batch, (list, tuple)):
            #         hid_states_batch = [
            #             tuple(x.detach() if x is not None else None for x in h) if isinstance(h, (list, tuple))
            #             else (h.detach() if h is not None else None)
            #             for h in hid_states_batch
            #         ]
            #     else:
            #         hid_states_batch = hid_states_batch.detach()

            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            # Normalize advantages per mini batch if needed
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # Repeat batch elements
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions and values with current policy
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            
            # Get both scalar values and quantile distributions
            print("critic_obs_batch shape:", critic_obs_batch.shape)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            quantiles_batch = self.policy.evaluate_quantiles(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            
            # Entropy (only for original samples)
            mu_batch = self.policy.action_mean[:original_batch_size].clone()
            sigma_batch = self.policy.action_std[:original_batch_size].clone()
            entropy_batch = self.policy.entropy[:original_batch_size].clone()

            # Adaptive KL scheduling
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss (using scalar values)
            # TODO: Split function from forward to compute loss 
            if self.use_clipped_value_loss:
                print("Clippp",target_values_batch.shape, value_batch.shape)
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Distributional loss (using quantile distributions)
            distributional_loss = self.compute_distributional_loss(quantiles_batch, returns_batch)

            # Total loss
            loss = (surrogate_loss + 
                   self.value_loss_coef * value_loss + 
                   self.quantile_loss_coef * distributional_loss - 
                   self.entropy_coef * entropy_batch.mean())

            # print(loss)
            # Symmetry loss
            # if self.symmetry:
            #     if not self.symmetry["use_data_augmentation"]:
            #         data_augmentation_func = self.symmetry["data_augmentation_func"]
            #         obs_batch, _ = data_augmentation_func(
            #             obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
            #         )
            #         num_aug = int(obs_batch.shape[0] / original_batch_size)

            #     mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
            #     action_mean_orig = mean_actions_batch[:original_batch_size]
            #     _, actions_mean_symm_batch = data_augmentation_func(
            #         obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
            #     )

            #     mse_loss = torch.nn.MSELoss()
            #     symmetry_loss = mse_loss(
            #         mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
            #     )
                
            #     if self.symmetry["use_mirror_loss"]:
            #         loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
            #     else:
            #         symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # if self.rnd:
            #     predicted_embedding = self.rnd.predictor(rnd_state_batch)
            #     target_embedding = self.rnd.target(rnd_state_batch).detach()
            #     mseloss = torch.nn.MSELoss()
            #     rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute gradients
            self.optimizer.zero_grad()
            
            # if self.rnd:
            #     loss.backward(retain_graph=True)
            #     self.rnd_optimizer.zero_grad()
            #     rnd_loss.backward()
            
            # else:
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()

            # Collect gradients from all GPUs
            # if self.is_multi_gpu:
            #     self.reduce_parameters()

            # Apply gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_distributional_loss += distributional_loss.item()
            
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distributional_loss /= num_updates
        
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear storage
        self.storage.clear()

        # Construct loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "distributional": mean_distributional_loss,
        }
        
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        
        torch.distributed.broadcast_object_list(model_params, src=0)
        
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel