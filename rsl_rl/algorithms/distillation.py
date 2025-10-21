# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.modules import obs_encoder
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None
        
        # Encoder components
        self.encoder_obs = False
        self.student_encoder = None
        self.teacher_encoder = None
        self.student_encoder_optimizer = None
        self.teacher_encoder_optimizer = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs, extras = None):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        self.transition.perception_obs = extras["observations"].get("perception", None) if extras is not None and "observations" in extras else None
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_encoder_loss = 0
        # accumulators for gradient steps must be tensors so we can call backward()
        policy_loss_accum = torch.tensor(0.0, device=self.device)
        encoder_loss_accum = torch.tensor(0.0, device=self.device)
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, dones,perception in self.storage.generator():

                # Encode observations if encoder is enabled
                if self.encoder_obs:
                    # get the full modified observation (with encoded tail)
                    encoded_obs = self.encode_obs(obs, is_teacher=False, extras=perception)

                    # separate the encoded part and ensure the policy forward uses a detached copy
                    start_idx = self.encoder_cfg.get("obs_indices", 36)
                    encoded_part = encoded_obs[:, start_idx:]

                    # build policy input where the encoded part is detached to prevent policy loss
                    policy_input = encoded_obs.clone()
                    policy_input[:, start_idx:] = encoded_part.detach()

                    actions = self.policy.act_inference(policy_input)

                    # encoder loss compares the encoded representation to the original observation tail
                    encoder_loss = self.loss_fn(encoded_part, obs[:, start_idx:])
                else:
                    # Regular behavior cloning loss without encoders
                    actions = self.policy.act_inference(obs)
                
                # Compute behavior (policy) loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # Accumulate per-component losses
                policy_loss_accum = policy_loss_accum + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                if self.encoder_obs:
                    encoder_loss_accum = encoder_loss_accum + encoder_loss
                    mean_encoder_loss += encoder_loss.item()

                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    # --- Encoder update (student encoder) ---
                    if self.encoder_obs:
                        # zero encoder grads
                        self.student_encoder_optimizer.zero_grad()
                        # backward on accumulated encoder loss
                        encoder_loss_accum.backward()
                        # clip encoder grads
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.student_encoder.parameters(), self.max_grad_norm)
                        # step encoder optimizer
                        self.student_encoder_optimizer.step()

                    # --- Policy update ---
                    # zero policy grads
                    self.optimizer.zero_grad()
                    # backward on accumulated policy loss
                    policy_loss_accum.backward()

                    # Apply gradient clipping if needed for policy
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)

                    # Reduce parameters for multi-GPU setup (only policy parameters currently)
                    if self.is_multi_gpu:
                        self.reduce_parameters()

                    # step policy optimizer
                    self.optimizer.step()

                    # reset accumulators and detach hidden states
                    self.policy.detach_hidden_states()
                    policy_loss_accum = torch.tensor(0.0, device=self.device)
                    encoder_loss_accum = torch.tensor(0.0, device=self.device)

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        # compute encoder mean if any encoder losses were accumulated
        if self.encoder_obs:
            encoder_cnt = 0
            # derive encoder count from whether encoder was used in storage; fall back to cnt if unknown
            # we tracked mean_encoder_loss as a sum of items, so divide by number of times it was added
            # to keep it simple, if mean_encoder_loss is non-zero use cnt as denominator
            if mean_encoder_loss != 0:
                mean_encoder_loss = mean_encoder_loss / cnt
            else:
                mean_encoder_loss = 0
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}
        if self.encoder_obs:
            loss_dict["encoder"] = mean_encoder_loss

        return loss_dict

    """
    Encoder functions
    """
    
    def initialize_encoders(self, encoder_cfg, student_obs_shape, teacher_obs_shape):
        """Initialize the student and teacher observation encoders based on configuration."""
        self.encoder_obs = True
        self.encoder_cfg = encoder_cfg
        
        # Get encoder configurations
        student_type = self.encoder_cfg.get("student_type", "mlp")  # Default to MLP if not specified
        teacher_type = self.encoder_cfg.get("teacher_type", "mlp")  # Default to MLP if not specified
        student_output_dim = self.encoder_cfg.get("output_dim", 8)
        teacher_output_dim = self.encoder_cfg.get("output_dim", 8)
        # Base parameters for both encoders
        student_params = {
            "input_dim": student_obs_shape,
            "output_dim": student_output_dim,
            "hidden_dims": self.encoder_cfg.get("student_hidden_dims", [256, 128])
        }
        
        teacher_params = {
            "input_dim": teacher_obs_shape,
            "output_dim": teacher_output_dim,
            "hidden_dims": self.encoder_cfg.get("teacher_hidden_dims", [256, 128])
        }

        # Initialize student encoder based on type
        if student_type == "mlp":
            self.student_encoder = obs_encoder.ObsEncoder(**student_params).to(self.device)
        elif student_type == "gru":
            student_params.update({
                "gru_hidden_size": self.encoder_cfg.get("student_gru_hidden_size", 256),
                "gru_num_layers": self.encoder_cfg.get("student_gru_num_layers", 2)
            })
            self.student_encoder = obs_encoder.GRUEncoder(**student_params).to(self.device)
        elif student_type == "conv":
            student_params.update({
                "conv_channels": self.encoder_cfg.get("student_conv_channels", [32, 64, 128]),
                "conv_kernel_sizes": self.encoder_cfg.get("student_conv_kernel_sizes", [3, 3, 3]),
                "conv_strides": self.encoder_cfg.get("student_conv_strides", [1, 1, 1])
            })
            self.student_encoder = obs_encoder.ConvEncoder(**student_params).to(self.device)
        elif student_type == "convgru":
            student_params.update({
                # "input_shape": self.encoder_cfg.get("student_input_shape", (3, 64, 64)),
                "conv_channels": self.encoder_cfg.get("student_conv_channels", [32, 64, 128]),
                "conv_kernel_sizes": self.encoder_cfg.get("student_conv_kernel_sizes", [3, 3, 3]),
                "pool_sizes": self.encoder_cfg.get("student_pool_sizes", [2, 2, 2]),
                "gru_hidden_size": self.encoder_cfg.get("student_gru_hidden_size", 256),
                "gru_num_layers": self.encoder_cfg.get("student_gru_num_layers", 1)
            })
            self.student_encoder = obs_encoder.ConvGRUEncoder(**student_params).to(self.device)
        else:
            raise ValueError(f"Unsupported student encoder type: {student_type}")
            
        print(f"{student_type.upper()} Student Encoder Structure: {self.student_encoder}")

        # Initialize teacher encoder based on type
        if teacher_type == "mlp":
            self.teacher_encoder = obs_encoder.ObsEncoder(**teacher_params).to(self.device)
        elif teacher_type == "gru":
            teacher_params.update({
                "gru_hidden_size": self.encoder_cfg.get("teacher_gru_hidden_size", 256),
                "gru_num_layers": self.encoder_cfg.get("teacher_gru_num_layers", 2)
            })
            self.teacher_encoder = obs_encoder.GRUEncoder(**teacher_params).to(self.device)
        elif teacher_type == "conv":
            teacher_params.update({
                "conv_channels": self.encoder_cfg.get("teacher_conv_channels", [32, 64, 128]),
                "conv_kernel_sizes": self.encoder_cfg.get("teacher_conv_kernel_sizes", [3, 3, 3]),
                "conv_strides": self.encoder_cfg.get("teacher_conv_strides", [1, 1, 1])
            })
            self.teacher_encoder = obs_encoder.ConvEncoder(**teacher_params).to(self.device)
        elif teacher_type == "convgru":
            teacher_params.update({
                "input_shape": self.encoder_cfg.get("teacher_input_shape", (3, 64, 64)),
                "conv_channels": self.encoder_cfg.get("teacher_conv_channels", [32, 64, 128]),
                "conv_kernel_sizes": self.encoder_cfg.get("teacher_conv_kernel_sizes", [3, 3, 3]),
                "conv_strides": self.encoder_cfg.get("teacher_conv_strides", [2, 2, 2]),
                "gru_hidden_size": self.encoder_cfg.get("teacher_gru_hidden_size", 256),
                "gru_num_layers": self.encoder_cfg.get("teacher_gru_num_layers", 1)
            })
            self.teacher_encoder = obs_encoder.ConvGRUEncoder(**teacher_params).to(self.device)
        else:
            raise ValueError(f"Unsupported teacher encoder type: {teacher_type}")
            
        print(f"{teacher_type.upper()} Teacher Encoder Structure: {self.teacher_encoder}")
        
        # Initialize optimizer for student encoder only
        self.student_encoder_optimizer = torch.optim.Adam(
            self.student_encoder.parameters(), 
            lr=self.encoder_cfg.get("student_learning_rate", 3e-4)
        )
        
        # Freeze teacher encoder parameters
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
            
        self.teacher_encoder.eval()  # Set teacher encoder to evaluation mode
        self.teacher_encoder_optimizer = None  # No optimizer needed for teacher
    
    def encode_obs(self, obs, is_teacher=False, start_idx=None , extras = None):
        """Encode the observations using either student or teacher encoder."""
        if not self.encoder_obs or (not self.student_encoder and not self.teacher_encoder):
            return obs
            
        if start_idx is None:
            start_idx = self.encoder_cfg.get("obs_indices", 36)
            
        selected_obs = obs[:,start_idx:]
        if is_teacher:
            encoded_obs = self.teacher_encoder(selected_obs)
        else:
            if extras is not None :
                if type(extras) == dict:
                    encoded_obs = self.student_encoder(extras["observations"]["perception"].to(self.device))
                else:
                    encoded_obs = self.student_encoder(extras)
            else:
                encoded_obs = self.student_encoder(selected_obs)
        
        modified_obs = obs.clone()
        modified_obs = modified_obs[:,:start_idx]
        modified_obs = torch.cat([modified_obs, encoded_obs], dim=1)
        
        return modified_obs

    def save(self, path: str, infos=None):
        """Return encoder state dicts if encoders are present."""
        if self.encoder_obs and self.student_encoder and self.teacher_encoder:
            return {
                "student_encoder_state_dict": self.student_encoder.state_dict(),
                "teacher_encoder_state_dict": self.teacher_encoder.state_dict(),
                "student_encoder_optimizer_state_dict": self.student_encoder_optimizer.state_dict(),
            }
        return {}

    def load(self, state_dict):
        """Load encoder states if present."""
        if self.encoder_obs and self.student_encoder and self.teacher_encoder:
            # Load student encoder state and optimizer
            if "student_encoder_state_dict" in state_dict:
                self.student_encoder.load_state_dict(state_dict["student_encoder_state_dict"])
                self.student_encoder_optimizer.load_state_dict(state_dict["student_encoder_optimizer_state_dict"])
            
            # Load teacher encoder state only (teacher is frozen)
            if "teacher_encoder_state_dict" in state_dict:
                self.teacher_encoder.load_state_dict(state_dict["teacher_encoder_state_dict"])
                # Ensure teacher encoder remains frozen after loading
                for param in self.teacher_encoder.parameters():
                    param.requires_grad = False
                self.teacher_encoder.eval()

    def load_teacher_encoder(self, state_dict):
        """Load teacher encoder parameters only (for distillation after RL training)."""
        if self.encoder_obs and self.teacher_encoder:
            # print(state_dict.keys())
            self.teacher_encoder.load_state_dict(state_dict["encoder_state_dict"])
            # Freeze teacher encoder parameters
            for param in self.teacher_encoder.parameters():
                param.requires_grad = False
            self.teacher_encoder.eval()  # Set teacher encoder to evaluation mode

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
