# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent, obs_encoder ,EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state


class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()
        self.encoder_obs = False
        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if self.encoder_obs:
            self.encoder_cfg = train_cfg["encoder"]
            self.obs_indices = obs[:,self.encoder_cfg.get("obs_indices", 36):]
            
            # Get encoder configuration
            encoder_type = self.encoder_cfg.get("type", "mlp")  # Default to MLP if not specified
            input_dim = self.obs_indices.shape[1]
            output_dim = self.encoder_cfg.get("output_dim", 8)
            
            # Initialize encoder based on type
            # Initialize encoder based on type
            encoder_params = {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dims": self.encoder_cfg.get("hidden_dims", [256, 128])
            }
            
            if encoder_type == "mlp":
                self.obs_encoder = obs_encoder.ObsEncoder(**encoder_params).to(device)
                print(f"MLP Encoder Structure: {self.obs_encoder}")
            
            elif encoder_type == "gru":
                # Add GRU specific parameters
                encoder_params.update({
                    "gru_hidden_size": self.encoder_cfg.get("gru_hidden_size", 256),
                    "gru_num_layers": self.encoder_cfg.get("gru_num_layers", 2)
                })
                self.obs_encoder = obs_encoder.GRUEncoder(**encoder_params).to(device)
                print(f"GRU Encoder Structure: {self.obs_encoder}")
            
            elif encoder_type == "conv":
                # Add Conv specific parameters
                encoder_params.update({
                    "conv_channels": self.encoder_cfg.get("conv_channels", [32, 64, 128]),
                    "conv_kernel_sizes": self.encoder_cfg.get("conv_kernel_sizes", [3, 3, 3]),
                    "conv_strides": self.encoder_cfg.get("conv_strides", [1, 1, 1])
                })
                self.obs_encoder = obs_encoder.ConvEncoder(**encoder_params).to(device)
                print(f"Conv Encoder Structure: {self.obs_encoder}")
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")

            # Initialize optimizer
            self.encoder_optimizer = torch.optim.Adam(self.obs_encoder.parameters(), lr=self.encoder_cfg.get("learning_rate", 3e-4))
            
            # Load pre-trained model if in navigate mode and model path exists
            if self.navigate and hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'encoderbase_model'):
                model_state = torch.load(self.env.cfg.encoderbase_model)
                if "encoder_state_dict" in model_state:
                    self.obs_encoder.load_state_dict(model_state["encoder_state_dict"])

            # Update observation dimension
            num_obs = self.encoder_cfg.get("obs_indices", 36) + output_dim
        if "teacher" in extras["observations"]:
            self.privileged_obs_type = "teacher"  # policy distillation
        else:
            self.privileged_obs_type = None
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: StudentTeacher | StudentTeacherRecurrent | Quantile_NN = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["teacher"])

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()
        # check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    """
    Helper methods.
    """

    def _construct_algorithm(self, obs) -> Distillation:
        """Construct the distillation algorithm."""
        # initialize the actor-critic
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: StudentTeacher | StudentTeacherRecurrent = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg