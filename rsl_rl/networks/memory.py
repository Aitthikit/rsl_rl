# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.rnn_type = type.lower()
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode: needs saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
            # print("Unpadded trajectories shape:", out.shape)
        else:
            # inference/distillation mode: uses hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        """Reset hidden states for done environments or completely."""
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:  # GRU case
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                raise NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        """Detach hidden states from computation graph."""
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:  # GRU case
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:  # GRU case
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()

    @property
    def device(self):
        """Get device of the memory module."""
        return next(self.parameters()).device

    def get_hidden_state_shape(self, batch_size):
        """Get the shape of hidden states for initialization."""
        if self.rnn_type == "lstm":
            return (
                (self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                (self.rnn.num_layers, batch_size, self.rnn.hidden_size)
            )
        else:  # GRU
            return (self.rnn.num_layers, batch_size, self.rnn.hidden_size)

    def init_hidden_states(self, batch_size):
        """Initialize hidden states with zeros."""
        device = self.device
        if self.rnn_type == "lstm":
            self.hidden_states = (
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device),
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            )
        else:  # GRU
            self.hidden_states = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)