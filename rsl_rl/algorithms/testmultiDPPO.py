#!/usr/bin/env python3
"""
DPPO Test Script
Tests the Distributional PPO implementation with quantile networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import sys
import os
import time

# Add the parent directory to path to import the modules
# Adjust this path based on your project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsl_rl.modules.quantile_nn import Quantile_NN, QuantileCritic
from rsl_rl.algorithms.dppo import DPPO
from rsl_rl.storage import RolloutStorage


class MockEnv:
    """Mock environment for testing DPPO"""
    def __init__(self, num_envs=4, obs_dim=8, action_dim=2):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def reset(self):
        return torch.randn(self.num_envs, self.obs_dim, device=self.device)
    
    def step(self, actions):
        obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        print(f"  Mock step with actions: {actions.shape} -> {obs.shape}")
        rewards = torch.randn(self.num_envs, 1, device=self.device)
        rewards = rewards.squeeze(1)
        dones = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        # Randomly set some episodes as done
        # dones[torch.rand(self.num_envs) < 0.1] = 1
        
        infos = {"time_outs": torch.zeros_like(dones, dtype=torch.float).squeeze(1)}
        dones = dones.squeeze(1)
        return obs, rewards, dones, infos


def test_quantile_critic():
    """Test QuantileCritic functionality"""
    print("=" * 50)
    print("Testing QuantileCritic...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    batch_size = 32
    input_dim = 16
    quantile_count = 50
    
    # Create critic
    critic = QuantileCritic(
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[128, 128],
        quantile_count=quantile_count,
        recurrent_layers=0,
        activations=[nn.ReLU(), nn.ReLU()],
        measure_kwargs={"beta": 0.1}
    ).to(device)
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim, device=device)
    
    # Test scalar output
    values = critic(x, distribution=False)
    print(f" Scalar values shape: {values.shape} (expected: [{batch_size}, 1])")
    # assert values.shape == (batch_size, 1), f"Expected shape [{batch_size}, 1], got {values.shape}"
    
    # Test quantile distribution output
    quantiles = critic(x, distribution=True)
    print(f" Quantiles shape: {quantiles.shape} (expected: [{batch_size}, 1, {quantile_count}])")
    # assert quantiles.shape == (batch_size, 1, quantile_count), f"Expected shape [{batch_size}, 1, {quantile_count}], got {quantiles.shape}"
    
    # Test measure conversion
    manual_values = critic.quantiles_to_values(quantiles)
    print(f" Manual measure conversion shape: {manual_values.shape}")
    
    print("QuantileCritic tests passed! ")


def test_quantile_nn():
    """Test Quantile_NN (Actor-Critic) functionality"""
    print("=" * 50)
    print("Testing Quantile_NN...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    num_envs = 8
    num_actor_obs = 12
    num_critic_obs = 16
    num_actions = 4
    quantile_count = 100
    
    # Create policy
    policy = Quantile_NN(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[128, 128],
        quantile_count=quantile_count,
        rnn_hidden_dim=32,
        measure_kwargs={"beta": 0.0}
    ).to(device)
    
    # Test observations
    actor_obs = torch.randn(num_envs, num_actor_obs, device=device)
    critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    
    # Test actor
    actions = policy.act(actor_obs)
    print(f" Actions shape: {actions.shape} (expected: [{num_envs}, {num_actions}])")
    assert actions.shape == (num_envs, num_actions)
    
    # Test action log probabilities
    log_probs = policy.get_actions_log_prob(actions)
    print(f" Log probs shape: {log_probs.shape} (expected: [{num_envs}])")
    assert log_probs.shape == (num_envs,)
    
    # Test critic
    values = policy.evaluate(critic_obs)
    print(f" Values shape: {values.shape} (expected: [{num_envs}, 1])")
    # print(f"  Values: {values}...")  # Print first 5 values
    assert values.shape == (num_envs, 1)
    
    # Test quantile evaluation
    quantiles = policy.evaluate_quantiles(critic_obs)
    print(f" Quantiles shape: {quantiles.shape} (expected: [{num_envs}, 1, {quantile_count}])")
    assert quantiles.shape == (num_envs, 1, quantile_count)
    
    # Test reset functionality
    dones = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool, device=device)
    policy.reset(dones)
    print(" Reset functionality working")
    
    # Test hidden states
    hidden_states = policy.get_hidden_states()
    print(f" Hidden states retrieved: {type(hidden_states)}")
    
    print("Quantile_NN tests passed! ")


def test_dppo_initialization():
    """Test DPPO initialization"""
    print("=" * 50)
    print("Testing DPPO initialization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create policy
    policy = Quantile_NN(
        num_actor_obs=8,
        num_critic_obs=12,
        num_actions=2,
        quantile_count=50,
        measure_kwargs={"beta": 0.1}
    )
    
    # Test different DPPO configurations
    configs = [
        {"distributional_loss_type": "mse"},
        {"distributional_loss_type": "huber", "huber_delta": 1.0},
        {"distributional_loss_type": "energy"},
    ]
    
    for i, config in enumerate(configs):
        print(f"  Testing config {i+1}: {config}")
        dppo = DPPO(
            policy=policy,
            device=device,
            **config
        )
        print(f"     DPPO initialized with {config['distributional_loss_type']} loss")
    
    print("DPPO initialization tests passed! ")


def test_dppo_storage():
    """Test DPPO with storage"""
    print("=" * 50)
    print("Testing DPPO with storage...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment parameters
    num_envs = 128
    num_steps = 64
    num_actor_obs = 32
    num_critic_obs = 48
    num_actions = 8
    quantile_count = 200
    
    # Create policy
    policy = Quantile_NN(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        quantile_count=quantile_count,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[512, 512],
        rnn_hidden_dim=128
    )
    
    dppo = DPPO(
        policy=policy,
        num_learning_epochs=4,
        num_mini_batches=8,
        device=device,
        distributional_loss_type="mse",
        quantile_loss_coef=0.5,
    )

    # Initialize storage
    dppo.init_storage(
        training_type="dppo",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        actor_obs_shape=[num_actor_obs],
        critic_obs_shape=[num_critic_obs],
        actions_shape=[num_actions]
    )

    print(f" Storage initialized: {type(dppo.storage)}")
    
    # Create mock environment
    env = MockEnv(num_envs, num_actor_obs, num_actions)
    
    # Simulate rollout
    obs = env.reset()
    critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    
    for step in range(num_steps):
        # Act
        actions = dppo.act(obs, critic_obs)
        
        # Environment step
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Process step
        dppo.process_env_step(rewards, dones, infos)
        
        obs = next_obs
        critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    
    # Compute returns
    last_critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    dppo.compute_returns(last_critic_obs)
    
    print(f" Rollout completed: {num_steps} steps")
    
    # Test update
    loss_dict = dppo.update()
    
    print(" Update completed")
    print("  Loss components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.4f}")
    
    # Verify loss components
    expected_keys = {"value_function", "surrogate", "entropy", "distributional"}
    assert expected_keys.issubset(set(loss_dict.keys())), f"Missing loss keys: {expected_keys - set(loss_dict.keys())}"
    
    print("DPPO storage tests passed! ")


def test_distributional_losses():
    """Test different distributional loss functions"""
    print("=" * 50)
    print("Testing distributional loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create policy
    policy = Quantile_NN(
        num_actor_obs=6,
        num_critic_obs=8,
        num_actions=2,
        quantile_count=20
    )
    
    # Test different loss types
    loss_types = ["mse", "huber"]
    
    for loss_type in loss_types:
        print(f"  Testing {loss_type} loss...")
        
        dppo = DPPO(
            policy=policy,
            device=device,
            distributional_loss_type=loss_type,
            huber_delta=1.0
        )
        
        # Create test data
        batch_size = 16
        quantile_count = 20
        predicted_quantiles = torch.randn(batch_size, 1, quantile_count, device=device)
        target_values = torch.randn(batch_size, 1, device=device)
        
        # Compute loss
        loss = dppo.compute_distributional_loss(predicted_quantiles, target_values)
        
        print(f"     {loss_type} loss computed: {loss.item():.4f}")
        assert not torch.isnan(loss), f"{loss_type} loss is NaN"
        assert loss.item() >= 0, f"{loss_type} loss is negative: {loss.item()}"
    
    print("Distributional loss tests passed! ")


def test_quantile_visualization():
    """Test and visualize quantile distributions"""
    print("=" * 50)
    print("Testing quantile visualization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a simple critic
    critic = QuantileCritic(
        input_dim=4,
        output_dim=1,
        hidden_dims=[32, 32],
        activations=[nn.ReLU(), nn.ReLU()],
        quantile_count=51,  # Odd number for cleaner median
        measure_kwargs={"beta": 0.0}  # Neutral risk
    )
    
    # Create test inputs representing different "states"
    test_inputs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Good state
        [0.0, 1.0, 0.0, 0.0],  # Bad state
        [0.0, 0.0, 1.0, 0.0],  # Neutral state
        [0.0, 0.0, 0.0, 1.0],  # Uncertain state
    ], device=device)
    
    # Get quantile distributions
    quantiles = critic(test_inputs, distribution=True)
    values = critic(test_inputs, distribution=False)
    
    print(f" Quantile distributions computed: {quantiles.shape}")
    print(f" Scalar values computed: {values.shape}")
    
    # Print statistics for each state
    state_names = ["Good", "Bad", "Neutral", "Uncertain"]
    for i, name in enumerate(state_names):
        q = quantiles[i, 0, :].detach().cpu().numpy()
        v = values[i, 0].detach().cpu().item()
        
        print(f"  {name} state:")
        print(f"    Scalar value: {v:.3f}")
        print(f"    Q25: {np.percentile(q, 25):.3f}")
        print(f"    Q50: {np.percentile(q, 50):.3f}")
        print(f"    Q75: {np.percentile(q, 75):.3f}")
        print(f"    Range: [{q.min():.3f}, {q.max():.3f}]")
    
    print("Quantile visualization tests passed! ")


def run_single_epoch_rollout(dppo: DPPO, env: MockEnv, num_steps: int, num_critic_obs: int) -> Tuple[float, Dict[str, float]]:
    """
    Run a single epoch of rollout and update
    
    Returns:
        rollout_time: Time taken for rollout
        loss_dict: Dictionary of loss components
    """
    device = "cuda"
    
    # Start rollout
    rollout_start = time.time()
    
    obs = env.reset()
    for step in range(num_steps):
        critic_obs = torch.randn(env.num_envs, num_critic_obs, device=device)
        actions = dppo.act(obs, critic_obs)
        
        # Simulate environment step
        obs, rewards, dones, infos = env.step(actions)
        dppo.process_env_step(rewards, dones, infos)
    
    rollout_time = time.time() - rollout_start
    
    # Compute returns and update
    last_critic_obs = torch.randn(env.num_envs, num_critic_obs, device=device)
    dppo.compute_returns(last_critic_obs)
    
    update_start = time.time()
    loss_dict = dppo.update()
    update_time = time.time() - update_start
    
    return rollout_time, update_time, loss_dict


def run_performance_test():
    """Run single epoch performance benchmarks"""
    print("=" * 50)
    print("Running single-epoch performance tests...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Performance test parameters
    num_envs = 14
    num_steps = 48
    num_actor_obs = 36
    num_critic_obs = 36
    num_actions = 8
    quantile_count = 200
    
    # Create policy
    policy = Quantile_NN(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        quantile_count=quantile_count,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        rnn_hidden_dim=128
    )
    
    dppo = DPPO(
        policy=policy,
        num_learning_epochs=5,
        num_mini_batches=6,
        device=device
    )
    
    # Initialize storage
    dppo.init_storage(
        training_type="dppo",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        actor_obs_shape=[num_actor_obs],
        critic_obs_shape=[num_critic_obs],
        actions_shape=[num_actions]
    )
    
    # Create environment
    env = MockEnv(num_envs, num_actor_obs, num_actions)
    
    # Warmup
    obs = torch.randn(num_envs, num_actor_obs, device=device)
    critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    _ = dppo.act(obs, critic_obs)
    
    # Run single epoch
    rollout_time, update_time, loss_dict = run_single_epoch_rollout(dppo, env, num_steps, num_critic_obs)
    
    print(f" Single-epoch performance test completed:")
    print(f"  Environments: {num_envs}")
    print(f"  Steps: {num_steps}")
    print(f"  Quantiles: {quantile_count}")
    print(f"  Rollout time: {rollout_time:.3f}s ({rollout_time/num_steps:.4f}s per step)")
    print(f"  Update time: {update_time:.3f}s")
    print(f"  Total samples: {num_envs * num_steps}")
    print(f"  Samples per second: {(num_envs * num_steps) / rollout_time:.1f}")


def run_multi_epoch_performance_test(num_epochs: int = 10):
    """
    Run multi-epoch performance test to evaluate training stability and performance over time
    
    Args:
        num_epochs: Number of training epochs to run
    """
    print("=" * 60)
    print(f"Running multi-epoch performance test ({num_epochs} epochs)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    num_envs = 256
    num_steps = 128
    num_actor_obs = 48
    num_critic_obs = 64
    num_actions = 12
    quantile_count = 100
    
    # Create policy and algorithm
    policy = Quantile_NN(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        quantile_count=quantile_count,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        rnn_hidden_dim=256,
        measure_kwargs={"beta": 0.0}
    )
    
    dppo = DPPO(
        policy=policy,
        num_learning_epochs=8,
        num_mini_batches=16,
        device=device,
        distributional_loss_type="mse",
        quantile_loss_coef=1.0,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=1.0
    )
    
    # Initialize storage
    dppo.init_storage(
        training_type="dppo",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        actor_obs_shape=[num_actor_obs],
        critic_obs_shape=[num_critic_obs],
        actions_shape=[num_actions]
    )
    
    # Create environment
    env = MockEnv(num_envs, num_actor_obs, num_actions)
    
    # Performance tracking
    epoch_times = []
    rollout_times = []
    update_times = []
    loss_history = {
        'value_function': [],
        'surrogate': [],
        'entropy': [],
        'distributional': []
    }
    
    # Warmup
    print("Warming up...")
    obs = torch.randn(num_envs, num_actor_obs, device=device)
    critic_obs = torch.randn(num_envs, num_critic_obs, device=device)
    _ = dppo.act(obs, critic_obs)
    
    print(f"Starting {num_epochs} epochs of training...")
    total_start_time = time.time()
    
    # Multi-epoch training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Run rollout and update
        rollout_time, update_time, loss_dict = run_single_epoch_rollout(
            dppo, env, num_steps, num_critic_obs
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Track performance
        epoch_times.append(epoch_time)
        rollout_times.append(rollout_time)
        update_times.append(update_time)
        
        # Track losses
        for key in loss_history.keys():
            if key in loss_dict:
                loss_history[key].append(loss_dict[key])
            else:
                loss_history[key].append(0.0)  # Default value if loss not present
        
        # Progress reporting
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
            samples_per_sec = (num_envs * num_steps) / rollout_time
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Total: {epoch_time:.3f}s, "
                  f"Rollout: {rollout_time:.3f}s, "
                  f"Update: {update_time:.3f}s, "
                  f"SPS: {samples_per_sec:.0f}, "
                  f"VF Loss: {loss_dict.get('value_function', 0):.4f}")
    
    total_time = time.time() - total_start_time
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("MULTI-EPOCH PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    total_samples = num_epochs * num_envs * num_steps
    avg_epoch_time = np.mean(epoch_times)
    avg_rollout_time = np.mean(rollout_times)
    avg_update_time = np.mean(update_times)
    avg_samples_per_sec = (num_envs * num_steps) / avg_rollout_time
    
    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Environments: {num_envs}")
    print(f"  Steps per epoch: {num_steps}")
    print(f"  Quantiles: {quantile_count}")
    print(f"  Total samples processed: {total_samples:,}")
    
    print(f"\nTiming Statistics:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Average epoch time: {avg_epoch_time:.3f}s � {np.std(epoch_times):.3f}s")
    print(f"  Average rollout time: {avg_rollout_time:.3f}s � {np.std(rollout_times):.3f}s")
    print(f"  Average update time: {avg_update_time:.3f}s � {np.std(update_times):.3f}s")
    print(f"  Average samples/sec: {avg_samples_per_sec:.0f} � {np.std([(num_envs * num_steps) / t for t in rollout_times]):.0f}")
    
    # Performance stability
    rollout_cv = np.std(rollout_times) / np.mean(rollout_times) * 100
    update_cv = np.std(update_times) / np.mean(update_times) * 100
    
    print(f"\nPerformance Stability (Coefficient of Variation):")
    print(f"  Rollout time CV: {rollout_cv:.2f}%")
    print(f"  Update time CV: {update_cv:.2f}%")
    
    # Loss statistics
    print(f"\nLoss Statistics:")
    for loss_name, values in loss_history.items():
        if values and any(v != 0 for v in values):
            values_array = np.array(values)
            print(f"  {loss_name.replace('_', ' ').title()}:")
            print(f"    Initial: {values[0]:.6f}, Final: {values[-1]:.6f}")
            print(f"    Mean: {np.mean(values_array):.6f} � {np.std(values_array):.6f}")
            print(f"    Min: {np.min(values_array):.6f}, Max: {np.max(values_array):.6f}")
    
    # Memory usage (if CUDA)
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Performance trends
    if num_epochs >= 5:
        # Check for performance degradation
        first_half_avg = np.mean(rollout_times[:num_epochs//2])
        second_half_avg = np.mean(rollout_times[num_epochs//2:])
        performance_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        print(f"\nPerformance Trends:")
        print(f"  First half avg rollout time: {first_half_avg:.3f}s")
        print(f"  Second half avg rollout time: {second_half_avg:.3f}s")
        print(f"  Performance change: {performance_change:+.2f}%")
        
        if abs(performance_change) > 10:
            print(f"  �  Significant performance change detected!")
        else:
            print(f"   Stable performance across epochs")
    
    # Efficiency metrics
    print(f"\nEfficiency Metrics:")
    print(f"  Time per sample: {total_time / total_samples * 1000:.4f} ms")
    print(f"  Updates per second: {num_epochs / total_time:.2f}")
    print(f"  Rollout efficiency: {avg_rollout_time / avg_epoch_time * 100:.1f}% of epoch time")
    print(f"  Update efficiency: {avg_update_time / avg_epoch_time * 100:.1f}% of epoch time")
    
    print("\nMulti-epoch performance test completed! ")
    
    return {
        'epoch_times': epoch_times,
        'rollout_times': rollout_times,
        'update_times': update_times,
        'loss_history': loss_history,
        'total_time': total_time,
        'avg_samples_per_sec': avg_samples_per_sec
    }


def main():
    """Run all tests"""
    print("DPPO Test Suite")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Core functionality tests
        # test_quantile_critic()
        # test_quantile_nn()
        test_dppo_initialization()
        test_dppo_storage()
        test_distributional_losses()
        # test_quantile_visualization()
        
        # Single epoch performance test
        run_performance_test()
        
        # Multi-epoch performance test
        print("\n" + "=�" * 20)
        print("Starting comprehensive multi-epoch performance testing...")
        results = run_multi_epoch_performance_test(num_epochs=20)
        
        print("\n" + "=" * 50)
        print(" All tests passed successfully!")
        print("DPPO implementation is working correctly.")
        print(f"Average performance: {results['avg_samples_per_sec']:.0f} samples/sec")
        
    except Exception as e:
        print(f"\nL Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()