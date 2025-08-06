#!/usr/bin/env python3
"""
Example usage of the Quantile Neural Network for reinforcement learning.

This script demonstrates how to:
1. Create and configure the quantile network
2. Process observations and generate actions
3. Compute values and quantile distributions
4. Use different risk measures
5. Handle recurrent states properly
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from quantile_nn import Quantile_NN, QuantileCritic, energy_loss, risk_measure_wang


def basic_usage_example():
    """Basic example of using the Quantile_NN"""
    print("=== Basic Usage Example ===")
    
    # Network configuration
    network = Quantile_NN(
        num_actor_obs=10,      # Dimension of actor observations
        num_critic_obs=15,     # Dimension of critic observations  
        num_actions=4,         # Number of actions
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_hidden_dim=256,
        quantile_count=200,    # Number of quantiles for value distribution
        measure_kwargs={"beta": 0.2}  # Risk measure parameters
    )
    
    batch_size = 8
    network.reset()  # Reset RNN states
    
    # Sample observations
    actor_obs = torch.randn(batch_size, 10)
    critic_obs = torch.randn(batch_size, 15)
    
    # Generate actions (stochastic)
    actions = network.act(actor_obs)
    print(f"Actions shape: {actions.shape}")
    print(f"Sample action: {actions[0]}")
    
    # Generate deterministic actions for inference
    actions_deterministic = network.act_inference(actor_obs)
    print(f"Deterministic actions shape: {actions_deterministic.shape}")
    
    # Evaluate state values (using risk measure)
    values = network.evaluate(critic_obs)
    print(f"Values shape: {values.shape}")
    print(f"Sample values: {values[:3].squeeze()}")
    
    # Get full quantile distributions
    quantiles = network.evaluate_quantiles(critic_obs)
    print(f"Quantiles shape: {quantiles.shape}")  # [batch, 1, quantile_count]
    
    # Compute action log probabilities and entropy
    log_probs = network.get_actions_log_prob(actions)
    entropy = network.entropy
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    print()


def sequential_processing_example():
    """Example of processing sequential data with RNN memory"""
    print("=== Sequential Processing Example ===")
    
    network = Quantile_NN(
        num_actor_obs=6,
        num_critic_obs=8,
        num_actions=2,
        quantile_count=100
    )
    
    batch_size = 4
    sequence_length = 20
    network.reset()
    
    print("Processing sequential observations...")
    
    actions_history = []
    values_history = []
    
    for t in range(sequence_length):
        # Generate time-varying observations
        actor_obs = torch.randn(batch_size, 6) + 0.1 * t  # Slowly changing
        critic_obs = torch.randn(batch_size, 8) + 0.05 * t
        
        # Process through network (RNN state is automatically updated)
        actions = network.act(actor_obs)
        values = network.evaluate(critic_obs)
        
        actions_history.append(actions.mean(dim=0))  # Track mean action per timestep
        values_history.append(values.mean())
        
        # Simulate some environments ending
        if t > 0 and t % 7 == 0:
            dones = torch.zeros(batch_size, dtype=torch.bool)
            dones[::2] = True  # Reset every other environment
            network.reset(dones)
            print(f"  Reset environments {torch.where(dones)[0].tolist()} at step {t}")
    
    # Show how actions evolved over time
    actions_array = torch.stack(actions_history)
    print(f"Action evolution shape: {actions_array.shape}")
    print(f"Initial mean actions: {actions_array[0]}")
    print(f"Final mean actions: {actions_array[-1]}")
    print(f"Action change magnitude: {torch.norm(actions_array[-1] - actions_array[0]):.4f}")
    
    print()


def risk_measure_example():
    """Example of using different risk measures"""
    print("=== Risk Measure Example ===")
    
    # Create standalone quantile critic
    critic = QuantileCritic(
        input_dim=5,
        output_dim=1,
        hidden_dims=[128, 64],
        quantile_count=200,
        measure_kwargs={"beta": 0.0}  # Neutral risk (expectation)
    )
    
    batch_size = 6
    critic.reset_full_hidden_state(batch_size)
    
    obs = torch.randn(batch_size, 5)
    
    # Get quantile distributions
    quantiles = critic(obs, distribution=True)
    print(f"Quantile distributions shape: {quantiles.shape}")
    
    # Compute values with different risk attitudes
    risk_levels = [-0.5, 0.0, 0.5, 1.0]  # Risk-seeking to risk-averse
    
    print("Risk measure comparison:")
    for beta in risk_levels:
        # Create risk measure for this beta
        risk_measure = risk_measure_wang(critic, beta)
        risk_values = risk_measure(quantiles)
        
        risk_type = "risk-seeking" if beta < 0 else "neutral" if beta == 0 else "risk-averse"
        print(f"  Beta {beta:4.1f} ({risk_type:12s}): mean value = {risk_values.mean():.4f}")
    
    print()


def energy_loss_example():
    """Example of using energy loss for quantile regression"""
    print("=== Energy Loss Example ===")
    
    # Generate synthetic data
    batch_size = 32
    quantile_count = 100
    
    # True quantiles (ground truth)
    true_values = torch.randn(batch_size)
    true_quantiles = true_values.unsqueeze(-1).expand(-1, quantile_count)
    
    # Predicted quantiles (with some noise)
    predicted_quantiles = true_quantiles + 0.1 * torch.randn_like(true_quantiles)
    
    # Compute energy loss
    loss = energy_loss(predicted_quantiles, true_quantiles)
    print(f"Energy loss: {loss:.6f}")
    
    # Compare with perfect predictions
    perfect_loss = energy_loss(true_quantiles, true_quantiles)
    print(f"Perfect prediction loss: {perfect_loss:.6f}")
    
    # Show that loss decreases as predictions improve
    noise_levels = [0.5, 0.2, 0.1, 0.05, 0.01]
    print("\nLoss vs prediction quality:")
    for noise in noise_levels:
        noisy_predictions = true_quantiles + noise * torch.randn_like(true_quantiles)
        loss = energy_loss(noisy_predictions, true_quantiles)
        print(f"  Noise level {noise:4.2f}: loss = {loss:.6f}")
    
    print()


def training_step_example():
    """Example of a complete training step"""
    print("=== Training Step Example ===")
    
    network = Quantile_NN(
        num_actor_obs=12,
        num_critic_obs=12,
        num_actions=3,
        quantile_count=150
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    
    batch_size = 16
    network.reset()
    
    # Sample batch of experience
    observations = torch.randn(batch_size, 12)
    rewards = torch.randn(batch_size)
    next_observations = torch.randn(batch_size, 12)
    dones = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
    
    # Forward pass
    actions = network.act(observations)
    values = network.evaluate(observations)
    log_probs = network.get_actions_log_prob(actions)
    entropy = network.entropy
    
    # Compute next values for TD target
    with torch.no_grad():
        next_values = network.evaluate(next_observations)
        td_targets = rewards + 0.99 * next_values.squeeze() * (~dones)
    
    # Compute losses
    advantages = td_targets - values.squeeze()
    
    # Actor loss (policy gradient with entropy bonus)
    actor_loss = -(log_probs * advantages.detach()).mean()
    entropy_bonus = -0.01 * entropy.mean()
    
    # Critic loss (using quantiles)
    quantiles_pred = network.evaluate_quantiles(observations)
    target_quantiles = network.critic.make_diracs(td_targets.unsqueeze(-1))
    critic_loss = energy_loss(quantiles_pred.squeeze(1), target_quantiles.squeeze(1))
    
    total_loss = actor_loss + entropy_bonus + critic_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
    
    optimizer.step()
    
    print(f"Training step completed:")
    print(f"  Actor loss: {actor_loss:.6f}")
    print(f"  Critic loss: {critic_loss:.6f}")
    print(f"  Entropy bonus: {entropy_bonus:.6f}")
    print(f"  Total loss: {total_loss:.6f}")
    print(f"  Mean advantage: {advantages.mean():.4f}")
    print(f"  Mean reward: {rewards.mean():.4f}")
    
    print()


def visualization_example():
    """Example of visualizing quantile distributions"""
    print("=== Visualization Example ===")
    
    try:
        import matplotlib.pyplot as plt
        
        critic = QuantileCritic(
            input_dim=2,
            output_dim=1,
            quantile_count=100
        )
        
        # Generate different types of observations
        obs_types = [
            torch.tensor([[1.0, 1.0]]),    # High value state
            torch.tensor([[0.0, 0.0]]),    # Neutral state
            torch.tensor([[-1.0, -1.0]]),  # Low value state
        ]
        
        plt.figure(figsize=(12, 4))
        
        for i, obs in enumerate(obs_types):
            critic.reset_full_hidden_state(1)
            
            # Get quantile distribution
            quantiles = critic(obs, distribution=True).squeeze()
            
            # Plot quantile function
            tau_values = torch.linspace(0, 1, 100)
            
            plt.subplot(1, 3, i+1)
            plt.plot(tau_values, quantiles.detach(), 'b-', linewidth=2)
            plt.fill_between(tau_values, quantiles.detach(), alpha=0.3)
            plt.xlabel('Quantile Level (τ)')
            plt.ylabel('Value')
            plt.title(f'State {i+1}: {obs.squeeze().tolist()}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantile_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Quantile distribution plot saved as 'quantile_distributions.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    print()


def main():
    """Run all examples"""
    print(" Quantile Neural Network Examples")
    print("=" * 50)
    
    # Run examples
    basic_usage_example()
    sequential_processing_example()
    risk_measure_example()
    energy_loss_example()
    training_step_example()
    visualization_example()
    
    print("=" * 50)
    print(" All examples completed successfully!")


if __name__ == "__main__":
    main()