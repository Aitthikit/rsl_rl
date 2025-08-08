import torch
import torch.nn as nn
import numpy as np
import pytest
from rsl_rl.modules.quantile_nn_temp import Quantile_NN


class TestQuantileNN:
    """Test suite for Quantile_NN (Actor-Critic) class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.num_actor_obs = 20
        self.num_critic_obs = 25
        self.num_actions = 6
        self.batch_size = 4
        self.seq_len = 10
        
        self.network = Quantile_NN(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=[128, 64],
            critic_hidden_dims=[128, 64],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
            init_noise_std=1.0,
            quantile_count=100,
            noise_std_type="scalar",
            measure_kwargs={"beta": 0.0}
        )
        
        # Sample data
        self.actor_obs = torch.randn(self.batch_size, self.num_actor_obs)
        self.critic_obs = torch.randn(self.batch_size, self.num_critic_obs)
        self.seq_actor_obs = torch.randn(self.seq_len, self.batch_size, self.num_actor_obs)
        
        # Reset network state
        self.network.reset()
    
    def test_initialization(self):
        """Test proper initialization of Quantile_NN"""
        assert self.network.is_recurrent == True
        assert isinstance(self.network.actor, nn.Sequential)
        assert hasattr(self.network, 'critic')
        assert hasattr(self.network, 'memory_a')
        
        # Check noise std initialization
        if self.network.noise_std_type == "scalar":
            assert hasattr(self.network, 'std')
            assert self.network.std.shape == torch.Size([self.num_actions])
        elif self.network.noise_std_type == "log":
            assert hasattr(self.network, 'log_std')
    
    def test_actor_forward(self):
        """Test actor forward pass"""
        # Reset network state first
        self.network.reset()
        
        # Prepare input through memory
        rnn_output = self.network.memory_a(self.actor_obs)
        actor_output = self.network.actor(rnn_output.squeeze(0))
        
        assert actor_output.shape == (self.batch_size, self.num_actions)
        assert torch.isfinite(actor_output).all()
    
    def test_critic_forward_values(self):
        """Test critic forward pass returning values"""
        # Reset both network and critic hidden states
        self.network.reset()
        self.network.critic.reset_full_hidden_state(self.batch_size)
        
        values = self.network.evaluate(self.critic_obs)
        
        # Should return scalar values for each batch item
        assert values.shape == (self.batch_size, 1) or values.shape == (self.batch_size,)
        assert torch.isfinite(values).all()
    
    def test_critic_forward_quantiles(self):
        """Test critic forward pass returning quantile distributions"""
        # Reset both network and critic hidden states
        self.network.reset()
        self.network.critic.reset_full_hidden_state(self.batch_size)
        
        quantiles = self.network.evaluate_quantiles(self.critic_obs)
        
        # Should return quantile distributions
        expected_shape = (self.batch_size, 1, 100)  # batch_size x output_dim x quantile_count
        assert quantiles.shape == expected_shape, f"Expected {expected_shape}, got {quantiles.shape}"
        assert torch.isfinite(quantiles).all()
    
    def test_action_sampling(self):
        """Test action sampling with distribution update"""
        # Reset network state first
        self.network.reset()
        
        actions = self.network.act(self.actor_obs)
        
        assert actions.shape == (self.batch_size, self.num_actions)
        assert torch.isfinite(actions).all()
        
        # Check that distribution was created
        assert self.network.distribution is not None
        assert self.network.action_mean.shape == (self.batch_size, self.num_actions)
        assert self.network.action_std.shape == (self.batch_size, self.num_actions)
    
    def test_action_inference(self):
        """Test deterministic action inference"""
        # Reset network state first
        self.network.reset()
        
        actions_mean = self.network.act_inference(self.actor_obs)
        
        assert actions_mean.shape == (self.batch_size, self.num_actions)
        assert torch.isfinite(actions_mean).all()
    
    def test_log_prob_computation(self):
        """Test log probability computation"""
        # Reset network state first
        self.network.reset()
        
        # First sample actions to create distribution
        actions = self.network.act(self.actor_obs)
        
        # Compute log probabilities
        log_probs = self.network.get_actions_log_prob(actions)
        
        assert log_probs.shape == (self.batch_size,)
        assert torch.isfinite(log_probs).all()
        assert (log_probs <= 0).all()  # Log probabilities should be negative
    
    def test_entropy_computation(self):
        """Test entropy computation"""
        # Reset network state first
        self.network.reset()
        
        # Sample actions to create distribution
        self.network.act(self.actor_obs)
        
        entropy = self.network.entropy
        
        assert entropy.shape == (self.batch_size,)
        assert torch.isfinite(entropy).all()
        assert (entropy >= 0).all()  # Entropy should be non-negative
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        # Process some data first
        self.network.act(self.actor_obs)
        
        # Test full reset
        self.network.reset()
        
        # Test partial reset with done mask
        dones = torch.zeros(self.batch_size, dtype=torch.bool)
        dones[0] = True  # Mark first environment as done
        dones[2] = True  # Mark third environment as done
        
        self.network.reset(dones)
        
        # No assertion needed, just check it doesn't crash
    
    def test_sequential_processing(self):
        """Test processing sequential data"""
        # Reset network state at the beginning
        self.network.reset()
        
        actions_list = []
        values_list = []
        
        # Process sequence
        for t in range(self.seq_len):
            obs_t = self.seq_actor_obs[t]
            critic_obs_t = torch.randn(self.batch_size, self.num_critic_obs)
            
            # Reset critic hidden state for each step (since it's independent)
            self.network.critic.reset_full_hidden_state(self.batch_size)
            
            actions = self.network.act(obs_t)
            values = self.network.evaluate(critic_obs_t)
            
            actions_list.append(actions)
            values_list.append(values)
            
            assert actions.shape == (self.batch_size, self.num_actions)
            assert torch.isfinite(actions).all()
            assert torch.isfinite(values).all()
        
        # Check that actions change over time (due to RNN state)
        first_actions = actions_list[0]
        last_actions = actions_list[-1]
        
        # Actions should generally be different due to changing RNN states
        # (though they might occasionally be close, so we use a loose check)
        mean_diff = torch.abs(first_actions - last_actions).mean()
        assert mean_diff > 1e-6, "Actions should change over sequence due to RNN"
    
    def test_hidden_state_management(self):
        """Test hidden state retrieval"""
        # Reset and process some data
        self.network.reset()
        self.network.act(self.actor_obs)
        
        # Get hidden states
        actor_hidden, critic_hidden = self.network.get_hidden_states()
        
        assert actor_hidden is not None
        # Critic hidden should be None as returned by get_hidden_states
        assert critic_hidden is None
    
    def test_different_noise_types(self):
        """Test different noise standard deviation types"""
        # Test scalar noise type
        network_scalar = Quantile_NN(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            noise_std_type="scalar"
        )
        
        network_scalar.reset()
        actions_scalar = network_scalar.act(self.actor_obs)
        assert actions_scalar.shape == (self.batch_size, self.num_actions)
        
        # Test log noise type
        network_log = Quantile_NN(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            noise_std_type="log"
        )
        
        network_log.reset()
        actions_log = network_log.act(self.actor_obs)
        assert actions_log.shape == (self.batch_size, self.num_actions)
    
    def test_state_dict_operations(self):
        """Test state dictionary save/load"""
        # Get initial state
        initial_state = self.network.state_dict()
        
        # Modify network
        self.network.act(self.actor_obs)
        self.network.evaluate(self.critic_obs)
        
        # Save state and create new network
        network_copy = Quantile_NN(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions
        )
        
        # Load state
        result = network_copy.load_state_dict(initial_state)
        assert result == True  # Should return True indicating successful load
    
    def test_gradient_flow(self):
        """Test gradient flow through the entire network"""
        # Reset network state
        self.network.reset()
        self.network.critic.reset_full_hidden_state(self.batch_size)
        
        # Forward pass
        actions = self.network.act(self.actor_obs)
        values = self.network.evaluate(self.critic_obs)
        
        # Compute simple loss
        actor_loss = actions.sum()
        critic_loss = values.sum()
        total_loss = actor_loss + critic_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"
    
    def test_device_consistency(self):
        """Test device consistency"""
        device = torch.device("cpu")
        
        # Move network to device
        self.network = self.network.to(device)
        
        # Reset network state
        self.network.reset()
        self.network.critic.reset_full_hidden_state(self.batch_size)
        
        # Move data to device
        actor_obs_device = self.actor_obs.to(device)
        critic_obs_device = self.critic_obs.to(device)
        
        # Test operations
        actions = self.network.act(actor_obs_device)
        values = self.network.evaluate(critic_obs_device)
        
        assert actions.device == device
        assert values.device == device
    
    def test_batch_size_variations(self):
        """Test network with different batch sizes"""
        batch_sizes = [1, 3, 8, 16]
        
        for bs in batch_sizes:
            self.network.reset()
            self.network.critic.reset_full_hidden_state(bs)
            
            obs = torch.randn(bs, self.num_actor_obs)
            critic_obs = torch.randn(bs, self.num_critic_obs)
            
            actions = self.network.act(obs)
            values = self.network.evaluate(critic_obs)
            
            assert actions.shape == (bs, self.num_actions)
            assert values.shape[0] == bs  # First dimension should match batch size
            assert torch.isfinite(actions).all()
            assert torch.isfinite(values).all()


class TestIntegration:
    """Integration tests for the complete quantile network system"""
    
    def test_training_step_simulation(self):
        """Simulate a complete training step"""
        network = Quantile_NN(
            num_actor_obs=10,
            num_critic_obs=12,
            num_actions=4,
            quantile_count=50
        )
        
        batch_size = 8
        network.reset()
        network.critic.reset_full_hidden_state(batch_size)
        
        # Generate sample data
        obs = torch.randn(batch_size, 10)
        critic_obs = torch.randn(batch_size, 12)
        rewards = torch.randn(batch_size)
        
        # Forward pass
        actions = network.act(obs)
        values = network.evaluate(critic_obs)
        quantiles = network.evaluate_quantiles(critic_obs)
        
        # Compute sample losses
        log_probs = network.get_actions_log_prob(actions)
        entropy = network.entropy
        
        # Simple loss computation
        actor_loss = -(log_probs * rewards).mean() - 0.01 * entropy.mean()
        critic_loss = ((values.squeeze() - rewards) ** 2).mean()
        
        total_loss = actor_loss + critic_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that all went well
        assert torch.isfinite(total_loss)
        assert not torch.isnan(total_loss)
        
        # Check gradients
        has_gradients = False
        for param in network.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                assert torch.isfinite(param.grad).all()
        
        assert has_gradients, "Network should have some gradients after backward pass"
    
    def test_multi_environment_simulation(self):
        """Test network with multiple parallel environments"""
        network = Quantile_NN(
            num_actor_obs=15,
            num_critic_obs=15,
            num_actions=3,
            quantile_count=25
        )
        
        num_envs = 6
        seq_length = 20
        
        network.reset()
        
        # Simulate episode
        for step in range(seq_length):
            obs = torch.randn(num_envs, 15)
            
            # Reset critic hidden state for each step
            network.critic.reset_full_hidden_state(num_envs)
            
            # Some environments might be done
            dones = torch.zeros(num_envs, dtype=torch.bool)
            if step > 0 and step % 7 == 0:  # Reset some envs periodically
                dones[::2] = True  # Reset every other environment
                network.reset(dones)
            
            actions = network.act(obs)
            values = network.evaluate(obs)
            
            assert actions.shape == (num_envs, 3)
            assert values.shape[0] == num_envs
            assert torch.isfinite(actions).all()
            assert torch.isfinite(values).all()


def run_all_tests():
    """Run all tests and report results"""
    test_classes = [TestQuantileNN, TestIntegration]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== Testing {test_class.__name__} ===")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                # Setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"PASS {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {test_method_name}: {str(e)}")
                # Uncomment the next line for detailed error info during debugging
                # import traceback; traceback.print_exc()
    
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")


if __name__ == "__main__":
    run_all_tests()