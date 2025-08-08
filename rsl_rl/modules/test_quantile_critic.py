import torch
import torch.nn as nn
import numpy as np
import pytest
from rsl_rl.modules.quantile_nn_temp import QuantileCritic, energy_loss, risk_measure_wang


class TestQuantileCritic:
    """Test suite for QuantileCritic class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.input_dim = 10
        self.output_dim = 1
        self.hidden_dims = [64, 32]
        self.quantile_count = 50
        self.batch_size = 4
        
        self.critic = QuantileCritic(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            quantile_count=self.quantile_count,
            recurrent_layers=1,
            activations=[nn.ReLU, nn.ReLU, nn.Tanh],
        )
        
        # Sample input data
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """Test proper initialization of QuantileCritic"""
        assert self.critic._quantile_count == self.quantile_count
        assert len(self.critic._quantile_layers) == self.output_dim
        assert self.critic._tau.shape[0] == self.quantile_count + 1
        assert self.critic._tau_hat.shape[0] == self.quantile_count
        
        # Check if LSTM is properly initialized
        assert isinstance(self.critic._features[0], nn.LSTM)
        assert self.critic._features[0].input_size == self.input_dim
        assert self.critic._features[0].hidden_size == self.hidden_dims[0]
    
    def test_forward_pass_values(self):
        """Test forward pass returning values (not distributions)"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        output = self.critic(self.sample_input, distribution=False)
        
        # Check output shape - should be batch_size x output_dim
        assert output.shape == (self.batch_size, self.output_dim)
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_forward_pass_distributions(self):
        """Test forward pass returning quantile distributions"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        quantiles = self.critic(self.sample_input, distribution=True)
        
        # Check output shape - should be batch_size x output_dim x quantile_count
        expected_shape = (self.batch_size, self.output_dim, self.quantile_count)
        assert quantiles.shape == expected_shape, f"Expected {expected_shape}, got {quantiles.shape}"
        assert torch.isfinite(quantiles).all(), "Quantiles contain non-finite values"
    
    def test_hidden_state_management(self):
        """Test hidden state reset and management"""
        # Test full reset
        self.critic.reset_full_hidden_state(self.batch_size)
        assert self.critic.hidden_state is not None
        assert self.critic.hidden_state[0].shape == (1, self.batch_size, self.hidden_dims[0])
        
        # Test partial reset
        indices_to_reset = torch.tensor([0, 2])
        old_hidden = self.critic.hidden_state[0].clone()
        self.critic.reset_hidden_state(indices_to_reset)
        
        # Check that specified indices were reset
        assert torch.allclose(self.critic.hidden_state[0][:, indices_to_reset], 
                            torch.zeros_like(self.critic.hidden_state[0][:, indices_to_reset]))
        # Check that other indices remain unchanged
        assert torch.allclose(self.critic.hidden_state[0][:, 1], old_hidden[:, 1])
        assert torch.allclose(self.critic.hidden_state[0][:, 3], old_hidden[:, 3])
    
    def test_make_diracs(self):
        """Test dirac delta generation"""
        values = torch.randn(self.batch_size, self.output_dim)
        diracs = self.critic.make_diracs(values)
        
        expected_shape = (self.batch_size, self.output_dim, self.quantile_count)
        assert diracs.shape == expected_shape
        
        # Check that all quantiles equal the input values
        for i in range(self.batch_size):
            for j in range(self.output_dim):
                assert torch.allclose(diracs[i, j], values[i, j])
    
    def test_quantiles_to_values_conversion(self):
        """Test conversion from quantiles to values using risk measure"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        # Get quantiles
        quantiles = self.critic(self.sample_input, distribution=True)
        
        # Convert to values
        values = self.critic.quantiles_to_values(quantiles)
        
        assert values.shape == (self.batch_size, self.output_dim)
        assert torch.isfinite(values).all()
    
    def test_measure_with_custom_args(self):
        """Test risk measure with custom arguments"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        quantiles = self.critic(self.sample_input, distribution=True)
        
        # Test with different beta values for Wang's risk measure
        beta_values = torch.tensor([0.5, -0.3, 0.0, 1.0])
        values = self.critic.quantiles_to_values(quantiles, beta_values)
        
        assert values.shape == (beta_values.shape[0], self.output_dim)
        assert torch.isfinite(values).all()
    
    def test_device_consistency(self):
        """Test device consistency across operations"""
        device = torch.device("cpu")  # Use CPU for testing
        self.critic = self.critic.to(device)
        input_data = self.sample_input.to(device)
        
        self.critic.reset_full_hidden_state(self.batch_size)
        output = self.critic(input_data)
        
        assert output.device == device
        assert self.critic.device == device
    
    def test_sequential_forward_passes(self):
        """Test multiple sequential forward passes with recurrent state"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        outputs = []
        for _ in range(5):
            output = self.critic(self.sample_input)
            outputs.append(output)
            assert torch.isfinite(output).all()
        
        # Outputs should be different due to changing hidden states
        assert not torch.allclose(outputs[0], outputs[-1], atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        self.critic.reset_full_hidden_state(self.batch_size)
        
        output = self.critic(self.sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for parameters
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"


class TestEnergyLoss:
    """Test suite for energy loss function"""
    
    def test_energy_loss_basic(self):
        """Test basic energy loss computation"""
        batch_size, quantile_count = 4, 50
        predictions = torch.randn(batch_size, quantile_count)
        targets = torch.randn(batch_size, quantile_count)
        
        loss = energy_loss(predictions, targets)
        
        assert loss.shape == torch.Size([])  # Scalar loss
        assert torch.isfinite(loss)
    
    def test_energy_loss_identical_inputs(self):
        """Test energy loss with identical predictions and targets"""
        batch_size, quantile_count = 4, 50
        data = torch.randn(batch_size, quantile_count)
        
        loss = energy_loss(data, data)
        
        # Loss should be 0 when predictions equal targets
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_energy_loss_gradient_flow(self):
        """Test gradient flow through energy loss"""
        batch_size, quantile_count = 4, 50
        predictions = torch.randn(batch_size, quantile_count, requires_grad=True)
        targets = torch.randn(batch_size, quantile_count)
        
        loss = energy_loss(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()


class TestRiskMeasures:
    """Test suite for risk measures"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.input_dim = 5
        self.quantile_count = 100
        self.critic = QuantileCritic(
            input_dim=self.input_dim,
            output_dim=1,
            quantile_count=self.quantile_count,
        )
    
    def test_wang_risk_measure(self):
        """Test Wang's risk measure"""
        beta = 0.5
        measure_func = risk_measure_wang(self.critic, beta)
        
        # Test with sample quantiles
        batch_size = 3
        quantiles = torch.randn(batch_size, 1, self.quantile_count)
        
        risk_values = measure_func(quantiles)
        
        assert risk_values.shape == (batch_size, 1)
        assert torch.isfinite(risk_values).all()
    
    def test_wang_risk_measure_tensor_beta(self):
        """Test Wang's risk measure with tensor beta"""
        beta = torch.tensor([0.0, 0.5, -0.3])
        measure_func = risk_measure_wang(self.critic, beta)
        
        batch_size = 1
        quantiles = torch.randn(batch_size, 1, self.quantile_count)
        
        risk_values = measure_func(quantiles)
        
        assert risk_values.shape == (beta.shape[0], 1)
        assert torch.isfinite(risk_values).all()


def run_all_tests():
    """Run all tests and report results"""
    test_classes = [TestQuantileCritic, TestEnergyLoss, TestRiskMeasures]
    
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
                
                print(f"✅ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {test_method_name}: {str(e)}")
    
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")


if __name__ == "__main__":
    run_all_tests()