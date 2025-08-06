import sys
import time
import torch
import traceback
from pathlib import Path

# Add the directory containing your quantile_nn module to path
# Adjust this path as needed based on your project structure
sys.path.append('.')

try:
    from test_quantile_critic import run_all_tests as run_critic_tests
    from test_quantile_nn import run_all_tests as run_nn_tests
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure both test files are in the same directory as this script.")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = ['torch', 'numpy']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Missing required modules: {', '.join(missing)}")
        print("Please install them using: pip install torch numpy")
        return False
    
    return True


def print_system_info():
    """Print system and PyTorch information"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device being used: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()


def run_performance_test():
    """Run basic performance tests"""
    print("=== Performance Test ===")
    
    # Import here to avoid circular imports
    from quantile_nn import Quantile_NN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a moderately sized network
    network = Quantile_NN(
        num_actor_obs=50,
        num_critic_obs=50,
        num_actions=10,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        quantile_count=200,
        rnn_hidden_dim=256
    ).to(device)
    
    batch_size = 32
    network.reset()
    
    # Warm up
    obs = torch.randn(batch_size, 50, device=device)
    for _ in range(5):
        _ = network.act(obs)
        _ = network.evaluate(obs)
    
    # Time forward passes
    n_iterations = 100
    
    # Actor timing
    start_time = time.time()
    for _ in range(n_iterations):
        actions = network.act(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    actor_time = (time.time() - start_time) / n_iterations
    
    # Critic timing
    start_time = time.time()
    for _ in range(n_iterations):
        values = network.evaluate(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    critic_time = (time.time() - start_time) / n_iterations
    
    # Quantile critic timing
    start_time = time.time()
    for _ in range(n_iterations):
        quantiles = network.evaluate_quantiles(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    quantile_time = (time.time() - start_time) / n_iterations
    
    print(f"Actor forward pass: {actor_time*1000:.2f} ms")
    print(f"Critic forward pass (values): {critic_time*1000:.2f} ms")
    print(f"Critic forward pass (quantiles): {quantile_time*1000:.2f} ms")
    print(f"Device: {device}")
    print()


def run_memory_test():
    """Run memory usage tests"""
    print("=== Memory Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory test")
        return
    
    from quantile_nn import Quantile_NN
    
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Create network
    network = Quantile_NN(
        num_actor_obs=100,
        num_critic_obs=100,
        num_actions=20,
        quantile_count=500,  # Large quantile count
        rnn_hidden_dim=512   # Large hidden size
    ).to(device)
    
    network_memory = torch.cuda.memory_allocated() - initial_memory
    
    # Process data
    batch_size = 64
    network.reset()
    obs = torch.randn(batch_size, 100, device=device)
    
    actions = network.act(obs)
    values = network.evaluate(obs)
    quantiles = network.evaluate_quantiles(obs)
    
    total_memory = torch.cuda.memory_allocated() - initial_memory
    
    print(f"Network parameters: {network_memory / 1024**2:.1f} MB")
    print(f"Total memory usage: {total_memory / 1024**2:.1f} MB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Clean up
    del network, actions, values, quantiles, obs
    torch.cuda.empty_cache()
    print()

def run_comprehensive_test():
    """Run a comprehensive test combining all components"""
    print("=== Comprehensive Integration Test ===")
    
    try:
        from quantile_nn import Quantile_NN, energy_loss
        
        # Create network
        network = Quantile_NN(
            num_actor_obs=20,
            num_critic_obs=25,
            num_actions=6,
            quantile_count=100
        )
        
        batch_size = 8
        seq_len = 10
        network.reset()
        
        # Simulate training episode
        total_reward = 0
        
        for step in range(seq_len):
            # Generate observations
            actor_obs = torch.randn(batch_size, 20)
            critic_obs = torch.randn(batch_size, 25)
            
            # Forward pass
            actions = network.act(actor_obs)
            values = network.evaluate(critic_obs)
            quantiles_pred = network.evaluate_quantiles(critic_obs)
            
            # Simulate rewards and next values
            rewards = torch.randn(batch_size)
            next_values = network.evaluate(critic_obs)  # Simplified
            
            # Compute losses
            log_probs = network.get_actions_log_prob(actions)
            entropy = network.entropy
            
            # Actor loss (simplified policy gradient)
            advantages = rewards - values.squeeze()
            actor_loss = -(log_probs * advantages.detach()).mean()
            entropy_loss = -0.01 * entropy.mean()
            
            # Critic loss using energy loss for quantiles
            target_quantiles = network.critic.make_diracs(rewards.unsqueeze(-1))
            quantile_loss = energy_loss(quantiles_pred.squeeze(1), target_quantiles.squeeze(1))
            
            # Total loss
            total_loss = actor_loss + entropy_loss + quantile_loss
            total_reward += rewards.mean().item()
            
            # Backward pass
            total_loss.backward()
            
            # Simple gradient step (no optimizer for test)
            with torch.no_grad():
                for param in network.parameters():
                    if param.grad is not None:
                        param -= 0.001 * param.grad
                        param.grad.zero_()
            
            # Check for NaN/Inf
            assert torch.isfinite(total_loss), f"Loss became non-finite at step {step}"
            
            # Reset some environments occasionally
            if step % 3 == 0:
                dones = torch.zeros(batch_size, dtype=torch.bool)
                dones[::2] = True
                network.reset(dones)
        
        print(f"✅ Comprehensive test passed!")
        print(f"   Average reward over episode: {total_reward/seq_len:.3f}")
        print(f"   Final loss: {total_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {str(e)}")
        traceback.print_exc()
    print()

def main():
    """Main test runner"""
    print(" Quantile Neural Network Test Suite")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Print system info
    print_system_info()
    
    # Run performance test
    try:
        run_performance_test()
    except Exception as e:
        print(f"Performance test failed: {e}")
    
    # Run memory test
    try:
        run_memory_test()
    except Exception as e:
        print(f"Memory test failed: {e}")
    
    # Run comprehensive test
    run_comprehensive_test()
    
    print("=" * 50)
    print(" Running Unit Tests")
    print("=" * 50)
    
    overall_start = time.time()
    
    try:
        print("Testing QuantileCritic component...")
        run_critic_tests()
        print()
        
        print("Testing Quantile_NN component...")
        run_nn_tests()
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    total_time = time.time() - overall_start
    
    print("=" * 50)
    print(" Test Suite Complete")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()