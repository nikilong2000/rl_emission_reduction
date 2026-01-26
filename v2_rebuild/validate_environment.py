"""
Validation script for environment.py

This script tests the environment with random actions and verifies:
1. Observation space dimensions (should be 6 including error)
2. State shapes are correct
3. Reward calculations work properly
"""
import numpy as np
import os

# Note: This validation script cannot run until the Keras compatibility issue is resolved
# It's provided as a template for when models are available

def validate_environment():
    """Test environment with random actions."""
    from environment import VehicleControlEnvironment
    
    # Paths to models
    base_path = "/home/runner/work/rl_emission_reduction/rl_emission_reduction/controller_for_ICE_PG"
    ice_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/ICE")
    pg_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/PG")
    data_dir = os.path.join(base_path, "src/data")
    
    # Check for alternative model paths
    if not os.path.exists(ice_dir):
        ice_dir = os.path.join(base_path, "src/models_markus/ICE_Model_Update_01")
    if not os.path.exists(pg_dir):
        pg_dir = os.path.join(base_path, "src/models_markus/PG_v3")
    
    print("="*70)
    print("Environment Validation")
    print("="*70)
    print(f"ICE model: {ice_dir}")
    print(f"PG model:  {pg_dir}")
    print(f"Data dir:  {data_dir}")
    print()
    
    # Create environment
    print("Creating environment...")
    env = VehicleControlEnvironment(
        ice_model_dir=ice_dir,
        pg_model_dir=pg_dir,
        data_dir=data_dir,
        max_steps=100
    )
    print(f"✓ Environment created")
    print(f"  Observation space dim: {env.observation_space_dim}")
    print(f"  Action space dim: {env.action_space_dim}")
    print()
    
    # Test observation space dimension
    assert env.observation_space_dim == 6, f"Expected obs dim 6, got {env.observation_space_dim}"
    assert env.action_space_dim == 3, f"Expected action dim 3, got {env.action_space_dim}"
    print("✓ Observation and action space dimensions correct")
    print()
    
    # Reset environment
    print("Testing reset...")
    obs = env.reset(vel_target=70.0)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"
    print("✓ Reset works correctly")
    print()
    
    # Verify observation structure
    vel_target, vel, mf, brk, ice_sp, error = obs
    print("Observation breakdown:")
    print(f"  [0] vel_target: {vel_target:.2f} km/h")
    print(f"  [1] vel:        {vel:.2f} km/h")
    print(f"  [2] mf:         {mf:.2f} mg")
    print(f"  [3] brk:        {brk:.2f} %")
    print(f"  [4] ice_sp:     {ice_sp:.2f} RPM")
    print(f"  [5] error:      {error:.2f} km/h (vel_target - vel)")
    print()
    
    # Verify error calculation
    expected_error = vel_target - vel
    assert abs(error - expected_error) < 0.01, f"Error mismatch: {error} vs {expected_error}"
    print(f"✓ Error calculation correct: {vel_target:.2f} - {vel:.2f} = {error:.2f}")
    print()
    
    # Test stepping with random actions
    print("Testing environment steps with random actions...")
    n_steps = 10
    
    for i in range(n_steps):
        # Random action (scaled appropriately)
        action = np.array([
            np.random.uniform(3.0, 20.0),    # mf
            np.random.uniform(0.0, 100.0),   # brk
            np.random.uniform(800.0, 3000.0) # ice_sp
        ], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify observation shape
        assert obs.shape == (6,), f"Step {i}: Expected obs shape (6,), got {obs.shape}"
        
        # Verify observation values
        vel_target_obs, vel_obs, mf_obs, brk_obs, ice_sp_obs, error_obs = obs
        
        # Verify error
        expected_error = vel_target_obs - vel_obs
        assert abs(error_obs - expected_error) < 0.01, f"Step {i}: Error mismatch"
        
        # Verify reward is in valid range
        assert 0.0 <= reward <= 1.0, f"Step {i}: Reward {reward} out of range [0, 1]"
        
        if i % 2 == 0:
            print(f"  Step {i:2d}: vel={vel_obs:6.2f}, error={error_obs:6.2f}, reward={reward:.4f}, "
                  f"term={terminated}, trunc={truncated}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {i}")
            break
    
    print()
    print("✓ All steps completed successfully")
    print()
    
    # Test multiple episodes
    print("Testing multiple episode resets...")
    for episode in range(3):
        obs = env.reset(vel_target=70.0 + episode * 10)
        assert obs.shape == (6,), f"Episode {episode}: Invalid obs shape"
        print(f"  Episode {episode}: reset successful, vel_target={obs[0]:.1f}")
    
    print()
    print("="*70)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*70)
    print()
    print("Key Fixes Verified:")
    print("  ✓ Observation space includes Error (6th dimension)")
    print("  ✓ Error = vel_target - vel")
    print("  ✓ Observation shape is (6,)")
    print("  ✓ Reward calculation works correctly")
    print("  ✓ Reset and step functions work as expected")
    print()


if __name__ == "__main__":
    try:
        validate_environment()
    except Exception as e:
        print(f"\n❌ Validation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print()
        print("Note: If this is a Keras compatibility error, the environment")
        print("      interface is still correct but models need to be updated.")
        import traceback
        traceback.print_exc()
