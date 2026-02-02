"""
Validation script for the Environment component.
Tests reset, step, and verifies observation/action shapes and reward calculations.
"""

import numpy as np
from simulation import Simulation
from environment import VehicleEnvironment


def validate_environment():
    """Validate environment with random actions."""

    # Paths to models
    ice_path = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE"
    pg_path = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/PG"

    print("=" * 70)
    print("ENVIRONMENT VALIDATION")
    print("=" * 70)

    # Initialize simulation
    print("\n1. Initializing simulation...")
    sim = Simulation(ice_model_path=ice_path, pg_model_path=pg_path, soc_initial=0.7)

    # Initialize environment
    print("\n2. Initializing environment...")
    env = VehicleEnvironment(simulation=sim, max_steps=100, vel_target=70.0)

    # Validate spaces
    print("\n3. Validating spaces...")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    assert env.observation_space.shape == (
        7,
    ), f"Expected obs shape (7,), got {env.observation_space.shape}"
    print("   ✓ Spaces validated!")

    # Test reset
    print("\n4. Testing reset...")
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial observation: {obs}")
    print(f"   Initial info: {info}")
    assert obs.shape == (7,), f"Expected obs shape (7,), got {obs.shape}"
    print("   ✓ Reset validated!")

    # Test step with random actions
    print("\n5. Testing step with random actions...")
    n_steps = 50
    total_reward = 0.0

    for i in range(n_steps):
        # Sample random action
        action = env.action_space.sample()

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Validate shapes
        assert next_obs.shape == (
            7,
        ), f"Step {i}: Expected obs shape (7,), got {next_obs.shape}"
        assert isinstance(reward, (int, float)), f"Step {i}: Reward should be numeric"
        assert isinstance(terminated, bool), f"Step {i}: terminated should be bool"
        assert isinstance(truncated, bool), f"Step {i}: truncated should be bool"

        if i % 10 == 0:
            print(
                f"   Step {i:3d}: obs_shape={next_obs.shape}, reward={reward:.3f}, "
                f"vel={info['velocity']:.2f}, error={info['error']:.2f}"
            )

        if terminated or truncated:
            print(
                f"   Episode ended at step {i}: terminated={terminated}, truncated={truncated}"
            )
            break

    print(f"\n   Total reward over {i+1} steps: {total_reward:.2f}")
    print("   ✓ Step validated!")

    # Test observation components
    print("\n6. Testing observation components...")
    obs, _ = env.reset()
    action = np.array([0.5, -0.2, 0.3, 0.0], dtype=np.float32)
    next_obs, reward, _, _, info = env.step(action)

    print(f"   Observation breakdown:")
    print(f"   - vel_target_norm: {next_obs[0]:.4f} (target: {env.vel_target} km/h)")
    print(
        f"   - velocity_norm:   {next_obs[1]:.4f} (actual: {info['velocity']:.2f} km/h)"
    )
    print(f"   - mf (action[0]):  {next_obs[2]:.4f}")
    print(f"   - brk (action[1]): {next_obs[3]:.4f}")
    print(f"   - ice_sp (action[2]): {next_obs[4]:.4f}")
    print(f"   - em2_torque (action[3]): {next_obs[5]:.4f}")
    print(f"   - error_norm:      {next_obs[6]:.4f} (raw: {info['error']:.2f} km/h)")

    # Verify error calculation
    expected_error_norm = (env.vel_target - info["velocity"]) / env.max_error
    assert abs(next_obs[6] - expected_error_norm) < 1e-5, "Error calculation mismatch"
    print("   ✓ Observation components validated!")

    # Test reward calculation
    print("\n7. Testing reward calculation...")
    error_norm = info["error_norm"]
    expected_reward = env.alpha * (1.0 - error_norm**2)
    assert (
        abs(reward - expected_reward) < 1e-5
    ), f"Reward mismatch: {reward} vs {expected_reward}"
    print(f"   Reward formula: α * (1 - error_norm²)")
    print(
        f"   Calculated: {env.alpha} * (1 - {error_norm:.4f}²) = {expected_reward:.4f}"
    )
    print(f"   Actual reward: {reward:.4f}")
    print("   ✓ Reward calculation validated!")

    print("\n" + "=" * 70)
    print("✓ ENVIRONMENT VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_environment()
