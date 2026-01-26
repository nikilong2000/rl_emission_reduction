"""
Validation script for the Simulation component.
Tests the simulation with constant inputs and plots the output.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation


def validate_simulation():
    """Run simulation with constant inputs and visualize results."""

    # Paths to models (adjust if needed)
    ice_path = "../controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    pg_path = "../controller_for_ICE_PG/src/models_markus/PG_v2"

    print("=" * 70)
    print("SIMULATION VALIDATION")
    print("=" * 70)

    # Initialize simulation
    print("\n1. Initializing simulation...")
    sim = Simulation(ice_model_path=ice_path, pg_model_path=pg_path, soc_initial=0.7)

    # Reset simulation
    print("\n2. Resetting simulation...")
    initial_state = sim.reset()
    print(
        f"   Initial state: velocity={initial_state['velocity'][0]:.2f}, SOC={initial_state['soc'][0]:.3f}"
    )

    # Run simulation with constant actions
    print("\n3. Running simulation with constant actions...")
    n_steps = 200

    # Constant actions (moderate values)
    mf_action = 0.3  # 30% motor front torque
    brk_action = 0.0  # No braking
    ice_sp_action = 0.5  # 50% ICE speed

    # Storage for results
    results = {
        "velocity": [],
        "soc": [],
        "torque": [],
        "no": [],
        "no2": [],
        "co": [],
        "co2": [],
    }

    for step in range(n_steps):
        state = sim.step(mf_action, brk_action, ice_sp_action)

        # Store results
        for key in results.keys():
            results[key].append(state[key][0])

        if step % 50 == 0:
            print(
                f"   Step {step:3d}: vel={state['velocity'][0]:6.2f}, SOC={state['soc'][0]:.3f}, torque={state['torque'][0]:6.2f}"
            )

    # Convert to numpy arrays
    for key in results.keys():
        results[key] = np.array(results[key])

    # Plot results
    print("\n4. Plotting results...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"Simulation Validation (MF={mf_action}, BRK={brk_action}, ICE_SP={ice_sp_action})",
        fontsize=14,
        fontweight="bold",
    )

    # Velocity
    axes[0, 0].plot(results["velocity"], "b-", linewidth=2)
    axes[0, 0].set_ylabel("Velocity [km/h]")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Vehicle Velocity")

    # SOC
    axes[0, 1].plot(results["soc"], "g-", linewidth=2)
    axes[0, 1].set_ylabel("SOC [-]")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("State of Charge")
    axes[0, 1].axhline(y=0.2, color="r", linestyle="--", alpha=0.5, label="Min SOC")
    axes[0, 1].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Max SOC")
    axes[0, 1].legend()

    # Torque
    axes[1, 0].plot(results["torque"], "orange", linewidth=2)
    axes[1, 0].set_ylabel("Torque [Nm]")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("ICE Torque")

    # Emissions - NO
    axes[1, 1].plot(results["no"], "r-", linewidth=2, label="NO")
    axes[1, 1].plot(results["no2"], "purple", linewidth=2, label="NO2")
    axes[1, 1].set_ylabel("Emissions [g/s]")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("NOx Emissions")
    axes[1, 1].legend()

    # Emissions - CO
    axes[2, 0].plot(results["co"], "brown", linewidth=2)
    axes[2, 0].set_ylabel("CO [g/s]")
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title("CO Emissions")

    # Emissions - CO2
    axes[2, 1].plot(results["co2"], "darkgreen", linewidth=2)
    axes[2, 1].set_ylabel("CO2 [g/s]")
    axes[2, 1].set_xlabel("Step")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title("CO2 Emissions")

    plt.tight_layout()

    # Save plot
    output_file = "simulation_validation.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"   Plot saved to: {output_file}")

    # Summary statistics
    print("\n5. Summary Statistics:")
    print(
        f"   Velocity: mean={results['velocity'].mean():.2f}, std={results['velocity'].std():.2f}, "
        f"min={results['velocity'].min():.2f}, max={results['velocity'].max():.2f}"
    )
    print(
        f"   SOC:      mean={results['soc'].mean():.3f}, std={results['soc'].std():.3f}, "
        f"min={results['soc'].min():.3f}, max={results['soc'].max():.3f}"
    )
    print(
        f"   Torque:   mean={results['torque'].mean():.2f}, std={results['torque'].std():.2f}, "
        f"min={results['torque'].min():.2f}, max={results['torque'].max():.2f}"
    )

    print("\n" + "=" * 70)
    print("✓ SIMULATION VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    validate_simulation()
