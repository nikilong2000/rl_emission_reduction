"""
Validation script for simulation.py

This script tests the SimulationModel with constant inputs and plots the output.
"""
import numpy as np
import matplotlib.pyplot as plt
from simulation import create_simulation_from_directory
import os


def validate_simulation():
    """Test the simulation with constant inputs and plot results."""
    
    # Paths to models (adjust based on your setup)
    base_path = "/home/runner/work/rl_emission_reduction/rl_emission_reduction/controller_for_ICE_PG"
    ice_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/ICE")
    pg_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/PG")
    
    # Check if paths exist
    if not os.path.exists(ice_dir):
        ice_dir = os.path.join(base_path, "src/models_markus/ICE_Model_Update_01")
    if not os.path.exists(pg_dir):
        pg_dir = os.path.join(base_path, "src/models_markus/PG_v3")
    
    print(f"Loading models from:")
    print(f"  ICE: {ice_dir}")
    print(f"  PG:  {pg_dir}")
    
    # Create simulation
    sim = create_simulation_from_directory(ice_dir, pg_dir)
    
    # Run simulation with constant inputs
    n_steps = 100
    results = {
        'car_speed': [],
        'ice_torque': [],
        'soc': [],
        'no': [],
        'co': []
    }
    
    # Constant inputs
    ice_speed_rpm = 1500.0
    fuel_mg = 15.0
    em2_torque = 0.0
    brake_perc = 0.0
    
    print(f"\nRunning simulation for {n_steps} steps with constant inputs:")
    print(f"  ICE Speed: {ice_speed_rpm} RPM")
    print(f"  Fuel: {fuel_mg} mg")
    print(f"  EM2 Torque: {em2_torque} Nm")
    print(f"  Brake: {brake_perc}%")
    
    for i in range(n_steps):
        output = sim.step(
            ice_speed_rpm=ice_speed_rpm,
            fuel_mg=fuel_mg,
            em2_torque_Nm=em2_torque,
            brake_perc=brake_perc
        )
        
        results['car_speed'].append(output['car_speed'])
        results['ice_torque'].append(output['ice_torque'])
        results['soc'].append(output['soc'])
        results['no'].append(output['no'])
        results['co'].append(output['co'])
        
        if i % 20 == 0:
            print(f"  Step {i:3d}: Speed={output['car_speed']:.2f} km/h, "
                  f"Torque={output['ice_torque']:.2f} Nm, SOC={output['soc']:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Simulation Validation - Constant Inputs', fontsize=14)
    
    # Car speed
    axes[0, 0].plot(results['car_speed'], 'b-')
    axes[0, 0].set_title('Car Speed')
    axes[0, 0].set_ylabel('Speed (km/h)')
    axes[0, 0].grid(True)
    
    # ICE torque
    axes[0, 1].plot(results['ice_torque'], 'r-')
    axes[0, 1].set_title('ICE Torque')
    axes[0, 1].set_ylabel('Torque (Nm)')
    axes[0, 1].grid(True)
    
    # SOC
    axes[1, 0].plot(results['soc'], 'g-')
    axes[1, 0].set_title('State of Charge (SOC)')
    axes[1, 0].set_ylabel('SOC')
    axes[1, 0].grid(True)
    
    # NO emissions
    axes[1, 1].plot(results['no'], 'm-')
    axes[1, 1].set_title('NO Emissions')
    axes[1, 1].set_ylabel('NO')
    axes[1, 1].grid(True)
    
    # CO emissions
    axes[2, 0].plot(results['co'], 'c-')
    axes[2, 0].set_title('CO Emissions')
    axes[2, 0].set_ylabel('CO')
    axes[2, 0].set_xlabel('Timestep')
    axes[2, 0].grid(True)
    
    # Hide the last subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = '/home/runner/work/rl_emission_reduction/rl_emission_reduction/v2_rebuild/simulation_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Test reset
    print("\nTesting reset functionality...")
    sim.reset_states()
    output_after_reset = sim.step(ice_speed_rpm, fuel_mg, em2_torque, brake_perc)
    print(f"  After reset: Speed={output_after_reset['car_speed']:.2f} km/h, "
          f"Torque={output_after_reset['ice_torque']:.2f} Nm")
    
    print("\n✅ Simulation validation complete!")
    return sim, results


if __name__ == "__main__":
    sim, results = validate_simulation()
