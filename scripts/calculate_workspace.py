#!/usr/bin/env python3
import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config.config_loader import load_config
from src.reachability.workspace_analysis import ReachabilityAnalyzer
from src.visualization.render import visualize_reachability

def main():
    print("UR10e Reachability Analysis for Drywall Finishing")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Path to the MuJoCo model
    model_path = os.path.join(project_root, "models", "reach_comparison.xml")
    
    # Create the reachability analyzer
    analyzer = ReachabilityAnalyzer(model_path, config=config)
    
    # Parameters for the reachability analysis
    y_range = config.get('workspace', {}).get('y_range', (-0.8, 0.8))
    z_range = config.get('workspace', {}).get('z_range', (0.5, 2.0))
    resolution = config.get('workspace', {}).get('resolution', 0.1)
    
    print(f"Analyzing reachability in the following workspace:")
    print(f"  Wall X position: {analyzer.wall_x} m")
    print(f"  Y range: {y_range} m")
    print(f"  Z range: {z_range} m")
    print(f"  Resolution: {resolution} m")
    
    # Calculate total points 
    grid_y_points = int((y_range[1]-y_range[0])/resolution+1)
    grid_z_points = int((z_range[1]-z_range[0])/resolution+1)
    total_points = grid_y_points * grid_z_points
    
    print(f"This will check {total_points} points.")
    print(f"Estimated analysis time: {total_points * 0.02:.1f} seconds")
    
    # Ask for confirmation
    input("Press Enter to start the analysis (this may take several minutes)...")
    
    # Start time
    start_time = time.time()
    
    print("Analyzing reachability...")
    
    # Run the reachability analysis
    reachability_data = analyzer.analyze_wall_reachability(
        y_range=y_range,
        z_range=z_range,
        resolution=resolution
    )
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Analysis completed in {elapsed_time:.1f} seconds.")
    
    # Count reachable points
    flat_count = np.sum(reachability_data['flat_reachable'])
    perp_count = np.sum(reachability_data['perp_reachable'])
    both_count = np.sum(np.logical_and(reachability_data['flat_reachable'], reachability_data['perp_reachable']))
    total_points = reachability_data['flat_reachable'].size
    
    print(f"Flat robot can reach {flat_count} points ({flat_count / total_points * 100:.1f}% of workspace)")
    print(f"Perpendicular robot can reach {perp_count} points ({perp_count / total_points * 100:.1f}% of workspace)")
    print(f"Both robots can reach {both_count} points ({both_count / total_points * 100:.1f}% of workspace)")
    
    # Save the results and configuration used
    save_data = {
        'reachability_data': reachability_data,
        'config': config,
        'analysis_time': elapsed_time,
        'timestamp': time.time()
    }
    
    save_path = os.path.join(project_root, "reachability_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Reachability data saved to {save_path}")
    
    # Visualize the results
    print("Visualizing reachability map...")
    title = f"UR10e Wall Reachability Analysis (Resolution: {resolution}m)"
    visualize_reachability(reachability_data, title=title)
    
    print("Analysis complete.")
    print("Run 'python3 scripts/demonstrate_reach.py' to demonstrate the robot motion.")

if __name__ == "__main__":
    main() 