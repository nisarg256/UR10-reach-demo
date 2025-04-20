#!/usr/bin/env python3
import os
import sys
import pickle
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.visualization.render import visualize_reachability

def main():
    print("UR10e Reachability Visualization")
    print("="*50)
    
    # Check if reachability data exists
    data_path = os.path.join(project_root, "reachability_data.pkl")
    if not os.path.exists(data_path):
        print(f"Reachability data not found at {data_path}")
        print("Please run 'python3 scripts/calculate_workspace.py' first.")
        return
    
    # Load the reachability data
    print(f"Loading reachability data from {data_path}")
    with open(data_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Check if we have the new or old data format
    if isinstance(saved_data, dict) and 'reachability_data' in saved_data:
        # New format
        reachability_data = saved_data['reachability_data']
        config = saved_data.get('config', {})
        resolution = config.get('workspace', {}).get('resolution', 0.1)
        
        # Print analysis info
        print(f"Analysis completed in {saved_data.get('analysis_time', 'N/A')} seconds")
        print(f"Grid resolution: {resolution}m")
    else:
        # Old format
        reachability_data = saved_data
        resolution = 0.1  # Default value
        print("Using legacy format reachability data")
    
    # Count reachable points
    flat_count = sum(sum(reachability_data['flat_reachable']))
    perp_count = sum(sum(reachability_data['perp_reachable']))
    both_count = sum(sum(reachability_data['flat_reachable'] & reachability_data['perp_reachable']))
    total_points = reachability_data['flat_reachable'].size
    
    print(f"Flat robot can reach {flat_count} points ({flat_count / total_points * 100:.1f}% of workspace)")
    print(f"Perpendicular robot can reach {perp_count} points ({perp_count / total_points * 100:.1f}% of workspace)")
    print(f"Both robots can reach {both_count} points ({both_count / total_points * 100:.1f}% of workspace)")
    
    # Visualize the results
    print("Visualizing reachability map...")
    title = f"UR10e Wall Reachability Analysis (Resolution: {resolution}m)"
    visualize_reachability(reachability_data, title=title)
    
    print("Visualization complete.")
    print("Run 'python3 scripts/demonstrate_reach.py' to demonstrate the robot motion.")

if __name__ == "__main__":
    main() 