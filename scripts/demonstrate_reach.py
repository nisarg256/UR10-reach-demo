#!/usr/bin/env python3
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import mujoco
from src.trajectory.s_pattern import generate_s_pattern
from src.visualization.render import MujocoRenderer

def main():
    print("UR10e Reach Demonstration for Drywall Finishing")
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
        reachability_data = pickle.load(f)
    
    # Generate S-pattern trajectories
    print("Generating S-pattern trajectories...")
    num_passes = 10  # Number of horizontal passes
    trajectory_data = generate_s_pattern(reachability_data, num_passes)
    
    flat_trajectory = trajectory_data['flat_trajectory']
    perp_trajectory = trajectory_data['perp_trajectory']
    
    print(f"Generated {len(flat_trajectory)} points for flat robot")
    print(f"Generated {len(perp_trajectory)} points for perpendicular robot")
    
    # Load the MuJoCo model
    model_path = os.path.join(project_root, "models", "reach_comparison.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Create renderer
    renderer = MujocoRenderer(model, data)
    
    # Create a visualization marker for the current target point
    target_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flat_reach_point")
    
    # Get joint IDs
    flat_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"flat_{joint}")
        for joint in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ]
    
    perp_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"perp_{joint}")
        for joint in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ]
    
    # Ask user which robot to demonstrate
    while True:
        print("\nSelect a robot to demonstrate reachability:")
        print("1. Flat mounted robot")
        print("2. Perpendicular mounted robot")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            demonstrate_robot(model, data, renderer, flat_trajectory, flat_joint_ids, 
                             target_marker_id, "Flat Mounted Robot")
        elif choice == '2':
            demonstrate_robot(model, data, renderer, perp_trajectory, perp_joint_ids,
                             target_marker_id, "Perpendicular Mounted Robot")
        elif choice == '3':
            renderer.close()
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
    
    print("Demonstration complete.")

def demonstrate_robot(model, data, renderer, trajectory, joint_ids, marker_id, robot_name):
    """
    Demonstrate a robot's reachability by following its trajectory.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MujocoRenderer instance
        trajectory: List of points and joint configurations
        joint_ids: List of joint IDs
        marker_id: ID of the site to visualize the current target point
        robot_name: Name of the robot for display
    """
    if not trajectory:
        print(f"No reachable points for {robot_name}")
        return
    
    print(f"\nDemonstrating {robot_name} with {len(trajectory)} reachable points")
    print("Press Ctrl+C to stop the demonstration")
    
    # Zero all joint positions first (to avoid weird visual artifacts)
    for i in range(model.nq):
        data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)
    
    # Visualize the trajectory
    print("Press Enter to start following the S-pattern trajectory...")
    input()
    
    try:
        for i, (point, joint_config) in enumerate(trajectory):
            # Update the progress
            if i % 10 == 0 or i == len(trajectory) - 1:
                print(f"Point {i+1}/{len(trajectory)}: y={point[0]:.2f}, z={point[1]:.2f}")
            
            # Place the target marker at the current point
            data.site_xpos[marker_id] = [model.site_pos[marker_id][0], point[0], point[1]]
            
            # Set the joint positions
            for joint_id, position in zip(joint_ids, joint_config):
                data.qpos[joint_id] = position
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Render the scene
            done = renderer.render()
            if done:
                break
            
            # Slow down to see the motion
            time.sleep(0.1)
        
        print("Trajectory complete. Press Enter to continue...")
        input()
        
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")

if __name__ == "__main__":
    main() 