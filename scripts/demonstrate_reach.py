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
from src.config.config_loader import load_config
from src.trajectory.s_pattern import generate_s_pattern
from src.visualization.render import MujocoRenderer
from src.trajectory.motion_planning import create_smooth_trajectory

def main():
    print("UR10e Reach Demonstration for Drywall Finishing")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
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
        saved_config = saved_data.get('config', {})
        # Merge the saved config with the current one, prioritizing saved values
        for key, value in saved_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # Merge nested dictionaries
                config[key].update(value)
            else:
                # Replace top-level keys
                config[key] = value
        
        # Print analysis info
        print(f"Using analysis data completed in {saved_data.get('analysis_time', 'N/A')} seconds")
    else:
        # Old format
        reachability_data = saved_data
        print("Using legacy format reachability data")
    
    # Generate S-pattern trajectories
    print("Generating S-pattern trajectories...")
    trajectory_data = generate_s_pattern(reachability_data, config=config)
    
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
    
    # Get motion planning parameters from config
    motion_planning_config = config.get('motion_planning', {})
    spline_density = motion_planning_config.get('spline_density', 5)
    velocity_limit = motion_planning_config.get('velocity_limit', 1.0)
    acceleration_limit = motion_planning_config.get('acceleration_limit', 2.0)
    check_collisions = motion_planning_config.get('check_collisions', True)
    
    # Convert trajectories to smooth trajectories
    print("Creating smooth trajectories with motion planning...")
    
    # Extract waypoints and joint configs from trajectories
    flat_waypoints = [point for point, _ in flat_trajectory]
    flat_joint_configs = [config for _, config in flat_trajectory]
    
    perp_waypoints = [point for point, _ in perp_trajectory]
    perp_joint_configs = [config for _, config in perp_trajectory]
    
    # Create smooth trajectories
    smooth_flat_trajectory = create_smooth_trajectory(
        flat_waypoints, flat_joint_configs, model, data, flat_joint_ids,
        spline_density=spline_density, velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit, check_collisions=check_collisions
    )
    
    smooth_perp_trajectory = create_smooth_trajectory(
        perp_waypoints, perp_joint_configs, model, data, perp_joint_ids,
        spline_density=spline_density, velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit, check_collisions=check_collisions
    )
    
    print(f"Generated {len(smooth_flat_trajectory)} points for smooth flat robot trajectory")
    print(f"Generated {len(smooth_perp_trajectory)} points for smooth perpendicular robot trajectory")
    
    # Ask user which robot to demonstrate
    while True:
        print("\nSelect a robot to demonstrate reachability:")
        print("1. Flat mounted robot")
        print("2. Perpendicular mounted robot")
        print("3. Both robots simultaneously")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ")
        
        if choice == '1':
            demonstrate_robot_smooth(model, data, renderer, smooth_flat_trajectory, flat_joint_ids, 
                             target_marker_id, "Flat Mounted Robot", config)
        elif choice == '2':
            demonstrate_robot_smooth(model, data, renderer, smooth_perp_trajectory, perp_joint_ids,
                             target_marker_id, "Perpendicular Mounted Robot", config)
        elif choice == '3':
            demonstrate_both_robots_smooth(model, data, renderer, smooth_flat_trajectory, smooth_perp_trajectory,
                                   flat_joint_ids, perp_joint_ids, target_marker_id, config)
        elif choice == '4':
            renderer.close()
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")
    
    print("Demonstration complete.")

def demonstrate_robot_smooth(model, data, renderer, trajectory, joint_ids, marker_id, robot_name, config):
    """
    Demonstrate a robot's reachability by following a smooth trajectory.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MujocoRenderer instance
        trajectory: List of (point, joint_config, time_to_next) from motion planning
        joint_ids: List of joint IDs
        marker_id: ID of the site to visualize the current target point
        robot_name: Name of the robot for display
        config: Configuration dictionary
    """
    if not trajectory:
        print(f"No reachable points for {robot_name}")
        return
    
    print(f"\nDemonstrating {robot_name} with {len(trajectory)} points in smooth trajectory")
    print("Press Ctrl+C to stop the demonstration")
    
    # Zero all joint positions first (to avoid weird visual artifacts)
    for i in range(model.nq):
        data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)
    
    # Get base speed from config
    base_speed = config.get('visualization', {}).get('trajectory_speed', 0.015)
    
    # Visualize the trajectory
    print("Press Enter to start following the smooth S-pattern trajectory...")
    input()
    
    try:
        for i, (point, joint_config, time_to_next) in enumerate(trajectory):
            # Update the progress
            if i % 50 == 0 or i == len(trajectory) - 1:  # Reduced frequency of progress updates
                print(f"Point {i+1}/{len(trajectory)}: y={point[0]:.2f}, z={point[1]:.2f}")
            
            # Place the target marker at the current point
            wall_x = config.get('wall', {}).get('x_position', 1.2)
            data.site_xpos[marker_id] = [wall_x, point[0], point[1]]
            
            # Set the joint positions
            for j, position in enumerate(joint_config):
                data.qpos[joint_ids[j]] = position
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Render the scene
            done = renderer.render()
            if done:
                break
            
            # Use a reasonable wait time (capped to avoid getting stuck and lower minimum)
            wait_time = min(time_to_next, 0.03) if time_to_next > 0.005 else base_speed
            time.sleep(wait_time)
        
        print("Trajectory complete. Press Enter to continue...")
        input()
        
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")

def demonstrate_both_robots_smooth(model, data, renderer, flat_trajectory, perp_trajectory, 
                          flat_joint_ids, perp_joint_ids, marker_id, config):
    """
    Demonstrate both robots simultaneously with smooth trajectories.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MujocoRenderer instance
        flat_trajectory: Smooth trajectory for flat robot
        perp_trajectory: Smooth trajectory for perpendicular robot
        flat_joint_ids: Joint IDs for flat robot
        perp_joint_ids: Joint IDs for perpendicular robot
        marker_id: ID of the site to visualize the current target point
        config: Configuration dictionary
    """
    if not flat_trajectory or not perp_trajectory:
        print("At least one robot has no trajectory")
        return
    
    print("\nDemonstrating both robots simultaneously with smooth trajectories")
    print("Press Ctrl+C to stop the demonstration")
    
    # Zero all joint positions first
    for i in range(model.nq):
        data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)
    
    # Get base speed from config
    base_speed = config.get('visualization', {}).get('trajectory_speed', 0.015)
    
    # Create interpolated trajectories with the same number of points
    max_points = max(len(flat_trajectory), len(perp_trajectory))
    flat_indices = np.linspace(0, len(flat_trajectory)-1, max_points, dtype=int)
    perp_indices = np.linspace(0, len(perp_trajectory)-1, max_points, dtype=int)
    
    # Calculate combined time intervals
    time_intervals = []
    for i in range(max_points):
        flat_idx = flat_indices[i]
        perp_idx = perp_indices[i]
        
        flat_time = flat_trajectory[flat_idx][2] if flat_idx < len(flat_trajectory) else 0.02
        perp_time = perp_trajectory[perp_idx][2] if perp_idx < len(perp_trajectory) else 0.02
        
        # Use the maximum time interval to ensure both robots move smoothly
        # Cap the time to prevent getting stuck (reduced maximum)
        time_intervals.append(min(max(flat_time, perp_time), 0.03))
    
    # Visualize the trajectory
    print("Press Enter to start following both smooth trajectories...")
    input()
    
    try:
        for i in range(max_points):
            # Update the progress
            if i % 50 == 0 or i == max_points - 1:  # Reduced frequency of progress updates
                print(f"Point {i+1}/{max_points}")
            
            # Get flat robot point and joint config
            flat_idx = flat_indices[i]
            if flat_idx < len(flat_trajectory):
                flat_point, flat_config, _ = flat_trajectory[flat_idx]
                # Set flat robot joint positions
                for j, position in enumerate(flat_config):
                    data.qpos[flat_joint_ids[j]] = position
                    
            # Get perp robot point and joint config
            perp_idx = perp_indices[i]
            if perp_idx < len(perp_trajectory):
                perp_point, perp_config, _ = perp_trajectory[perp_idx]
                # Set perp robot joint positions
                for j, position in enumerate(perp_config):
                    data.qpos[perp_joint_ids[j]] = position
            
            # Place the target marker at the flat point
            wall_x = config.get('wall', {}).get('x_position', 1.2)
            if flat_idx < len(flat_trajectory):
                data.site_xpos[marker_id] = [wall_x, flat_point[0], flat_point[1]]
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Render the scene
            done = renderer.render()
            if done:
                break
            
            # Use a reasonable wait time with a reduced minimum
            wait_time = time_intervals[i] if time_intervals[i] > 0.005 else base_speed
            time.sleep(wait_time)
        
        print("Trajectory complete. Press Enter to continue...")
        input()
        
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")

# Keep the original functions for backwards compatibility
def demonstrate_robot(model, data, renderer, trajectory, joint_ids, marker_id, robot_name, config):
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
        config: Configuration dictionary
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
    
    # Get trajectory speed from config
    trajectory_speed = config.get('visualization', {}).get('trajectory_speed', 0.1)
    
    # Visualize the trajectory
    print("Press Enter to start following the S-pattern trajectory...")
    input()
    
    try:
        for i, (point, joint_config) in enumerate(trajectory):
            # Update the progress
            if i % 20 == 0 or i == len(trajectory) - 1:
                print(f"Point {i+1}/{len(trajectory)}: y={point[0]:.2f}, z={point[1]:.2f}")
            
            # Place the target marker at the current point
            wall_x = config.get('wall', {}).get('x_position', 1.2)
            data.site_xpos[marker_id] = [wall_x, point[0], point[1]]
            
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
            time.sleep(trajectory_speed)
        
        print("Trajectory complete. Press Enter to continue...")
        input()
        
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")

def demonstrate_both_robots(model, data, renderer, flat_trajectory, perp_trajectory, 
                          flat_joint_ids, perp_joint_ids, marker_id, config):
    """
    Demonstrate both robots simultaneously.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MujocoRenderer instance
        flat_trajectory: Trajectory for flat robot
        perp_trajectory: Trajectory for perpendicular robot
        flat_joint_ids: Joint IDs for flat robot
        perp_joint_ids: Joint IDs for perpendicular robot
        marker_id: ID of the site to visualize the current target point
        config: Configuration dictionary
    """
    if not flat_trajectory or not perp_trajectory:
        print("At least one robot has no trajectory")
        return
    
    print("\nDemonstrating both robots simultaneously")
    print("Press Ctrl+C to stop the demonstration")
    
    # Zero all joint positions first
    for i in range(model.nq):
        data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)
    
    # Get trajectory speed from config
    trajectory_speed = config.get('visualization', {}).get('trajectory_speed', 0.1)
    
    # Create interpolated trajectories with the same number of points
    max_points = max(len(flat_trajectory), len(perp_trajectory))
    flat_indices = np.linspace(0, len(flat_trajectory)-1, max_points, dtype=int)
    perp_indices = np.linspace(0, len(perp_trajectory)-1, max_points, dtype=int)
    
    # Visualize the trajectory
    print("Press Enter to start following both trajectories...")
    input()
    
    try:
        for i in range(max_points):
            # Update the progress
            if i % 20 == 0 or i == max_points - 1:
                print(f"Point {i+1}/{max_points}")
            
            # Get flat robot point and joint config
            flat_point, flat_config = flat_trajectory[flat_indices[i]]
            # Set flat robot joint positions
            for joint_id, position in zip(flat_joint_ids, flat_config):
                data.qpos[joint_id] = position
                
            # Get perp robot point and joint config
            perp_point, perp_config = perp_trajectory[perp_indices[i]]
            # Set perp robot joint positions
            for joint_id, position in zip(perp_joint_ids, perp_config):
                data.qpos[joint_id] = position
            
            # Place the target marker at the midpoint between the two points
            wall_x = config.get('wall', {}).get('x_position', 1.2)
            data.site_xpos[marker_id] = [wall_x, flat_point[0], flat_point[1]]
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Render the scene
            done = renderer.render()
            if done:
                break
            
            # Slow down to see the motion
            time.sleep(trajectory_speed)
        
        print("Trajectory complete. Press Enter to continue...")
        input()
        
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")

if __name__ == "__main__":
    main() 