import numpy as np

def generate_s_pattern(reachability_data, num_passes=10):
    """
    Generate an S-pattern trajectory through the reachable workspace.
    
    Args:
        reachability_data: Dictionary containing reachability analysis results
        num_passes: Number of horizontal passes to make
        
    Returns:
        Dictionary with trajectory points and joint configurations
    """
    # Extract data from the reachability analysis
    y_points = reachability_data['y_points']
    z_points = reachability_data['z_points']
    
    # Get the full reachable workspace map
    flat_reachable = reachability_data['flat_reachable']
    perp_reachable = reachability_data['perp_reachable']
    
    # Get joint configurations
    flat_configs = reachability_data['flat_configs']
    perp_configs = reachability_data['perp_configs']
    
    # Create trajectories for both robots
    flat_trajectory = create_robot_trajectory(y_points, z_points, flat_reachable, flat_configs, num_passes)
    perp_trajectory = create_robot_trajectory(y_points, z_points, perp_reachable, perp_configs, num_passes)
    
    return {
        'flat_trajectory': flat_trajectory,
        'perp_trajectory': perp_trajectory
    }

def create_robot_trajectory(y_points, z_points, reachable_map, configs, num_passes):
    """
    Create an S-pattern trajectory for a single robot.
    
    Args:
        y_points: Array of y-coordinates
        z_points: Array of z-coordinates
        reachable_map: 2D boolean array indicating reachable points
        configs: Dictionary mapping (y,z) points to joint configurations
        num_passes: Number of horizontal passes to make
        
    Returns:
        List of (point, joint_config) for the trajectory
    """
    # Determine the range of reachable z values
    reachable_z_indices = np.where(np.any(reachable_map, axis=1))[0]
    
    if len(reachable_z_indices) == 0:
        # No reachable points
        return []
    
    min_z_idx = reachable_z_indices.min()
    max_z_idx = reachable_z_indices.max()
    
    # Calculate the height of each pass
    z_height = len(reachable_z_indices)
    pass_height = z_height / num_passes
    
    # Generate horizontal passes
    trajectory = []
    
    for i in range(num_passes):
        # Calculate the z-index for this pass
        z_idx = max_z_idx - int(i * pass_height)
        z_idx = min(z_idx, max_z_idx)  # Ensure we don't go beyond the reachable area
        
        # Get the z-coordinate
        z = z_points[z_idx]
        
        # Find reachable y-indices at this z-level
        reachable_y_indices = np.where(reachable_map[z_idx, :])[0]
        
        if len(reachable_y_indices) == 0:
            continue
        
        # Get min and max y-indices
        min_y_idx = reachable_y_indices.min()
        max_y_idx = reachable_y_indices.max()
        
        # Determine the direction of this pass (left to right or right to left)
        if i % 2 == 0:
            # Left to right
            y_indices = range(min_y_idx, max_y_idx + 1)
        else:
            # Right to left
            y_indices = range(max_y_idx, min_y_idx - 1, -1)
        
        # Add points to the trajectory
        for y_idx in y_indices:
            if reachable_map[z_idx, y_idx]:
                y = y_points[y_idx]
                point = (y, z)
                
                # Get the joint configuration for this point
                joint_config = configs.get(point)
                
                if joint_config is not None:
                    trajectory.append((point, joint_config))
    
    return trajectory 