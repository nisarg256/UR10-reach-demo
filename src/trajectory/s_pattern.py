import numpy as np

def generate_s_pattern(reachability_data, num_passes=None, point_spacing=None, config=None):
    """
    Generate an S-pattern trajectory through the reachable workspace.
    
    Args:
        reachability_data: Dictionary containing reachability analysis results
        num_passes: Number of horizontal passes to make (default: from config)
        point_spacing: Spacing between points in trajectory (default: from config)
        config: Configuration dictionary
        
    Returns:
        Dictionary with trajectory points and joint configurations
    """
    # Get configuration parameters
    if config is None:
        config = {}
    
    if num_passes is None:
        num_passes = config.get('trajectory', {}).get('num_passes', 10)
        
    if point_spacing is None:
        point_spacing = config.get('trajectory', {}).get('point_spacing', 0.05)
    
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
    flat_trajectory = create_robot_trajectory(y_points, z_points, flat_reachable, flat_configs, num_passes, point_spacing)
    perp_trajectory = create_robot_trajectory(y_points, z_points, perp_reachable, perp_configs, num_passes, point_spacing)
    
    return {
        'flat_trajectory': flat_trajectory,
        'perp_trajectory': perp_trajectory
    }

def create_robot_trajectory(y_points, z_points, reachable_map, configs, num_passes, point_spacing):
    """
    Create an S-pattern trajectory for a single robot.
    
    Args:
        y_points: Array of y-coordinates
        z_points: Array of z-coordinates
        reachable_map: 2D boolean array indicating reachable points
        configs: Dictionary mapping (y,z) points to joint configurations
        num_passes: Number of horizontal passes to make
        point_spacing: Spacing between points in trajectory (meters)
        
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
            y_start = y_points[min_y_idx]
            y_end = y_points[max_y_idx]
            y_dir = 1
        else:
            # Right to left
            y_indices = range(max_y_idx, min_y_idx - 1, -1)
            y_start = y_points[max_y_idx]
            y_end = y_points[min_y_idx]
            y_dir = -1
        
        # Create denser y-points for smoother trajectory
        y_length = abs(y_end - y_start)
        num_dense_points = max(int(y_length / point_spacing), 1)
        dense_y_points = np.linspace(y_start, y_end, num_dense_points)
        
        # Add points to the trajectory with interpolated joint configurations
        prev_joint_config = None
        
        for y in dense_y_points:
            # Find the nearest grid points for interpolation
            y_idx = np.argmin(np.abs(y_points - y))
            
            # Check if this point is reachable
            if reachable_map[z_idx, y_idx]:
                # Get the closest grid point
                grid_y = y_points[y_idx]
                grid_point = (grid_y, z)
                
                # Get the joint configuration for this point
                joint_config = configs.get(grid_point)
                
                if joint_config is not None:
                    # Add the interpolated point to the trajectory
                    trajectory.append(((y, z), joint_config))
                    prev_joint_config = joint_config
                elif prev_joint_config is not None:
                    # If we can't find a configuration but have a previous one, use linear interpolation
                    # (This keeps the trajectory continuous)
                    trajectory.append(((y, z), prev_joint_config))
    
    return trajectory 