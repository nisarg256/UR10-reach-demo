#!/usr/bin/env python3
import numpy as np
import time
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist

class RRTMotionPlanner:
    """
    RRT-based motion planner for generating collision-free trajectories between waypoints.
    """
    def __init__(self, model, data, joint_ids, joint_limits=None, max_iterations=1000):
        """
        Initialize the motion planner.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            joint_ids: List of joint IDs for the robot
            joint_limits: Optional dictionary with 'lower' and 'upper' arrays for joint limits
            max_iterations: Maximum iterations for RRT algorithm
        """
        self.model = model
        self.data = data
        self.joint_ids = joint_ids
        self.max_iterations = max_iterations
        
        # Set joint limits (if not provided, extract from model)
        if joint_limits is None:
            self.joint_limits = self._extract_joint_limits()
        else:
            self.joint_limits = joint_limits
            
        # RRT parameters
        self.step_size = 0.2        # Step size for extending tree
        self.goal_sample_rate = 0.2 # Probability of sampling the goal
        
        # Collision detection
        self.collision_check_resolution = 0.1  # Fraction of step size to check for collisions
        
    def _extract_joint_limits(self):
        """Extract joint limits from the MuJoCo model."""
        lower_limits = []
        upper_limits = []
        
        for joint_id in self.joint_ids:
            # Get joint range information from the model
            qpos_addr = self.model.jnt_qposadr[joint_id]
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            
            lower_limits.append(lower)
            upper_limits.append(upper)
            
        return {'lower': np.array(lower_limits), 'upper': np.array(upper_limits)}
    
    def _is_collision_free(self, joint_config):
        """Check if a joint configuration is collision-free."""
        # Store original joint positions
        original_qpos = self.data.qpos.copy()
        
        # Set the joint positions
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = joint_config[i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Check for collisions
        is_collision_free = not self._has_collision()
        
        # Restore original joint positions
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return is_collision_free
    
    def _has_collision(self):
        """Check if the robot is in collision with environment or itself."""
        # Quick check - if no contacts, return False
        if self.data.ncon == 0:
            return False
            
        # Only check a few contacts to improve performance
        max_contacts_to_check = min(10, self.data.ncon)
        
        for i in range(max_contacts_to_check):
            contact = self.data.contact[i]
            # Skip contacts that are part of the robot's structure
            # Only consider actual collisions (negative distance)
            if contact.dist < -0.001:  # Small threshold to account for numerical errors
                return True
        return False
    
    def _random_sample(self):
        """Generate a random joint configuration within joint limits."""
        return np.random.uniform(
            self.joint_limits['lower'],
            self.joint_limits['upper']
        )
    
    def _nearest_node(self, nodes, sample):
        """Find the nearest node in the tree to the sample."""
        distances = np.sum((nodes - sample[np.newaxis, :])**2, axis=1)
        return np.argmin(distances)
    
    def _steer(self, from_node, to_node):
        """Steer from one node toward another within step_size limit."""
        if np.allclose(from_node, to_node):
            return to_node
            
        vector = to_node - from_node
        length = np.linalg.norm(vector)
        
        if length > self.step_size:
            vector = vector / length * self.step_size
            
        return from_node + vector
    
    def _is_path_collision_free(self, from_node, to_node):
        """Check if the path between nodes is collision-free."""
        # Quick check of endpoints
        if not self._is_collision_free(from_node) or not self._is_collision_free(to_node):
            return False
            
        # Check a few points along the path
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
            
        direction = direction / distance
        
        # For short paths, just check endpoints
        if distance < self.step_size:
            return True
            
        # For longer paths, check a few intermediate points
        num_checks = max(3, int(distance / self.step_size))
        
        for i in range(1, num_checks):
            t = i / num_checks
            intermediate_node = from_node + t * distance * direction
            if not self._is_collision_free(intermediate_node):
                return False
                
        return True
    
    def plan_path(self, start_config, goal_config, timeout=2.0):
        """
        Plan a path from start to goal configuration using RRT.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            timeout: Maximum planning time in seconds
            
        Returns:
            List of joint configurations forming a path, or None if no path found
        """
        # First, check if direct path is possible
        if self._is_path_collision_free(start_config, goal_config):
            return [start_config, goal_config]
            
        # Initialize the RRT
        nodes = [start_config]
        parent_indices = [0]  # Parent index for each node
        
        start_time = time.time()
        
        # Main RRT loop
        for i in range(self.max_iterations):
            # Check timeout
            if time.time() - start_time > timeout:
                print("  RRT planning timeout reached")
                return None
                
            # Sample a random point or goal
            if np.random.random() < self.goal_sample_rate:
                sample = goal_config
            else:
                sample = self._random_sample()
                
            # Find nearest node
            nearest_idx = self._nearest_node(np.array(nodes), sample)
            nearest_node = nodes[nearest_idx]
            
            # Steer toward sample
            new_node = self._steer(nearest_node, sample)
            
            # Skip if too similar to nearest node
            if np.allclose(new_node, nearest_node):
                continue
                
            # Check if path is collision-free
            if self._is_path_collision_free(nearest_node, new_node):
                # Add node to the tree
                nodes.append(new_node)
                parent_indices.append(nearest_idx)
                
                # Check if goal is reached
                if np.linalg.norm(new_node - goal_config) < self.step_size:
                    # Check direct connection to goal
                    if self._is_path_collision_free(new_node, goal_config):
                        # Add goal to the tree
                        nodes.append(goal_config)
                        parent_indices.append(len(nodes) - 2)
                        
                        # Extract path
                        path = self._extract_path(nodes, parent_indices)
                        return path
        
        print(f"  RRT planning failed: maximum iterations ({self.max_iterations}) reached")
        return None
    
    def _extract_path(self, nodes, parent_indices):
        """Extract path from RRT tree."""
        path = [nodes[-1]]  # Start with the goal
        current_idx = len(nodes) - 1
        
        while current_idx != 0:
            current_idx = parent_indices[current_idx]
            path.append(nodes[current_idx])
            
        path.reverse()  # Reverse to get path from start to goal
        return path

def plan_smooth_trajectory(waypoints, joint_configs, model, data, joint_ids, num_points=100, check_collisions=True):
    """
    Plan a smooth trajectory through a series of waypoints.
    
    Args:
        waypoints: List of (y, z) waypoints on the wall
        joint_configs: List of joint configurations for each waypoint
        model: MuJoCo model
        data: MuJoCo data
        joint_ids: List of joint IDs
        num_points: Number of points in the interpolated trajectory
        check_collisions: Whether to check for collisions during planning
        
    Returns:
        List of (point, joint_config) pairs for the smooth trajectory
    """
    if len(waypoints) < 2:
        # Not enough waypoints
        return [(waypoints[0], joint_configs[0])] if waypoints else []
    
    # Create motion planner for collision checking
    planner = RRTMotionPlanner(model, data, joint_ids) if check_collisions else None
    
    # Initialize smooth trajectory
    smooth_trajectory = []
    
    # Process consecutive waypoint pairs
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]
        start_config = joint_configs[i]
        end_config = joint_configs[i+1]
        
        # Check if we need to plan around obstacles
        if check_collisions and not planner._is_path_collision_free(start_config, end_config):
            # Plan a path using RRT
            path = planner.plan_path(start_config, end_config)
            
            if path is None:
                # If planning fails, just use linear interpolation
                print(f"Warning: Path planning failed between points {i} and {i+1}, using linear interpolation instead")
                path = [start_config, end_config]
        else:
            # No collision or collision checking disabled, use direct path
            path = [start_config, end_config]
        
        # Number of points for this segment
        segment_points = max(2, int(num_points * (i+1) / len(waypoints)))
        
        # Interpolate joint configurations
        alphas = np.linspace(0, 1, segment_points)
        
        # Create cubic spline for joint trajectories if we have enough points
        if len(path) >= 4:
            # Parameterize path by distance
            distances = [0]
            for j in range(1, len(path)):
                distances.append(distances[-1] + np.linalg.norm(path[j] - path[j-1]))
            
            # Normalize distances
            distances = np.array(distances) / distances[-1]
            
            # Create cubic splines for each joint
            splines = []
            for j in range(len(path[0])):
                joint_values = [config[j] for config in path]
                spline = CubicSpline(distances, joint_values)
                splines.append(spline)
            
            # Interpolate using splines
            for alpha in alphas:
                # Interpolate joint values
                joint_values = [spline(alpha) for spline in splines]
                
                # Interpolate position (linear interpolation for wall points)
                y = (1 - alpha) * start_point[0] + alpha * end_point[0]
                z = (1 - alpha) * start_point[1] + alpha * end_point[1]
                
                smooth_trajectory.append(((y, z), joint_values))
        else:
            # Linear interpolation for shorter paths
            for alpha in alphas:
                # Interpolate joint values
                joint_values = (1 - alpha) * path[0] + alpha * path[-1]
                
                # Interpolate position
                y = (1 - alpha) * start_point[0] + alpha * end_point[0]
                z = (1 - alpha) * start_point[1] + alpha * end_point[1]
                
                smooth_trajectory.append(((y, z), joint_values))
    
    return smooth_trajectory

def time_parameterization(trajectory, joint_velocities, max_velocity=1.0, max_acceleration=2.0):
    """
    Apply time parameterization to ensure smooth motion with velocity and acceleration limits.
    
    Args:
        trajectory: List of (point, joint_config) pairs
        joint_velocities: List of joint velocities between points
        max_velocity: Maximum joint velocity
        max_acceleration: Maximum joint acceleration
        
    Returns:
        List of time intervals between points
    """
    if len(trajectory) <= 1:
        return [0.0]
    
    # Initialize time intervals with a small default value
    time_intervals = [0.03] * len(trajectory)  # Use a consistent default time
    
    # Scale time intervals based on joint velocities
    for i, velocity in enumerate(joint_velocities):
        # Get maximum joint velocity
        max_joint_vel = np.max(np.abs(velocity))
        
        # Scale time interval based on velocity
        if max_joint_vel > 1e-6:
            # Minimum time needed to respect velocity limits
            min_time = max_joint_vel / max_velocity
            # Use a consistent scaling for smooth motion
            time_intervals[i] = min(max(0.02, min_time * 0.1), 0.06)  # Higher minimum, lower maximum
    
    # Smooth out time intervals to avoid sudden changes
    smoothed_intervals = smooth_time_intervals(time_intervals, max_acceleration)
    
    return smoothed_intervals

def smooth_time_intervals(time_intervals, max_acceleration):
    """
    Smooth out time intervals to ensure acceleration limits are respected.
    
    Args:
        time_intervals: Initial time intervals
        max_acceleration: Maximum acceleration
        
    Returns:
        Smoothed time intervals
    """
    if len(time_intervals) <= 2:
        return time_intervals
    
    # Create a copy to avoid modifying the original
    smoothed = time_intervals.copy()
    
    # Apply multi-pass smoothing for more consistent timing
    for _ in range(3):  # Multiple smoothing passes
        # Simple smoothing using a sliding window
        smoothed_copy = smoothed.copy()
        for i in range(1, len(time_intervals)-1):
            # Average with neighbors with weight
            smoothed[i] = (smoothed_copy[i-1] + 2*smoothed_copy[i] + smoothed_copy[i+1]) / 4
    
    # Apply time constraints - keeping times consistent for more controlled motion
    for i in range(len(smoothed)):
        # More consistent timing range
        smoothed[i] = min(max(0.02, smoothed[i]), 0.06)
    
    return smoothed

def create_smooth_trajectory(waypoints, joint_configs, model, data, joint_ids, 
                            spline_density=5, velocity_limit=1.0, 
                            acceleration_limit=2.0, check_collisions=False,
                            rrt_timeout=2.0, max_iterations=1000):
    """
    Create a smooth trajectory through waypoints with time parameterization.
    
    Args:
        waypoints: List of (y, z) waypoints
        joint_configs: List of joint configurations
        model: MuJoCo model
        data: MuJoCo data
        joint_ids: List of joint IDs
        spline_density: Points per segment (higher = smoother but slower)
        velocity_limit: Maximum joint velocity
        acceleration_limit: Maximum joint acceleration
        check_collisions: Whether to check for collisions
        rrt_timeout: Timeout for RRT planning in seconds
        max_iterations: Maximum iterations for RRT planning
        
    Returns:
        Time-parameterized smooth trajectory
    """
    if len(waypoints) < 2:
        # Not enough waypoints for a trajectory
        if waypoints:
            return [(waypoints[0], joint_configs[0], 0.1)]
        return []
    
    # Number of points in the interpolated trajectory
    num_segments = len(waypoints) - 1
    num_points = num_segments * spline_density
    
    # Create smooth waypoint paths with collision checking if enabled
    if check_collisions:
        print("Performing collision-aware trajectory planning...")
        smooth_path = create_collision_free_path(
            waypoints, joint_configs, model, data, joint_ids, 
            num_points, rrt_timeout, max_iterations
        )
    else:
        # Proceed with standard spline interpolation without collision checking
        smooth_path = create_spline_path(waypoints, joint_configs, num_points)
    
    # Calculate joint velocities for time parameterization
    joint_velocities = calculate_joint_velocities(smooth_path)
    
    # Apply time parameterization
    time_intervals = time_parameterization(
        smooth_path, joint_velocities, 
        velocity_limit, acceleration_limit
    )
    
    # Create final trajectory with timing information
    timed_trajectory = []
    for i, (point, config) in enumerate(smooth_path):
        time_to_next = time_intervals[i] if i < len(time_intervals) else 0.0
        timed_trajectory.append((point, config, time_to_next))
    
    return timed_trajectory

def create_spline_path(waypoints, joint_configs, num_points):
    """
    Create a smooth path using cubic spline interpolation without collision checking.
    
    Args:
        waypoints: List of (y, z) waypoints
        joint_configs: List of joint configurations
        num_points: Number of points to generate
        
    Returns:
        List of (point, joint_config) pairs
    """
    # Create parameter array (0 to 1) for the whole path
    t_eval = np.linspace(0, 1, num_points)
    
    # Create parameter values for the waypoints
    t_waypoints = np.linspace(0, 1, len(waypoints))
    
    # Prepare arrays for interpolation
    waypoint_ys = np.array([p[0] for p in waypoints])
    waypoint_zs = np.array([p[1] for p in waypoints])
    
    # Convert joint configurations to numpy array
    joint_array = np.array(joint_configs)
    num_joints = joint_array.shape[1]
    
    # Create splines for y, z coordinates and each joint
    y_spline = CubicSpline(t_waypoints, waypoint_ys)
    z_spline = CubicSpline(t_waypoints, waypoint_zs)
    
    # Create splines for each joint
    joint_splines = []
    for j in range(num_joints):
        joint_values = joint_array[:, j]
        joint_splines.append(CubicSpline(t_waypoints, joint_values))
    
    # Generate smooth trajectory points
    smooth_trajectory = []
    
    # Evaluate splines at each interpolation point
    for i, t in enumerate(t_eval):
        # Interpolate wall position
        y = y_spline(t)
        z = z_spline(t)
        
        # Interpolate joint values
        joint_values = np.array([spline(t) for spline in joint_splines])
        
        # Store the position and configuration
        smooth_trajectory.append(((y, z), joint_values))
    
    return smooth_trajectory

def create_collision_free_path(waypoints, joint_configs, model, data, joint_ids, 
                              num_points, rrt_timeout=2.0, max_iterations=1000):
    """
    Create a collision-free path through waypoints using RRT planning and cubic spline interpolation.
    
    Args:
        waypoints: List of (y, z) waypoints
        joint_configs: List of joint configurations
        model: MuJoCo model
        data: MuJoCo data
        joint_ids: List of joint IDs
        num_points: Number of points in final trajectory
        rrt_timeout: Timeout for RRT planning
        max_iterations: Maximum iterations for RRT
        
    Returns:
        List of (point, joint_config) pairs
    """
    # Initialize RRT planner
    planner = RRTMotionPlanner(model, data, joint_ids, max_iterations=max_iterations)
    
    # Initialize path segments
    path_segments = []
    
    # Store any paths that failed RRT planning
    failed_segments = []
    
    # Process consecutive waypoint pairs
    print(f"Planning collision-free path with {len(waypoints)-1} segments...")
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i+1]
        start_config = joint_configs[i]
        end_config = joint_configs[i+1]
        
        # Check if direct path is collision-free
        if planner._is_path_collision_free(start_config, end_config):
            # No collision, use direct path
            path_segments.append((start_point, end_point, [start_config, end_config]))
        else:
            # Plan a path using RRT
            path = planner.plan_path(start_config, end_config, timeout=rrt_timeout)
            
            if path is not None:
                path_segments.append((start_point, end_point, path))
            else:
                # If planning fails, just use linear interpolation and mark as failed
                failed_segments.append(i)
                path_segments.append((start_point, end_point, [start_config, end_config]))
                print(f"  Path planning failed for segment {i} - using direct path")
    
    # Calculate points per segment based on segment length
    total_length = 0
    segment_lengths = []
    for start_point, end_point, _ in path_segments:
        length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        segment_lengths.append(length)
        total_length += length
    
    # Allocate points proportionally to segment length
    points_per_segment = []
    remaining_points = num_points
    for i, length in enumerate(segment_lengths):
        # Allocate at least 2 points per segment
        if i < len(segment_lengths) - 1:
            segment_points = max(2, int(length / total_length * num_points))
            remaining_points -= segment_points
            points_per_segment.append(segment_points)
        else:
            # Last segment gets remaining points
            points_per_segment.append(max(2, remaining_points))
    
    # Generate smooth path
    smooth_trajectory = []
    
    for i, ((start_point, end_point, path), num_segment_points) in enumerate(zip(path_segments, points_per_segment)):
        # Parameter for interpolation
        alphas = np.linspace(0, 1, num_segment_points)
        
        # Interpolate position
        ys = np.linspace(start_point[0], end_point[0], num_segment_points)
        zs = np.linspace(start_point[1], end_point[1], num_segment_points)
        
        # If we have enough points for cubic spline interpolation
        if len(path) >= 4:
            # Parameterize the path
            distances = [0]
            for j in range(1, len(path)):
                distances.append(distances[-1] + np.linalg.norm(path[j] - path[j-1]))
            
            if distances[-1] > 0:  # Ensure the path has non-zero length
                # Normalize distances
                distances = np.array(distances) / distances[-1]
                
                # Create cubic splines for each joint
                splines = []
                for j in range(len(path[0])):
                    joint_values = [config[j] for config in path]
                    spline = CubicSpline(distances, joint_values)
                    splines.append(spline)
                
                # Interpolate using splines
                for alpha, y, z in zip(alphas, ys, zs):
                    # Get interpolated joint values
                    joint_values = np.array([spline(alpha) for spline in splines])
                    smooth_trajectory.append(((y, z), joint_values))
            else:
                # Path has zero length, use direct interpolation
                for alpha, y, z in zip(alphas, ys, zs):
                    joint_values = (1 - alpha) * path[0] + alpha * path[-1]
                    smooth_trajectory.append(((y, z), joint_values))
        else:
            # Linear interpolation for short paths
            for alpha, y, z in zip(alphas, ys, zs):
                joint_values = (1 - alpha) * path[0] + alpha * path[-1]
                smooth_trajectory.append(((y, z), joint_values))
    
    if failed_segments:
        print(f"Completed path planning with {len(failed_segments)} failed segments (using direct paths instead)")
    else:
        print("Successfully planned collision-free path for all segments")
        
    return smooth_trajectory

def calculate_joint_velocities(trajectory):
    """
    Calculate joint velocities between each pair of points in the trajectory.
    
    Args:
        trajectory: List of (point, joint_config) pairs
        
    Returns:
        List of joint velocity vectors
    """
    if len(trajectory) <= 1:
        return []
    
    joint_configs = [config for _, config in trajectory]
    velocities = []
    
    for i in range(len(joint_configs) - 1):
        velocity = joint_configs[i+1] - joint_configs[i]
        velocities.append(velocity)
    
    return velocities 