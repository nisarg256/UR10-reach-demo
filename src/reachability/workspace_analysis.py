import numpy as np
import mujoco
from scipy.optimize import minimize

class ReachabilityAnalyzer:
    def __init__(self, model_path, flat_prefix="flat_", perp_prefix="perp_"):
        """
        Initialize the reachability analyzer for UR10e robots.
        
        Args:
            model_path: Path to the MuJoCo XML model file
            flat_prefix: Prefix for the flat-mounted robot's joint names
            perp_prefix: Prefix for the perpendicularly-mounted robot's joint names
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Store the prefixes for the two robot configurations
        self.flat_prefix = flat_prefix
        self.perp_prefix = perp_prefix
        
        # Get joint IDs for both robots
        self.flat_joint_ids = self._get_joint_ids(flat_prefix)
        self.perp_joint_ids = self._get_joint_ids(perp_prefix)
        
        # Get site IDs for tool tips
        self.flat_tool_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flat_tool_tip")
        self.perp_tool_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "perp_tool_tip")
        
        # Wall position (from the XML, the wall is at x=1.2)
        self.wall_x = 1.2
        
        # Set initial configurations to help IK solver
        # A more appropriate starting configuration for flat robot
        self.flat_initial_config = np.array([0.0, -1.0, 1.5, -0.5, 1.57, 0.0])
        self.perp_initial_config = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])
        
    def _get_joint_ids(self, prefix):
        """Get joint IDs for a robot configuration."""
        joint_names = [
            f"{prefix}shoulder_pan_joint",
            f"{prefix}shoulder_lift_joint",
            f"{prefix}elbow_joint",
            f"{prefix}wrist_1_joint",
            f"{prefix}wrist_2_joint",
            f"{prefix}wrist_3_joint"
        ]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    
    def _set_joint_positions(self, joint_ids, positions):
        """Set joint positions for a robot."""
        for joint_id, position in zip(joint_ids, positions):
            self.data.qpos[joint_id] = position
        mujoco.mj_forward(self.model, self.data)
    
    def _get_tool_tip_position(self, tool_tip_id):
        """Get the position of a tool tip."""
        return self.data.site_xpos[tool_tip_id].copy()
    
    def _get_tool_orientation(self, tool_tip_id):
        """Get the orientation matrix of a tool."""
        site_id = tool_tip_id
        # Get site orientation (3x3 rotation matrix)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id].reshape(3, 3))
        return quat
    
    def _check_perpendicularity(self, tool_tip_id):
        """
        Check if the tool is perpendicular to the wall.
        
        In our model, the tool is a cylinder whose principal axis should be aligned with the X-axis
        to be perpendicular to the YZ wall. In the MuJoCo model, the cylinder's local axis is oriented 
        along the Y-axis due to the quaternion rotation in the XML.
        """
        site_id = tool_tip_id
        # Get the orientation matrix (3x3) of the tool
        rot_mat = self.data.site_xmat[site_id].reshape(3, 3)
        
        # Extract the Y-axis of the tool orientation (second column of rotation matrix)
        # This represents the principal axis of the cylinder in our model
        tool_principal_axis = rot_mat[:, 1]
        
        # For perpendicularity to the wall, this axis should point in the positive X direction (1,0,0)
        target_axis = np.array([1.0, 0.0, 0.0])
        
        # Calculate the angle between the tool's principal axis and the target axis
        dot_product = np.dot(tool_principal_axis, target_axis)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Return True if the angle is close to 0 (perpendicular to wall)
        return angle < 0.35  # About 20 degrees
    
    def _ik_objective(self, q, joint_ids, tool_tip_id, target_pos, robot_type):
        """Objective function for inverse kinematics."""
        self._set_joint_positions(joint_ids, q)
        current_pos = self._get_tool_tip_position(tool_tip_id)
        
        # Calculate position error
        pos_error = np.linalg.norm(current_pos - target_pos)
        
        # Add penalty for not being perpendicular to the wall
        perp_penalty = 0.0
        if not self._check_perpendicularity(tool_tip_id):
            # Stronger penalty to ensure perpendicularity
            perp_penalty = 5.0 if robot_type == "flat" else 5.0
        
        # Add penalty for not being at the wall
        wall_dist = abs(current_pos[0] - self.wall_x)
        wall_penalty = 2.0 * wall_dist
        
        return pos_error + perp_penalty + wall_penalty
    
    def is_point_reachable(self, point, robot_type="flat", max_attempts=15):  # Increased attempts
        """
        Check if a point is reachable by a robot while maintaining perpendicularity to the wall.
        
        Args:
            point: (y, z) coordinates on the wall
            robot_type: 'flat' or 'perp'
            max_attempts: Maximum number of IK attempts with different initial configurations
            
        Returns:
            (bool, np.array): Tuple of (is_reachable, joint_angles)
        """
        if robot_type == "flat":
            joint_ids = self.flat_joint_ids
            tool_tip_id = self.flat_tool_tip_id
            init_config = self.flat_initial_config
        else:
            joint_ids = self.perp_joint_ids
            tool_tip_id = self.perp_tool_tip_id
            init_config = self.perp_initial_config
        
        # The target position on the wall (x is fixed at wall position)
        target_pos = np.array([self.wall_x, point[0], point[1]])
        
        # Try multiple initial joint configurations
        best_result = None
        best_error = float('inf')
        
        # Start with a good initial configuration (use previous solution if available)
        attempts = []
        
        # First try the standard starting config
        attempts.append(init_config.copy())
        
        # Add some jittered versions of the starting config
        for _ in range(max_attempts - 1):
            # More exploration for flat robot
            jitter_range = 0.8 if robot_type == "flat" else 0.5
            jittered = init_config + np.random.uniform(-jitter_range, jitter_range, len(joint_ids))
            attempts.append(jittered)
        
        for initial_q in attempts:
            # Run inverse kinematics
            result = minimize(
                lambda q: self._ik_objective(q, joint_ids, tool_tip_id, target_pos, robot_type),
                initial_q,
                method='L-BFGS-B',
                bounds=[(-6.28, 6.28)] * len(joint_ids),
                options={'ftol': 1e-6, 'maxiter': 300}  # Increased iterations and precision
            )
            
            # Check if this solution is better
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
                
                # If we found a good solution, stop early
                if best_error < 0.05:  # Good solution threshold
                    break
        
        # Check if the best solution is good enough
        is_reachable = best_error < 0.15
        
        if is_reachable:
            # Set the joints to the solution and check if it's actually perpendicular
            self._set_joint_positions(joint_ids, best_result.x)
            pos = self._get_tool_tip_position(tool_tip_id)
            
            # Double check that the tool is at the wall and perpendicular to it
            at_wall = abs(pos[0] - self.wall_x) < 0.05
            perpendicular = self._check_perpendicularity(tool_tip_id)
            
            is_reachable = at_wall and perpendicular
            
            # Update the initial configuration for this robot for next time
            if robot_type == "flat":
                self.flat_initial_config = best_result.x
            else:
                self.perp_initial_config = best_result.x
        
        return is_reachable, best_result.x if is_reachable else None
    
    def analyze_wall_reachability(self, y_range=(-0.8, 0.8), z_range=(0.5, 2.0), resolution=0.1):
        """
        Analyze the reachability of points on the wall for both robot configurations.
        
        Args:
            y_range: (min_y, max_y) range to check (default: adjusted to more likely reachable area)
            z_range: (min_z, max_z) range to check (default: adjusted to more likely reachable area)
            resolution: Distance between points to check
            
        Returns:
            (flat_reachable, perp_reachable): Two 2D arrays of boolean values indicating reachability
        """
        # Create grid of points to check
        y_points = np.arange(y_range[0], y_range[1] + resolution, resolution)
        z_points = np.arange(z_range[0], z_range[1] + resolution, resolution)
        
        # Initialize reachability maps
        flat_reachable = np.zeros((len(z_points), len(y_points)), dtype=bool)
        perp_reachable = np.zeros((len(z_points), len(y_points)), dtype=bool)
        
        # Store joint configurations for reachable points
        flat_configs = {}
        perp_configs = {}
        
        # Check each point
        print("Checking reachability of grid points:")
        print(f"y range: {y_range}, z range: {z_range}, resolution: {resolution}")
        print(f"Total points: {len(y_points) * len(z_points)}")
        
        # Start with center points and work outward for better IK seeding
        center_y_idx = len(y_points) // 2
        center_z_idx = len(z_points) // 2
        
        # Process in a spiral pattern from center
        max_radius = max(len(y_points), len(z_points))
        
        for r in range(max_radius):
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    # Only process points on the perimeter of this radius
                    if abs(i) != r and abs(j) != r:
                        continue
                    
                    # Calculate actual indices
                    z_idx = center_z_idx + i
                    y_idx = center_y_idx + j
                    
                    # Skip if out of bounds
                    if z_idx < 0 or z_idx >= len(z_points) or y_idx < 0 or y_idx >= len(y_points):
                        continue
                    
                    # Get the actual coordinates
                    z = z_points[z_idx]
                    y = y_points[y_idx]
                    point = (y, z)
                    
                    # Check flat robot
                    flat_reach, flat_joints = self.is_point_reachable(point, "flat")
                    flat_reachable[z_idx, y_idx] = flat_reach
                    if flat_reach:
                        flat_configs[(y, z)] = flat_joints
                    
                    # Check perpendicular robot
                    perp_reach, perp_joints = self.is_point_reachable(point, "perp")
                    perp_reachable[z_idx, y_idx] = perp_reach
                    if perp_reach:
                        perp_configs[(y, z)] = perp_joints
        
        # Return the reachability maps and joint configurations
        return {
            'y_points': y_points,
            'z_points': z_points,
            'flat_reachable': flat_reachable,
            'perp_reachable': perp_reachable,
            'flat_configs': flat_configs,
            'perp_configs': perp_configs
        } 