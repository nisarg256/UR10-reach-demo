#!/usr/bin/env python3
import numpy as np
import time

class PIDController:
    """
    PID Controller for smoothing robot joint motion.
    Implements a Proportional-Integral-Derivative controller for robot trajectory following.
    """
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, dt=0.01, integral_limits=(-1.0, 1.0)):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Time step between updates (seconds)
            integral_limits: Tuple of (min, max) limits for the integral term
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limits = integral_limits
        
        # State variables for each joint
        self.prev_error = None
        self.integral = None
        
    def reset(self, num_joints):
        """
        Reset the controller state.
        
        Args:
            num_joints: Number of joints to control
        """
        self.prev_error = np.zeros(num_joints)
        self.integral = np.zeros(num_joints)
        
    def update(self, target, current):
        """
        Update the PID controller.
        
        Args:
            target: Target joint positions (array of shape (num_joints,))
            current: Current joint positions (array of shape (num_joints,))
            
        Returns:
            Control output (desired joint velocities)
        """
        # Initialize controller state if not done yet
        if self.prev_error is None or len(self.prev_error) != len(target):
            self.reset(len(target))
            
        # Calculate error
        error = target - current
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        # Apply integral limits to prevent wind-up
        self.integral = np.clip(self.integral, self.integral_limits[0], self.integral_limits[1])
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / self.dt if self.prev_error is not None else 0
        self.prev_error = error.copy()
        
        # Calculate control output
        control = p_term + i_term + d_term
        
        return control
        
    def step_towards_target(self, current_positions, target_positions, velocity_limit, steps=10):
        """
        Generate a smoother trajectory towards the target using multiple PID controller updates.
        
        Args:
            current_positions: Current joint positions (array of shape (num_joints,))
            target_positions: Target joint positions (array of shape (num_joints,))
            velocity_limit: Maximum joint velocity
            steps: Number of interpolation steps
            
        Returns:
            Interpolated positions (smoothed trajectory)
        """
        positions = current_positions.copy()
        result = [positions.copy()]
        
        for _ in range(steps):
            # Get control signal from PID controller
            control = self.update(target_positions, positions)
            
            # Limit velocities
            control = np.clip(control, -velocity_limit, velocity_limit)
            
            # Update positions
            positions += control * self.dt
            
            # Store result
            result.append(positions.copy())
            
        return result 