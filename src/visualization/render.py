import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import glfw

class MujocoRenderer:
    def __init__(self, model, data, width=1280, height=720):
        """
        Initialize the MuJoCo renderer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Width of the rendered image
            height: Height of the rendered image
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        # Create window
        glfw.window_hint(glfw.VISIBLE, 1)
        glfw.window_hint(glfw.RESIZABLE, 0)
        self.window = glfw.create_window(width, height, "UR10e Reach Demonstration", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        
        # Make the window's context current
        glfw.make_context_current(self.window)
        
        # Initialize MuJoCo visualization objects
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # Set camera default configuration
        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0
        self.camera.distance = 4.0
        self.camera.lookat[0] = 0.7  # focus on midpoint between robots and wall
        self.camera.lookat[1] = 0.0
        self.camera.lookat[2] = 1.2
        
        self.options = mujoco.MjvOption()
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Set the buffer size when window size changes
        glfw.set_framebuffer_size_callback(
            self.window, self._framebuffer_size_callback
        )
        
        # Mouse state
        self.mouse_button_left = False
        self.mouse_button_middle = False
        self.mouse_button_right = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Register mouse callbacks
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback
        )
        glfw.set_cursor_pos_callback(
            self.window, self._mouse_move_callback
        )
        glfw.set_scroll_callback(
            self.window, self._scroll_callback
        )
        
        # Register key callback
        glfw.set_key_callback(
            self.window, self._key_callback
        )
        
        # Display control instructions
        self._show_control_help()
    
    def _show_control_help(self):
        """Display camera control instructions in console"""
        print("\nCamera Control Instructions:")
        print("----------------------------")
        print("Left mouse button + drag: Rotate camera")
        print("Right mouse button + drag: Pan camera")
        print("Mouse wheel: Zoom in/out")
        print("R key: Reset camera view")
        print("----------------------------\n")
    
    def _framebuffer_size_callback(self, window, width, height):
        """Handle window resizing."""
        self.width = width
        self.height = height
        
        # Update context
        mujoco.mjr_freeContext(self.context)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    def _mouse_button_callback(self, window, button, act, mods):
        """Handle mouse button events."""
        # Update button states
        self.mouse_button_left = (button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS)
        self.mouse_button_middle = (button == glfw.MOUSE_BUTTON_MIDDLE and act == glfw.PRESS)
        self.mouse_button_right = (button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS)
        
        # Update mouse position
        x, y = glfw.get_cursor_pos(window)
        self.last_mouse_x = x
        self.last_mouse_y = y
    
    def _mouse_move_callback(self, window, x, y):
        """Handle mouse movement."""
        # Compute displacement
        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        
        # Update mouse position
        self.last_mouse_x = x
        self.last_mouse_y = y
        
        # Determine action based on mouse button
        if self.mouse_button_left:
            # Rotation: Adjust azimuth and elevation
            self.camera.azimuth -= 0.5 * dx
            # Keep azimuth in [0, 360]
            self.camera.azimuth = self.camera.azimuth % 360.0
            
            # INVERTED: negative dy for upward mouse movement will increase elevation
            self.camera.elevation -= 0.5 * dy
            # Limit elevation to [-90, 90]
            self.camera.elevation = np.clip(self.camera.elevation, -90.0, 90.0)
            
        elif self.mouse_button_right:
            # Pan: Adjust lookat position
            forward = np.array([np.cos(np.deg2rad(self.camera.azimuth)) * np.cos(np.deg2rad(self.camera.elevation)),
                             np.sin(np.deg2rad(self.camera.azimuth)) * np.cos(np.deg2rad(self.camera.elevation)),
                             np.sin(np.deg2rad(self.camera.elevation))])
            
            # Get camera right and up vectors
            right = np.cross(forward, np.array([0, 0, 1]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Scale movements by distance for more natural panning
            scale_factor = 0.0005 * self.camera.distance
            
            # Update lookat position
            self.camera.lookat[0] -= scale_factor * (dx * right[0] + dy * up[0])
            self.camera.lookat[1] -= scale_factor * (dx * right[1] + dy * up[1])
            self.camera.lookat[2] -= scale_factor * (dx * right[2] + dy * up[2])
    
    def _scroll_callback(self, window, x_offset, y_offset):
        """Handle mouse scroll for zooming."""
        # Adjust camera distance
        self.camera.distance -= 0.2 * y_offset
        # Ensure distance is positive
        self.camera.distance = max(0.1, self.camera.distance)
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if key == glfw.KEY_R and action == glfw.PRESS:
            # Reset camera view
            self.camera.azimuth = 90.0
            self.camera.elevation = -20.0
            self.camera.distance = 4.0
            self.camera.lookat[0] = 0.7
            self.camera.lookat[1] = 0.0
            self.camera.lookat[2] = 1.2
    
    def render(self):
        """Render the scene to the window."""
        # Check if window should close
        if glfw.window_should_close(self.window):
            glfw.terminate()
            return True
        
        # Clear buffers and update scene
        glfw.poll_events()
        mujoco.mjv_updateScene(
            self.model, self.data, self.options, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        
        # Render scene
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # Add on-screen controls help text
        self._add_onscreen_help()
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        
        return False  # Continue rendering
    
    def _add_onscreen_help(self):
        """Add on-screen help text for controls."""
        text = [
            "Left mouse: Rotate",
            "Right mouse: Pan",
            "Scroll: Zoom",
            "R key: Reset view"
        ]
        
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        
        # Set position and overlay
        for i, line in enumerate(text):
            x_pos = 10
            y_pos = self.height - 20 * (i + 1)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                line,
                "",
                self.context
            )
    
    def close(self):
        """Close the renderer and release resources."""
        glfw.terminate()

def visualize_reachability(reachability_data, wall_x=1.2, title=None):
    """
    Visualize the reachability map for both robot configurations.
    
    Args:
        reachability_data: Dictionary containing reachability analysis results
        wall_x: X-coordinate of the wall
        title: Optional title for the plot
    """
    y_points = reachability_data['y_points']
    z_points = reachability_data['z_points']
    flat_reachable = reachability_data['flat_reachable']
    perp_reachable = reachability_data['perp_reachable']
    
    # Create a colormap for the overlapping regions
    colors = [(0, 0, 0, 0), (1, 0, 0, 1), (0, 0, 1, 1), (0.5, 0, 0.5, 1)]
    positions = [0, 0.33, 0.67, 1.0]  # Fixed positions to start at 0 and end at 1
    cmap = LinearSegmentedColormap.from_list("reachability_cmap", list(zip(positions, colors)))
    
    # Create a combined reachability map
    combined = np.zeros(flat_reachable.shape, dtype=int)
    combined[flat_reachable] += 1  # Flat robot reachable points
    combined[perp_reachable] += 2  # Perpendicular robot reachable points
    
    plt.figure(figsize=(12, 8))
    plt.imshow(combined, cmap=cmap, origin='lower', 
               extent=[y_points.min(), y_points.max(), z_points.min(), z_points.max()])
    
    # Count points
    flat_count = np.sum(flat_reachable)
    perp_count = np.sum(perp_reachable)
    both_count = np.sum(np.logical_and(flat_reachable, perp_reachable))
    
    # Create a colorbar with custom labels
    cbar = plt.colorbar(ticks=[0, 0.33, 0.67, 1.0])
    cbar.set_ticklabels([
        'Not reachable', 
        f'Flat only ({flat_count} pts)', 
        f'Perpendicular only ({perp_count} pts)', 
        f'Both ({both_count} pts)'
    ])
    cbar.set_label('Reachability')
    
    plt.xlabel('Y Coordinate (m)')
    plt.ylabel('Z Coordinate (m)')
    
    if title:
        plt.title(title)
    else:
        plt.title('Robot Reachability Map')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reachability_map.png')
    plt.show() 