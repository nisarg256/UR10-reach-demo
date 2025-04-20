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
    
    def _framebuffer_size_callback(self, window, width, height):
        """Handle window resizing."""
        self.width = width
        self.height = height
        
        # Update context
        mujoco.mjr_freeContext(self.context)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
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
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        
        return False  # Continue rendering
    
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