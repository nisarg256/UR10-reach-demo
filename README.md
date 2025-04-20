# UR10e Reach Comparison for Drywall Finishing

This project analyzes and compares the reachability of UR10e robotic arms in two different mounting configurations:
1. Flat mounted (parallel to the ground)
2. Perpendicularly mounted (perpendicular to the ground)

The analysis focuses on drywall finishing applications, where the robot must maintain the tool perpendicular to the wall while reaching various points.

## Prerequisites

- **Python 3.6+**
- **MuJoCo 2.3.2**: The main physics engine used for simulation
- **NumPy**: For numerical operations
- **SciPy**: For optimization (inverse kinematics)
- **Matplotlib**: For visualization of reachability maps
- **GLFW**: For rendering the MuJoCo simulation
- **PyYAML**: For configuration file handling

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/UR10-reach-demo.git
   cd UR10-reach-demo
   ```

2. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Ensure MuJoCo 2.3.2 is properly installed and configured on your system.

## Project Structure

```
UR10-reach-demo/
├── config/
│   └── simulation_params.yaml  # Configuration parameters for the simulation
├── models/
│   └── reach_comparison.xml    # MuJoCo model with both robot configurations
├── src/
│   ├── config/                 # Configuration handling
│   │   ├── __init__.py
│   │   └── config_loader.py    # Loads YAML configuration
│   ├── reachability/           # Reachability analysis
│   │   ├── __init__.py
│   │   └── workspace_analysis.py  # Core IK solver and workspace analysis
│   ├── trajectory/             # Trajectory generation
│   │   ├── __init__.py
│   │   └── s_pattern.py        # S-pattern trajectory generator
│   └── visualization/          # Visualization utilities
│       ├── __init__.py
│       └── render.py           # MuJoCo renderer and plotting functions
├── scripts/
│   ├── calculate_workspace.py  # Script to analyze and visualize workspace
│   ├── demonstrate_reach.py    # Script to demonstrate robot motion
│   └── visualize_results.py    # Script to visualize saved analysis results
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Configuration System

The project now uses a YAML-based configuration system (`config/simulation_params.yaml`) that centralizes all parameters:

```yaml
# Example configuration parameters
wall:
  x_position: 1.2  # meters

workspace:
  y_range: [-1.0, 1.0]    # Horizontal range
  z_range: [0.2, 2.2]     # Vertical range
  resolution: 0.05        # Grid resolution

robot:
  perp_angle_threshold: 0.15  # radians (about 8.6 degrees)
```

This allows easy adjustment of analysis parameters without modifying code.

## Detailed Component Description

### MuJoCo Model (`models/reach_comparison.xml`)

The XML model defines two UR10e robots with different mounting configurations:
- **Flat Mounted**: Robot base is parallel to the ground, typical tabletop setup
- **Perpendicular Mounted**: Robot base is mounted perpendicular to the ground (wall-mounted style)

Both configurations include a cylindrical tool attachment representing a drywall finishing tool. The model also includes a wall placed at X=1.2m.

### Reachability Analysis (`src/reachability/workspace_analysis.py`)

The `ReachabilityAnalyzer` class provides the core functionality:

- **Inverse Kinematics**: Uses SciPy's minimize function to solve IK with constraints
- **Perpendicularity Check**: Ensures the tool is perpendicular to the wall at each point
- **Workspace Analysis**: Maps out reachable points on the wall for both configurations
- **Spiral Search Pattern**: Analyzes points starting from the center and moving outward for efficiency

Key parameters that can be configured (via `simulation_params.yaml`):
- `y_range` and `z_range`: Define the area on the wall to analyze
- `resolution`: Controls the density of points checked
- `perp_angle_threshold`: Controls strictness of perpendicularity (lower values = stricter)
- `ik_tolerance`: Tolerance for IK solutions

### Trajectory Generation (`src/trajectory/s_pattern.py`)

The `generate_s_pattern` function generates S-shaped trajectories through the reachable workspace:
- Creates horizontal passes at different heights (configurable number)
- Alternates left-to-right and right-to-left for efficient coverage
- Produces a sequence of points and joint configurations for smooth motion
- Uses interpolation between grid points for smoother trajectories
- Point spacing is configurable for varying density of path points

### Visualization (`src/visualization/render.py`)

Contains two main components:
- `MujocoRenderer`: Uses GLFW to create a window and render the MuJoCo simulation in real-time
- `visualize_reachability`: Creates a color-coded map showing which areas of the wall are reachable by each robot configuration

### Scripts

#### Workspace Analysis Script (`scripts/calculate_workspace.py`)

This script:
1. Loads configuration from `simulation_params.yaml`
2. Initializes the `ReachabilityAnalyzer` with the model and config
3. Performs the reachability analysis for both robots
4. Saves the results along with the used configuration to a pickle file
5. Generates and displays a visualization of the reachable areas

Example usage:
```bash
./scripts/calculate_workspace.py
```

#### Demonstration Script (`scripts/demonstrate_reach.py`)

This script:
1. Loads configuration and previously calculated reachability data
2. Generates smooth S-pattern trajectories through the reachable workspace
3. Creates a MuJoCo visualization window
4. Allows the user to select which robot to demonstrate:
   - Flat mounted robot
   - Perpendicular mounted robot
   - Both robots simultaneously
5. Shows the robot moving through its reachable workspace while maintaining tool perpendicularity

Example usage:
```bash
./scripts/demonstrate_reach.py
```

#### Visualization Script (`scripts/visualize_results.py`)

A dedicated script to visualize previously saved analysis results without re-running the analysis:
```bash
./scripts/visualize_results.py
```

## Understanding Reachability Constraints

The reachability analysis enforces two main constraints:
1. **Position**: The tool tip must be able to reach the point on the wall
2. **Orientation**: The tool's principal axis must be perpendicular to the wall (aligned with X-axis)

The perpendicularity constraint can be adjusted through the `perp_angle_threshold` parameter:
- **Less strict (0.35 radians ≈ 20°)**: Allows more deviation, resulting in larger reachable workspace
- **More strict (0.15 radians ≈ 8.6°)**: Requires more precise perpendicularity, resulting in smaller but more accurate workspace

This is critical for drywall finishing applications, as it ensures the tool is properly oriented against the wall.

## Results Interpretation

The reachability map visualization uses color coding:
- **Red**: Points reachable only by the flat-mounted robot
- **Blue**: Points reachable only by the perpendicular-mounted robot
- **Purple**: Points reachable by both configurations
- **Transparent**: Points not reachable by either configuration

### Key Findings

Analysis with different perpendicularity constraints shows significant differences:

| Constraint | Flat Robot | Perp Robot | Both Robots |
|------------|------------|------------|-------------|
| 20° (0.35 rad) | 48.5% | 66.2% | 38.7% |
| 8.6° (0.15 rad) | 34.4% | 50.3% | 20.9% |

- **Perpendicular configuration advantage**: The perpendicular-mounted robot consistently shows better reach coverage (about 16% more) for drywall finishing tasks
- **Stricter perpendicularity reduces coverage**: Tighter perpendicularity requirements reduce reachable workspace for both configurations
- **Specialized workspaces**: With stricter constraints, robots show less overlap in reachable areas, suggesting specialized roles may be optimal

## Customization

To adjust the analysis parameters:
1. Edit `config/simulation_params.yaml` to change any parameters:
   - Wall position
   - Workspace boundaries and resolution
   - Perpendicularity threshold
   - Trajectory generation parameters
   - Visualization options
2. Run `scripts/calculate_workspace.py` to re-analyze with new parameters
3. Use `scripts/demonstrate_reach.py` or `scripts/visualize_results.py` to view results

## Troubleshooting

### Common Issues:

1. **OpenGL/GLFW errors**: Ensure you have proper graphics drivers and GLFW is correctly installed
2. **MuJoCo import errors**: Check that MuJoCo 2.3.2 is properly installed
3. **PyYAML import errors**: Ensure PyYAML is installed with `pip install pyyaml`
4. **Low reachability coverage**: Try adjusting the perpendicularity threshold in the configuration file

### Performance Tips:

- Reduce the resolution for faster analysis (e.g., 0.2m instead of 0.05m)
- Limit the Y and Z ranges to focus on areas of interest
- For large workspaces, consider running the analysis in parallel (requires code modification)