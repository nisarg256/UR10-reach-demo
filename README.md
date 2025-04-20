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
├── models/
│   └── reach_comparison.xml  # MuJoCo model with both robot configurations
├── src/
│   ├── reachability/         # Reachability analysis
│   │   ├── __init__.py
│   │   └── workspace_analysis.py  # Core IK solver and workspace analysis
│   ├── trajectory/           # Trajectory generation
│   │   ├── __init__.py
│   │   └── s_pattern.py      # S-pattern trajectory generator
│   └── visualization/        # Visualization utilities
│       ├── __init__.py
│       └── render.py         # MuJoCo renderer and plotting functions
├── scripts/
│   ├── calculate_workspace.py  # Script to analyze and visualize workspace
│   └── demonstrate_reach.py    # Script to demonstrate robot motion
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

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

Key parameters that can be configured:
- `y_range` and `z_range`: Define the area on the wall to analyze
- `resolution`: Controls the density of points checked
- Perpendicularity threshold: Currently set to allow ~20 degrees of deviation

### Trajectory Generation (`src/trajectory/s_pattern.py`)

The `generate_s_pattern` function generates S-shaped trajectories through the reachable workspace:
- Creates horizontal passes at different heights
- Alternates left-to-right and right-to-left for efficient coverage
- Produces a sequence of points and joint configurations for smooth motion

### Visualization (`src/visualization/render.py`)

Contains two main components:
- `MujocoRenderer`: Uses GLFW to create a window and render the MuJoCo simulation in real-time
- `visualize_reachability`: Creates a color-coded map showing which areas of the wall are reachable by each robot configuration

### Scripts

#### Workspace Analysis Script (`scripts/calculate_workspace.py`)

This script:
1. Initializes the `ReachabilityAnalyzer` with the model
2. Defines the wall area to analyze and resolution
3. Performs the reachability analysis for both robots
4. Saves the results to a pickle file (`reachability_data.pkl`)
5. Generates and displays a visualization of the reachable areas

Example usage:
```bash
python3 scripts/calculate_workspace.py
```

#### Demonstration Script (`scripts/demonstrate_reach.py`)

This script:
1. Loads previously calculated reachability data
2. Generates S-pattern trajectories through the reachable workspace
3. Creates a MuJoCo visualization window
4. Allows the user to select which robot to demonstrate
5. Shows the robot moving through its reachable workspace while maintaining tool perpendicularity

Example usage:
```bash
python3 scripts/demonstrate_reach.py
```

## Understanding Reachability Constraints

The reachability analysis enforces two main constraints:
1. **Position**: The tool tip must be able to reach the point on the wall
2. **Orientation**: The tool's principal axis must be perpendicular to the wall (aligned with X-axis)

The perpendicularity constraint is critical for drywall finishing applications, as it ensures the tool is properly oriented against the wall.

## Results Interpretation

The reachability map visualization uses color coding:
- **Red**: Points reachable only by the flat-mounted robot
- **Blue**: Points reachable only by the perpendicular-mounted robot
- **Purple**: Points reachable by both configurations
- **Transparent**: Points not reachable by either configuration

The analysis typically shows that:
- The perpendicular robot has better reach for higher and lower points
- The flat robot has different coverage patterns
- There is significant overlap between the two configurations

## Customization

To adjust the analysis parameters:
1. Edit `scripts/calculate_workspace.py` to change the wall area or resolution
2. Modify the perpendicularity threshold in `ReachabilityAnalyzer._check_perpendicularity()`
3. Adjust the number of horizontal passes in `scripts/demonstrate_reach.py`

## Troubleshooting

### Common Issues:

1. **OpenGL/GLFW errors**: Ensure you have proper graphics drivers and GLFW is correctly installed
2. **MuJoCo import errors**: Check that MuJoCo 2.3.2 is properly installed
3. **Low reachability coverage**: Try adjusting the perpendicularity threshold or initial joint configurations

### Performance Tips:

- Reduce the resolution for faster analysis (e.g., 0.2m instead of 0.1m)
- Limit the Y and Z ranges to focus on areas of interest
- For large workspaces, consider running the analysis in parallel (requires code modification)