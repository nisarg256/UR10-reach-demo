import os
import yaml
from pathlib import Path

def load_config(config_file=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to the configuration file. If None, the default config is used.
        
    Returns:
        dict: Configuration dictionary
    """
    # If no config file provided, use the default one
    if config_file is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_file = os.path.join(project_root, "config", "simulation_params.yaml")
    
    # Load the configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 