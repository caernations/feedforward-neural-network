import yaml
from typing import Dict, Any

def load_config(config_path: str = 'src/configs/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters:
    -----------
    config_path : str, optional
        Path to the configuration file (default is 'config.yaml')

    Returns:
    --------
    Dict[str, Any]
        Loaded configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        raise