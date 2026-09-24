###########################################################################
### config.py
###
### Loads global and project-wide settings.
###########################################################################

import yaml
from os import path

CONFIG = {} 
    
def load_config(global_config = None, project_config = None):
    """
    Loads the default configuration file, then overwrites it with any settings found in 
    the global and project config files, if they exist.

    Args:
        global_config (str, optional): Path to the custom global config file, created by the user.
        project_config (str, optional): Path the the current project's config file, created by the user.
    """
    
    # Read default_config.yaml file from parent directory and save it as dictionary
    with open(path.join(path.dirname(__file__), 'default_config.yaml'), 'r') as f:
        CONFIG = yaml.safe_load(f)
        
    # If global_config is provided, read it and update the config dictionary
    if global_config is not None:
        with open(global_config, 'r') as f:
            global_config_dict = yaml.safe_load(f)
            CONFIG.update(global_config_dict)
            
    # If project_config is provided, read it and update the config dictionary
    if project_config is not None:
        with open(project_config, 'r') as f:
            project_config_dict = yaml.safe_load(f)
            CONFIG.update(project_config_dict)