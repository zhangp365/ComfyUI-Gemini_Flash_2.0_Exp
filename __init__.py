import os
import json
import importlib.util
import logging
logger = logging.getLogger(__name__)
# Get the current directory and nodes directory
current_path = os.path.dirname(os.path.realpath(__file__))
nodes_path = os.path.join(current_path, "nodes")
config_path = os.path.join(nodes_path, "config.json")

# Create config if it doesn't exist or is empty/invalid
if not os.path.isfile(config_path) or os.path.getsize(config_path) == 0:
    config = {
        "GEMINI_API_KEY": "your key",
        "PROXY": "",
        "MODEL_NAME": "models/gemini-2.0-flash-exp",
        "RPM_LIMIT": 10,
        "TPM_LIMIT": 4000000,
        "RPD_LIMIT": 1500,
        "DEFAULT_CHAT_MODE": False
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Load config
try:
    with open(config_path, "r") as f:
        config = json.load(f)
except json.JSONDecodeError:
    config = {
        "GEMINI_API_KEY": "your key",
        "PROXY": "",
        "MODEL_NAME": "models/gemini-2.0-flash-exp",
        "RPM_LIMIT": 10,
        "TPM_LIMIT": 4000000,
        "RPD_LIMIT": 1500,
        "DEFAULT_CHAT_MODE": False
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Import the node modules properly
def load_module(file_name):
    module_path = os.path.join(nodes_path, file_name)
    if not os.path.exists(module_path):
        logger.error(f"Cannot find {module_path}")
        return None
    
    spec = importlib.util.spec_from_file_location(file_name[:-3], module_path)
    if spec is None:
        logger.error(f"Failed to create spec for {file_name}")
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.exception(f"Error loading {file_name}: {str(e)}")
        return None

# Load the modules
gemini_module = load_module("Gemini_Flash_Node.py")
audio_module = load_module("nodes_audio_recorder.py")

# Combine mappings if modules loaded successfully
if gemini_module and audio_module:
    NODE_CLASS_MAPPINGS = {
        **gemini_module.NODE_CLASS_MAPPINGS,
        **audio_module.NODE_CLASS_MAPPINGS
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **gemini_module.NODE_DISPLAY_NAME_MAPPINGS,
        **audio_module.NODE_DISPLAY_NAME_MAPPINGS
    }
elif gemini_module:
    NODE_CLASS_MAPPINGS = gemini_module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = gemini_module.NODE_DISPLAY_NAME_MAPPINGS
else:
    # Fallback for error cases
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Define web directory
WEB_DIRECTORY = os.path.join(current_path, "web")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
