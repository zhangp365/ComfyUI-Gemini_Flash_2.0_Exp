# __init__.py
import os
import json

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")

# Create config if it doesn't exist or is empty/invalid
if not os.path.isfile(config_path) or os.path.getsize(config_path) == 0:
    config = {
        "GEMINI_API_KEY": "your key",
        "PROXY": "",
        "MODEL_NAME": "models/gemini-2.0-flash-exp",
        "RPM_LIMIT": 10,           # Requests per minute
        "TPM_LIMIT": 4000000,      # Tokens per minute
        "RPD_LIMIT": 1500,         # Requests per day
        "DEFAULT_CHAT_MODE": False  # Default chat mode setting
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

from .Gemini_Flash_Node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']