import json
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    def __init__(self, config_path='config.json'):
        if not os.path.exists(config_path):
            logging.error(f"Configuration file {config_path} not found.")
            sys.exit(1)
        with open(config_path, 'r') as f:
            try:
                self.config = json.load(f)
                logging.info(f"Configuration loaded from {config_path}.")
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing the configuration file: {e}")
                sys.exit(1)

    def get(self, *keys, default=None):
        """
        Access nested configuration parameters.
        Example: config.get('attacks', 'finetune', 'epochs')
        """
        cfg = self.config
        for key in keys:
            if isinstance(cfg, dict) and key in cfg:
                cfg = cfg[key]
            else:
                logging.warning(f"Configuration key {'.'.join(keys)} not found. Using default: {default}")
                return default
        return cfg if cfg else default


# Instantiate a global config object
try:
    config = Config()
except Exception as e:
    logging.error(f"Configuration Error: {e}")
    sys.exit(1)
