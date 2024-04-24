import logging
from pathlib import Path
from logging.config import dictConfig
from config import get_settings

config = get_settings()
APP_NAME = config.APP_NAME
LOG_DIR = Path(config.LOG_DIRECTORY)

# Ensure the log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define a simple logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
            'level': 'DEBUG',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': LOG_DIR / f'{APP_NAME}.log',
            'maxBytes': 5000000, # 5 MB
            'backupCount': 5,
        },
    },
    'loggers': {
        '': { # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        APP_NAME: { # specific logger for the application
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}

def setup_logging():
    """ Setup logging configuration 
    """
    dictConfig(LOGGING_CONFIG)

def get_logger(name=None):
    """
    Get a configured logger.
    """
    return logging.getLogger(name or APP_NAME)

setup_logging()