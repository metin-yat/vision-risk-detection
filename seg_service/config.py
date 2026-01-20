import os
from pathlib import Path

# Use environment variables if they exist, otherwise use your defaults
class Config:
    SOURCE_DIR = os.getenv("SOURCE_DIR", "source/safe_construction_zone.mp4")
    QUEUE_DIR = os.getenv("QUEUE_DIR", "data/queue")
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "person-helmet-2dfvf/1")#"helmet-person-person_with_helmet/1")
    
    # Hyperparameters
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.3))

    # Logging
    FPS_LOG_INTERVAL = int(os.getenv("FPS_LOG_INTERVAL", 10))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")