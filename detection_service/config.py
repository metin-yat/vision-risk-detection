import os
from pathlib import Path

class Config:
    SOURCE_DIR = os.getenv("SOURCE_DIR", "source/false1.mp4")
    QUEUE_DIR = os.getenv("QUEUE_DIR", "data/queue")
    RISKY_DIR = os.getenv("RISKY_DIR", "scenes")

    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "construction-safety-gsnvb/1")
    #"person-helmet-2dfvf/1")#"helmet-person-person_with_helmet/1")
    
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.1))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.6))
    WINDOW_SIZE = os.getenv("WINDOW_SIZE", 12)
    RISK_THRESHOLD = os.getenv("RISK_THRESHOLD", 0.90)