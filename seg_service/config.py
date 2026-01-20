# Helmet detection config

# File paths
RAW_IMAGES_DIR = "data/raw_images"
QUEUE_DIR = "data/queue"
MODEL_PATH = "models/yolov11m-seg.pt"

# Model settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3  # how much helmet overlaps with head

# Performance logging
FPS_LOG_INTERVAL = 10  # show fps every 10 frames