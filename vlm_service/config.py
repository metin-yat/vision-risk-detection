import os

class Config:
    REDIS_HOST = os.getenv("REDIS_HOST", "redis_service")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    INPUT_QUEUE = os.getenv("QUEUE_NAME", "ppe_event_queue")
    OUTPUT_QUEUE = os.getenv("RESULT_QUEUE", "vlm_results_queue")
    
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/smolvlm")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 300))
    
    SCENES_DIR = os.getenv("SCENES_DIR", "/app/scenes")
    
    SYSTEM_PROMPT = (
        "You are a safety compliance assistant. You will receive a sequence of 3 images "
        "from a construction site representing a 4-second window. "
        "Analyze the temporal progression of the people marked in the images. "
        "Identify if safety violations (like missing helmets) are persistent, "
        "resolved, or new across the sequence."
    )
