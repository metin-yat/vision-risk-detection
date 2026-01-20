import logging
from inference import get_model
import numpy as np
import time
import torch
import cv2, threading
from collections import deque

logger = logging.getLogger(__name__)

class SmartStreamer:
    def __init__(self, cap_object, batch_size=10, buffer_limit=50):
        # We take the object returned by data_loader.get_source()
        self.cap = cap_object 
        self.batch_size = batch_size
        self.frame_buffer = deque(maxlen=buffer_limit)
        self.is_running = True
        self.lock = threading.Lock()
        self.blur_counter = 0

    def start_stream(self):
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.is_running:
            # This works for both cv2.VideoCapture and data_loader.ImageSequenceCapture
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.is_running = False
                break

            scaled_frame = cv2.resize(frame, (640, 480))
            
            # Only high-quality frames. 
            is_blurry_, _ = is_blurry(frame, threshold=150.0)
            if is_blurry_:
                self.blur_counter += 1
                continue
            
            with self.lock:
                self.frame_buffer.append(scaled_frame)

    def get_batch(self):
        with self.lock:
            if len(self.frame_buffer) >= self.batch_size:
                return [self.frame_buffer.popleft() for _ in range(self.batch_size)]
        return None

def initialize_segmentation_model(model_id: str, api_key: str = "None"):
    """
    Downloads and initializes the Roboflow segmentation model.
    """
    logger.info(f"Starting model loading process for Model ID: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Checking hardware... Using: {device}")

    try:
        if api_key != "None":
            model = get_model(model_id=model_id, api_key=api_key)
        else: model = get_model(model_id=model_id)

        if model is None:
            logger.error("Model object returned None!")
            raise ValueError("Model loading failed.")
            
        logger.info(f"Model successfully loaded: {model_id}")
        return model

    except Exception as e:
        logger.critical(f"Critical error during model initialization: {e}", exc_info=True)
        raise SystemExit("Application terminated: Model could not be loaded.")


def test_segmentation_model(model):
    """
    Checks if the model is loaded properly by sending a dummy black picture.
    Returns True if successful, False otherwise.
    """
    logger.info("Starting model test with dummy input...")
    start_time = time.perf_counter()
    
    # Create a 640x640 black image (H, W, C)
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        results = model.infer(dummy_input)
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Model warm-up latency: {latency:.2f}ms")
        return True
        
    except Exception as e:
        logger.error(f"Model test failed! Error: {str(e)}", exc_info=True)
        return False

def is_blurry(frame, threshold=100.0):
    """
    Checks if a frame is blurry using the Variance of Laplacian method.
    Returns True if the variance is below the threshold (indicating blur).
    """
    # Convert to grayscale to reduce computational cost
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the Laplacian of the image and then the focus measure (variance)
    # CV_64F is used to handle negative values during the derivative calculation
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If the variance is less than the threshold, the image is considered blurry
    return variance < threshold, variance