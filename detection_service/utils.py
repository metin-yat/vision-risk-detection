import logging
from textwrap import indent
from inference import get_model
import numpy as np
import time, os, json
import torch
import cv2, threading
from collections import deque
from config import Config
from datetime import datetime
import redis

logger = logging.getLogger(__name__)

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "redis_service"), 
    port=6379, 
    db=0, 
    decode_responses=True
)

class SmartStreamer:
    def __init__(self, cap_object, batch_size=10, buffer_limit=50):
        self.cap = cap_object 
        self.batch_size = batch_size
        self.frame_buffer = deque(maxlen=buffer_limit)
        self.is_running = True
        self.lock = threading.Lock()
        
        # Counters for pipeline health monitoring
        self.blur_counter = 0
        self.motion_skip_counter = 0

        # Motion Detection
        self.avg_background = None 
        self.motion_threshold = 15  # Sensitivity: lower is more sensitive
        self.min_motion_area = 150  # Minimum pixel area to qualify as motion
        self.alpha = 0.1            # Background accumulation weight

    def start_stream(self):
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.is_running = False
                break

            # Scaling 
            scaled_frame = cv2.resize(frame, (640, 640))
            
            # Skip blurry frames
            is_blurry_, _ = is_blurry(scaled_frame, threshold=150.0)
            if is_blurry_:
                self.blur_counter += 1
                continue

            # Skip static frames
            if not self._detect_motion(scaled_frame):
                self.motion_skip_counter += 1
                continue
            
            with self.lock:
                self.frame_buffer.append(scaled_frame)

    def _detect_motion(self, frame):
        """
        Uses Background Subtraction to determine if enough movement exists.
        Returns True if motion is detected, False otherwise.
        """
        # Convert to grayscale and blur to reduce high-frequency noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize background if it's the first frame
        if self.avg_background is None:
            self.avg_background = gray.copy().astype("float")
            return True

        # Update the background model and compute the absolute difference
        cv2.accumulateWeighted(gray, self.avg_background, self.alpha)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_background))
        
        # Threshold the delta image to get a binary mask
        thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > self.min_motion_area:
                return True # Found significant movement
        
        return False # No movement detected

    def get_batch(self):
        with self.lock:
            if len(self.frame_buffer) >= self.batch_size:
                return [self.frame_buffer.popleft() for _ in range(self.batch_size)]
            
            if not self.is_running and len(self.frame_buffer) > 0:
                leftovers = list(self.frame_buffer)
                self.frame_buffer.clear()
                return leftovers
        return None

def initialize_model(model_id: str, api_key: str = "None"):
    """
    Downloads and initializes the Roboflow model.
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


def test_model(model):
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


def calculate_frame_ppe_score(people, helmets):
    """
    Calculates the average safety score for a single frame based on 
    the overlap between people and helmets.
    
    Args:
        people (list): List of person detection objects.
        helmets (list): List of helmet detection objects.
        
    Returns:
        float: Average safety score for the frame (0.0 to 1.0).
    """
    if not people:
        return None, None  # No people to analyze in this frame
    
    if not helmets:
        return 0.0, np.zeros((len(people), 0))  # People present but no helmets detected
    
    # Convert center-based coordinates (x, y, w, h) to corner-based (x1, y1, x2, y2)
    # Using NumPy for vectorized operations
    ppl_boxes = np.array([[p.x - p.width/2, p.y - p.height/2, p.x + p.width/2, p.y + p.height/2] for p in people])
    hmt_boxes = np.array([[h.x - h.width/2, h.y - h.height/2, h.x + h.width/2, h.y + h.height/2] for h in helmets])
    
    # Vectorized Overlap Calculation using Broadcasting
    # Compare every person (N) with every helmet (M) -> Shape: (N, M)
    x1 = np.maximum(ppl_boxes[:, None, 0], hmt_boxes[:, 0])
    y1 = np.maximum(ppl_boxes[:, None, 1], hmt_boxes[:, 1])
    x2 = np.minimum(ppl_boxes[:, None, 2], hmt_boxes[:, 2])
    y2 = np.minimum(ppl_boxes[:, None, 3], hmt_boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate area of helmets to find the percentage of helmet inside a person box
    helmet_areas = (hmt_boxes[:, 2] - hmt_boxes[:, 0]) * (hmt_boxes[:, 3] - hmt_boxes[:, 1])
    
    # Overlap ratio: How much of the helmet is contained within the person's bounding box
    overlap_ratios = intersection / helmet_areas  # Shape: (N, M)
    
    # For each person, find the helmet with the maximum overlap
    max_overlaps_per_person = np.max(overlap_ratios, axis=1)
    
    return np.mean(max_overlaps_per_person), overlap_ratios

def generate_som_analysis(image, people, helmets, overlap_matrix, threshold=0.70):
    """
    Applies Set-of-Marks (SoM) to the image and generates VLM-friendly metadata.
    Operates with high contrast colors and normalized coordinates.
    """
    height, width = image.shape[:2]
    marked_image = image.copy()
    
    metadata = {
        "canvas": {"size": [height, width], "unit": "normalized_0_1000"},
        "critical_alerts": [],
        "compliance_map": []
    }

    # Define high-contrast colors (BGR for OpenCV)
    COLOR_SAFE = (255, 255, 0)    # Cyan
    COLOR_RISK = (0, 0, 255)      # Red
    COLOR_HELMET = (0, 255, 255)  # Yellow
    COLOR_TEXT = (255, 255, 255)  # White

    # Process People and Assignments
    # Find max overlap for each person from the pre-calculated matrix
    # Shape of overlap_matrix: (N_people, M_helmets)
    if overlap_matrix is not None and overlap_matrix.size > 0:
        max_overlap_indices = np.argmax(overlap_matrix, axis=1)
        max_overlap_values = np.max(overlap_matrix, axis=1)
    else:
        max_overlap_indices = [None] * len(people)
        max_overlap_values = np.zeros(len(people))

    for i, p in enumerate(people):
        p_id = f"P{i+1}"
        score = max_overlap_values[i]
        is_safe = score >= threshold
        color = COLOR_SAFE if is_safe else COLOR_RISK
        
        # Bbox coordinates
        px1, py1 = int(p.x - p.width/2), int(p.y - p.height/2)
        px2, py2 = int(p.x + p.width/2), int(p.y + p.height/2)
        
        # Draw Person Bbox
        cv2.rectangle(marked_image, (px1, py1), (px2, py2), color, 2)
        
        # Label with background for readability
        label = p_id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(marked_image, (px1, py1), (px1 + tw, py1 + th + 5), color, -1)
        cv2.putText(marked_image, label, (px1, py1 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # Normalize coordinates for VLM (0-1000)
        norm_bbox = [int(py1/height*1000), int(px1/width*1000), int(py2/height*1000), int(px2/width*1000)]
        
        # Add to metadata
        mapping = {
            "p_id": p_id,
            "h_id": f"H{max_overlap_indices[i]+1}" if score > 0 else None,
            "overlap": round(float(score), 2),
            "status": "safe" if is_safe else "violation"
        }
        metadata["compliance_map"].append(mapping)

        if not is_safe:
            metadata["critical_alerts"].append({
                "id": p_id,
                "norm_bbox": norm_bbox,
                "reason": "insufficient_helmet_overlap"
            })

    # Process Helmets
    for j, h in enumerate(helmets):
        h_id = f"H{j+1}"
        hx1, hy1 = int(h.x - h.width/2), int(h.y - h.height/2)
        hx2, hy2 = int(h.x + h.width/2), int(h.y + h.height/2)
        
        cv2.rectangle(marked_image, (hx1, hy1), (hx2, hy2), COLOR_HELMET, 1)
        # Small ID for helmet
        cv2.putText(marked_image, h_id, (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HELMET, 1)

    return marked_image, metadata

def analyze_ppe_risk_batch(valid_frames_data, threshold=0.70):
    """
    Analyzes a batch of frames for PPE compliance and determines risk.
    
    Returns:
        tuple: (batch_score, representative_image, is_risky)
    """
    # logger.info(f"PPE Risk Analysis started for a batch of {len(valid_frames_data)} frames.")
    
    frame_scores = []
    rep_data = {}

    # Assuming batch_size is 10, index 4 or 5 is the middle (5th frame)
    target_rep_idx = len(valid_frames_data) // 2 

    for idx, data in enumerate(valid_frames_data):
        # frame_img = data.get('frame')
        predictions = data.get('predictions', [])
        
        # Split predictions into categories
        people = [p for p in predictions if p.class_name.lower() == 'person']
        helmets = [p for p in predictions if p.class_name.lower() == 'helmet']
        
        # Compute score for the current frame
        score, overlap_matrix = calculate_frame_ppe_score(people, helmets)
        
        if score is not None:
            frame_scores.append(score)
            
        # Capture the middle frame as the representative image
        if idx == target_rep_idx:
            rep_data = {
                "frame": data.get('frame'),
                "people": people,
                "helmets": helmets,
                "matrix": overlap_matrix
            }

    # Calculate final batch score (average of all frames with people)
    # If no people were found in the entire batch, we default to 1.0 (no risk)
    batch_score = sum(frame_scores) / len(frame_scores) if frame_scores else 1.0
    is_risky = batch_score < threshold

    # logger.info(f"Analysis finished. Score: {batch_score:.2f} | Risky: {is_risky}")

    # Return representative image only if the batch is considered risky
    final_rep_image = None;final_metadata = None

    if is_risky and rep_data.get("frame") is not None:
        # Generate the SoM image and JSON metadata
        final_rep_image, final_metadata = generate_som_analysis(
            rep_data["frame"], 
            rep_data["people"], 
            rep_data["helmets"], 
            rep_data["matrix"],
            threshold=threshold
        )
    
    return batch_score, final_rep_image, is_risky, final_metadata


def save_event_assets(selected_data, event_id):
    """
    It saves the images in the background and prepares the event_package.
    """
    event_dir = os.path.join(Config.RISKY_DIR, f"event_{event_id}")
    os.makedirs(event_dir, exist_ok=True)
    
    event_packet = {
        "event_id": event_id,
        "timestamp": datetime.now().isoformat(),
        "folder_path": event_dir,
        "snapshots": []
    }

    for i, data in enumerate(selected_data):
        img_filename = f"snapshot_{i+1}.jpg"
        img_path = os.path.join(event_dir, img_filename)
        
        # Saving the SoM image to mounted folder.
        cv2.imwrite(img_path, data['rep_image'])
        
        # Adding to the package
        event_packet["snapshots"].append({
            "image_path": img_path,
            "metadata": data['metadata']
        })

    try:
        logger.info(f"Event {event_id} successfully pushed to Redis queue.{event_dir}")
        redis_client.lpush("ppe_event_queue", json.dumps(event_packet))
    except Exception as e:
        logger.error(f"Failed to push event {event_id} to Redis: {e}")