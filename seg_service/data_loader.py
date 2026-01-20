"""
Data loading utilities for video, RTSP streams, and image folders.
Provides unified OpenCV-compatible interface for all source types.
"""

import cv2
from pathlib import Path
from typing import Tuple, Optional



class ImageSequenceCapture:
    """
    OpenCV VideoCapture-like interface for image sequences.
    
    Supports .jpg, .png, .jpeg files in a directory.
    Mimics cv2.VideoCapture API for seamless integration.
    """
    
    def __init__(self, image_dir: str):
        # Get all image files (jpg, png, jpeg)
        self.image_paths = sorted(
            Path(image_dir).glob('*.[jp][pn][g]')
        )
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        self.index = 0
        self.total_frames = len(self.image_paths)
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read next frame (mimics cv2.VideoCapture.read())"""
        if self.index >= len(self.image_paths):
            return False, None
        
        frame = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1
        return True, frame
    
    def get(self, prop_id: int) -> float:
        """Get video property (mimics cv2.VideoCapture.get())"""
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0  # Default FPS for image sequences
        return 0.0
    
    def release(self):
        """Release resources (compatibility)"""
        pass


def get_source(source_path: str):
    """
    Get OpenCV-compatible capture object from various sources.

    Args:
        source_path: Path to video file, RTSP URL, or image directory

    Returns:
        cv2.VideoCapture or ImageSequenceCapture object
    """
    
    if source_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video file
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source_path}")
        return cap
    
    elif source_path.startswith('rtsp://'):
        # RTSP stream
        cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot connect to RTSP: {source_path}")
        return cap
    
    elif Path(source_path).is_dir():
        # Image folder
        return ImageSequenceCapture(source_path)
    
    else:
        raise ValueError(
            f"Unsupported source: {source_path}\n"
            "Supported: .mp4 video, rtsp:// stream, or image directory"
        )