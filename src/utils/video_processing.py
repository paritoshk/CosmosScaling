import os
import cv2
import numpy as np
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path, num_frames=9):
    """
    Extract frames from a video
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        
    Returns:
        List of numpy arrays containing frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            # Evenly space the frames
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to 1024x640 as required by the model
                frame = cv2.resize(frame, (1024, 640))
                frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []

def resize_image(image_path, target_size=(1024, 640)):
    """
    Resize an image to the target size
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
        
    Returns:
        Resized image as numpy array
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None
            
        resized = cv2.resize(img, target_size)
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return None

def save_frames_as_video(frames, output_path, fps=30):
    """
    Save frames as a video
    
    Args:
        frames: List of numpy arrays containing frames
        output_path: Path to save the video
        fps: Frames per second
        
    Returns:
        Path to saved video
    """
    try:
        if not frames:
            logger.error("No frames to save")
            return None
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        logger.info(f"Video saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return None

def create_temp_video_from_image(image_path, num_frames=9):
    """
    Create a temporary video from a single image
    
    Args:
        image_path: Path to input image
        num_frames: Number of frames in the video
        
    Returns:
        Path to temporary video
    """
    try:
        img = resize_image(image_path)
        if img is None:
            return None
            
        frames = [img.copy() for _ in range(num_frames)]
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"temp_video_{os.getpid()}.mp4")
        
        return save_frames_as_video(frames, temp_video_path)
    except Exception as e:
        logger.error(f"Error creating temp video: {str(e)}")
        return None 