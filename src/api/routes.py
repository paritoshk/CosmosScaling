from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
import logging
from ..model.inference import InferenceEngine
from ..utils.video_processing import extract_frames, resize_image, create_temp_video_from_image

router = APIRouter()
inference_engine = InferenceEngine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/initialize")
async def initialize_model():
    """Initialize the model"""
    success = await inference_engine.initialize()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize model")
    return {"status": "Model initialized successfully"}

@router.post("/generate/from_video")
async def generate_from_video(
    video: UploadFile = File(...),
    text_prompt: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Generate future frames from a video
    
    Args:
        video: Input video file
        text_prompt: Text description
        
    Returns:
        Path to generated video
    """
    try:
        # Save uploaded video to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"input_{video.filename}")
        
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Generate frames
        output_path = await inference_engine.generate_from_video(
            video_path=temp_video_path,
            text_prompt=text_prompt
        )
        
        if not output_path:
            raise HTTPException(status_code=500, detail="Failed to generate video")
        
        # Clean up temporary files
        if background_tasks:
            background_tasks.add_task(os.remove, temp_video_path)
            background_tasks.add_task(os.remove, output_path)
        
        return FileResponse(output_path, media_type="video/mp4", filename="generated_video.mp4")
    except Exception as e:
        logger.error(f"Error in generate_from_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/from_image")
async def generate_from_image(
    image: UploadFile = File(...),
    text_prompt: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Generate future frames from an image
    
    Args:
        image: Input image file
        text_prompt: Text description
        
    Returns:
        Path to generated video
    """
    try:
        # Save uploaded image to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, f"input_{image.filename}")
        
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Generate frames
        output_path = await inference_engine.generate_from_image(
            image_path=temp_image_path,
            text_prompt=text_prompt
        )
        
        if not output_path:
            raise HTTPException(status_code=500, detail="Failed to generate video")
        
        # Clean up temporary files
        if background_tasks:
            background_tasks.add_task(os.remove, temp_image_path)
            background_tasks.add_task(os.remove, output_path)
        
        return FileResponse(output_path, media_type="video/mp4", filename="generated_video.mp4")
    except Exception as e:
        logger.error(f"Error in generate_from_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 