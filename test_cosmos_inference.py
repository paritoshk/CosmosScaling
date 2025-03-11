import os
import subprocess
import logging
import sys
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for Cosmos"""
    try:
        # Install required packages
        logger.info("Installing required packages...")
        subprocess.run(
            ["pip", "install", "--no-cache-dir", "imageio[ffmpeg]", "pyav", "iopath", "better_profanity", "peft", 
             "git+https://github.com/NVlabs/Pytorch_Retinaface.git@b843f45"],
            check=True
        )
        
        # Set environment variables - use our already downloaded model
        os.environ["HF_HOME"] = "/workspace/models"
        
        logger.info("Environment setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

def clone_cosmos_repo():
    """Clone the Cosmos repository"""
    try:
        logger.info("Cloning NVIDIA Cosmos repository...")
        
        # Check if directory exists already
        if os.path.exists("/workspace/Cosmos"):
            logger.info("Cosmos repository already exists, skipping clone")
        else:
            subprocess.run(
                ["git", "clone", "https://github.com/NVIDIA/Cosmos.git", "/workspace/Cosmos"],
                check=True
            )
        
        # Change to the Cosmos directory
        os.chdir("/workspace/Cosmos")
        
        # Pull large files
        logger.info("Pulling sample input video...")
        subprocess.run(
            ["git", "lfs", "pull", "cosmos1/models/autoregressive/assets/v1p0/input.mp4"],
            check=True
        )
        
        logger.info("Cosmos repository setup complete")
        return True
    except Exception as e:
        logger.error(f"Error cloning Cosmos repository: {str(e)}")
        return False

def run_inference():
    """Run inference with the Cosmos model using the official script"""
    try:
        logger.info("Running inference with Cosmos model...")
        
        # Check and copy your sample video if it exists
        if os.path.exists("/workspace/CosmosScaling/sample_video.mp4"):
            shutil.copy("/workspace/CosmosScaling/sample_video.mp4", "/workspace/Cosmos/sample_video.mp4")
            input_video = "sample_video.mp4"
            logger.info(f"Using your custom video: {input_video}")
        else:
            input_video = "cosmos1/models/autoregressive/assets/v1p0/input.mp4"
            logger.info(f"Using default video: {input_video}")
        
        # Create a symbolic link to your downloaded model within HF_HOME structure
        os.makedirs("/workspace/models/models--nvidia--Cosmos-1.0-Autoregressive-5B-Video2World/snapshots", exist_ok=True)
        model_link_dir = "/workspace/models/models--nvidia--Cosmos-1.0-Autoregressive-5B-Video2World/snapshots/latest"
        
        if not os.path.exists(model_link_dir):
            os.symlink("/workspace/models/Cosmos", model_link_dir)
            logger.info(f"Created symbolic link to your downloaded model at {model_link_dir}")
        
        # Run inference with the 5B Video2World model
        subprocess.run([
            "torchrun", "--nproc-per-node=1", 
            "cosmos1/models/autoregressive/nemo/inference/video2world.py",
            "--input_type", "video",
            "--input_image_or_video_path", input_video,
            "--prompt", "A detailed and realistic scene with natural motion",
            "--ar_model_dir", "/workspace/models/Cosmos",
            "--video_save_name", "/workspace/CosmosScaling/cosmos_generated_video.mp4"
        ], check=True)
        
        logger.info("Inference completed successfully!")
        logger.info("Generated video saved as /workspace/CosmosScaling/cosmos_generated_video.mp4")
        return True
    except Exception as e:
        logger.error(f"Error running inference: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Cosmos inference process...")
    
    if setup_environment() and clone_cosmos_repo():
        success = run_inference()
        if success:
            logger.info("Process completed successfully!")
        else:
            logger.error("Process failed during inference")
    else:
        logger.error("Failed to setup environment or clone repository")