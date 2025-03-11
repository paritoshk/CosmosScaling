import os
import subprocess
import logging
import sys
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install all required dependencies for Cosmos"""
    try:
        logger.info("Installing NeMo and other dependencies...")
        
        # Install NeMo Framework and core dependencies
        subprocess.run([
            "pip", "install", "--no-cache-dir",
            "nemo_toolkit[all]>=1.20.0",
            "pytorch-lightning>=2.0.0",
            "hydra-core>=1.3.2",
            "omegaconf>=2.3.0",
            "transformers>=4.26.0",
            "sentencepiece>=0.1.97",
            "youtokentome>=1.0.6",
            "h5py>=3.7.0",
            "pyannote.audio>=2.1.1",
            "torchmetrics>=0.11.4",
            "torch-ema>=0.3",
            "numba>=0.57.0",
            "gradio>=3.33.1",
            "matplotlib>=3.7.1",
            "sacrebleu>=2.3.1",
            "nltk>=3.8.1"
        ], check=True)
        
        # Install video processing dependencies
        subprocess.run([
            "pip", "install", "--no-cache-dir",
            "imageio[ffmpeg]",
            "pyav",
            "iopath",
            "better_profanity",
            "peft",
            "einops",
            "opencv-python",
            "huggingface_hub"
        ], check=True)
        
        # Install Retinaface
        subprocess.run([
            "pip", "install", "--no-cache-dir",
            "git+https://github.com/NVlabs/Pytorch_Retinaface.git@b843f45"
        ], check=True)
        
        logger.info("Dependency installation complete")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
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
        
        # Install Cosmos as a package
        os.chdir("/workspace/Cosmos")
        subprocess.run(["pip", "install", "-e", "."], check=True)
        
        # Add to Python path
        sys.path.append("/workspace/Cosmos")
        
        logger.info("Cosmos repository setup complete")
        return True
    except Exception as e:
        logger.error(f"Error cloning/installing Cosmos: {str(e)}")
        return False

def prepare_model():
    """Prepare the model for use with NeMo"""
    try:
        logger.info("Preparing model for NeMo compatibility...")
        
        # The model needs to be in the correct location for NeMo
        hf_path = "/workspace/models"
        os.makedirs(f"{hf_path}/models--nvidia--Cosmos-1.0-Autoregressive-5B-Video2World/snapshots", exist_ok=True)
        model_dir = f"{hf_path}/models--nvidia--Cosmos-1.0-Autoregressive-5B-Video2World/snapshots/latest"
        
        # Create link if it doesn't exist
        if not os.path.exists(model_dir):
            os.symlink("/workspace/models/Cosmos", model_dir)
            logger.info(f"Created symbolic link to model at {model_dir}")
        
        # Also download other required models
        logger.info("Setting up environment for model download...")
        os.environ["HF_HOME"] = hf_path
        
        # Run the downloader script
        if os.path.exists("/workspace/Cosmos/cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py"):
            logger.info("Running model download script...")
            os.chdir("/workspace/Cosmos")
            try:
                subprocess.run(["python", "cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py"], check=True)
            except Exception as e:
                logger.warning(f"Could not run download script: {str(e)}. Will try direct inference.")
        
        return True
    except Exception as e:
        logger.error(f"Error preparing model: {str(e)}")
        return False

def run_inference():
    """Run inference with the Cosmos model"""
    try:
        logger.info("Running inference with Cosmos model...")
        
        # Ensure we're in the Cosmos directory
        os.chdir("/workspace/Cosmos")
        
        # Check if sample_video.mp4 exists in CosmosScaling
        input_video = "/workspace/CosmosScaling/sample_video.mp4"
        
        if not os.path.exists(input_video):
            logger.warning(f"Could not find {input_video}")
            # Try to find an alternative
            cosmos_sample = "/workspace/Cosmos/cosmos1/models/autoregressive/assets/v1p0/input.mp4"
            if os.path.exists(cosmos_sample):
                input_video = cosmos_sample
                logger.info(f"Using Cosmos sample video: {input_video}")
            else:
                logger.error("No sample video found")
                return False
        
        # Run inference 
        try:
            # First try with our existing model
            logger.info("Attempting inference with local model...")
            cmd = [
                "torchrun", "--nproc-per-node=1", 
                "cosmos1/models/autoregressive/nemo/inference/video2world.py",
                "--input_type", "video",
                "--input_image_or_video_path", input_video,
                "--prompt", "A detailed and realistic scene with natural motion",
                "--ar_model_dir", "/workspace/models/Cosmos",
                "--video_save_name", "/workspace/CosmosScaling/cosmos_generated_video.mp4"
            ]
            subprocess.run(cmd, check=True)
        except Exception as e:
            logger.warning(f"Inference with local model failed: {str(e)}")
            
            # Try with Hugging Face model path
            logger.info("Attempting inference with HF model path...")
            cmd = [
                "torchrun", "--nproc-per-node=1", 
                "cosmos1/models/autoregressive/nemo/inference/video2world.py",
                "--input_type", "video",
                "--input_image_or_video_path", input_video,
                "--prompt", "A detailed and realistic scene with natural motion",
                "--ar_model_dir", "nvidia/Cosmos-1.0-Autoregressive-5B-Video2World",
                "--video_save_name", "/workspace/CosmosScaling/cosmos_generated_video.mp4"
            ]
            subprocess.run(cmd, check=True)
        
        logger.info("Inference completed successfully!")
        logger.info("Generated video saved as /workspace/CosmosScaling/cosmos_generated_video.mp4")
        return True
    except Exception as e:
        logger.error(f"Error running inference: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Cosmos environment setup...")
    
    if install_dependencies() and clone_cosmos_repo() and prepare_model():
        success = run_inference()
        if success:
            logger.info("Process completed successfully!")
        else:
            logger.error("Process failed during inference")
    else:
        logger.error("Failed to setup environment")