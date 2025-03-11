import os
import torch
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the Cosmos model using direct PyTorch loading"""
    model_path = "/workspace/models/Cosmos/model.pt"
    
    logger.info(f"Testing model loading: {model_path}")
    
    try:
        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Will attempt to load model on CPU, but this may fail due to memory constraints.")
        
        # Create offload directory
        os.makedirs("offload", exist_ok=True)
        
        # Try to add model directory to path to find modules
        cosmos_dir = "/workspace/models/Cosmos"
        if cosmos_dir not in sys.path:
            sys.path.append(cosmos_dir)
        
        # First attempt: Check if there's a NeMo module
        try:
            logger.info("Attempting to load via NeMo module if available...")
            from nemo.Cosmos1Autoregressive5BVideo2World import Cosmos1Autoregressive5BVideo2World
            
            model = Cosmos1Autoregressive5BVideo2World.from_pretrained(model_path)
            logger.info(f"Model loaded successfully via NeMo module: {type(model)}")
        except ImportError:
            logger.info("NeMo module not found, trying direct PyTorch loading...")
            
            # Second attempt: Direct PyTorch loading
            logger.info(f"Loading model from {model_path}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to load the model with different options
            try:
                model = torch.load(model_path, map_location=device)
                logger.info(f"Model loaded successfully via direct loading: {type(model)}")
            except Exception as e:
                logger.error(f"Error loading model directly: {str(e)}")
                
                # Try with different map_location strategies
                try:
                    logger.info("Attempting alternative loading strategy...")
                    model = torch.load(model_path, map_location="cpu")
                    if hasattr(model, "to") and torch.cuda.is_available():
                        model = model.to("cuda")
                    logger.info(f"Model loaded successfully with alternative strategy: {type(model)}")
                except Exception as e:
                    logger.error(f"All loading attempts failed: {str(e)}")
                    return False
        
        # Print model info
        if hasattr(model, "parameters"):
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model has {param_count:,} parameters")
            
            if torch.cuda.is_available():
                # Check where the model is
                try:
                    device_info = next(model.parameters()).device
                    logger.info(f"Model is on device: {device_info}")
                except:
                    logger.info("Could not determine model device")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)    # Convert to GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            logger.info(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        logger.info("Model test completed successfully")
    else:
        logger.error("Model test failed") 