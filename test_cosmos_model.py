import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the Cosmos model"""
    model_id = "/workspace/models/Cosmos"
    
    logger.info(f"Testing model loading: {model_id}")
    
    try:
        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Create offload directory
        os.makedirs("offload", exist_ok=True)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Tokenizer loaded successfully")
        
        # Load model with memory optimizations
        logger.info("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Use BF16 precision as recommended
            device_map="auto",           # Automatically manage device placement
            offload_folder="offload",    # Folder for offloading
        )
        logger.info("Model loaded successfully")
        
        # Print model info
        logger.info(f"Model type: {type(model)}")
        
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