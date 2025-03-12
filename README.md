# CosmosScaling

Custom world creation with Cosmos Video to World model for frame prediction problem to create better simulated worlds.

## Project Overview

This project implements NVIDIA's Cosmos-1.0-Autoregressive-5B-Video2World model for video generation and prediction. The model can:
- Extend 9-frame video inputs to 33 frames (generating 24 new frames)
- Generate 32 frames from a single image input
- Create 3D videos at 1024x640 resolution

## Environment Setup

### RunPod Configuration

For the Cosmos-1.0-Autoregressive-5B-Video2World model, we selected:
- **GPU**: NVIDIA A40 (48GB) or A100 (80GB)
- **vCPUs**: 8-16 cores
- **RAM**: 64GB minimum
- **Storage**: 100GB+ SSD
- **Container**: NVIDIA PyTorch container (latest) - `nvcr.io/nvidia/pytorch:23.10-py3`


This configuration accommodates the model's significant VRAM requirements (41.3GB with offloading) as documented in NVIDIA's specifications.

### Container Setup

Our Dockerfile:
Install basic dependencies
```
RUN apt-get update && apt-get install -y \
ffmpeg \
libsm6 \
libxext6 \
git \
wget \
&& rm -rf /var/lib/apt/lists/
```

Install Python dependencies using pip 
```
RUN pip install --no-cache-dir -r requirements.txt
```

Install Python dependencies using pip install --no-cache-dir
```
RUN pip install --no-cache-dir \
transformers \
huggingface_hub \
opencv-python \
numpy \
python-dotenv
```

Install FastAPI and Uvicorn
```
RUN pip install fastapi uvicorn
```

Download the model weights
Create directory for model
mkdir -p /workspace/models/Cosmos
Download model
```
huggingface-cli download nvidia/Cosmos-1.0-Autoregressive-5B-Video2World \
--local-dir /workspace/models/Cosmos \
--include ".pt" ".json" ".md"
```


### 2. Repository Setup

Clone this repository
git clone https://github.com/paritoshk/CosmosScaling.git
cd CosmosScaling

### 3. Model Loading Test

Our initial tests revealed that the model is provided as a dictionary of weights rather than a ready-to-use model:


Load model directly with PyTorch
```
model_path = "/workspace/models/Cosmos/model.pt"
model_dict = torch.load(model_path, map_location="cuda")
```

## Result: Dictionary containing model weights
Key findings:
- The model loads as a Python dictionary (8.54 GB on GPU)
- No callable modules were found in the model dictionary
- NVIDIA provides the architecture code separately

### 4. Model Architecture Investigation

We found that NVIDIA hosts the model architecture code in a separate repository:



Runtime Engine: https://github.com/NVIDIA/Cosmos

From the model README, we learned:
- The model requires 41.3GB VRAM with partial offloading
- It can process both image and video inputs
- Generation takes ~73 seconds on an H100 GPU
- It has a low failure rate (2% with 9-frame video input)

## System Architecture

Based on our testing, we've designed the following system architecture:

1. **Model Loading Layer**: 
   - Load model weights from file
   - Integrate with NVIDIA's Cosmos architecture

2. **API Layer**: 
   - FastAPI service for inference requests
   - Support for both image and video inputs

3. **Video Processing Layer**:
   - Extract frames from input videos
   - Process 9-frame sequences for best results
   - Generate 24 new frames (extending to 33 total)

4. **Parallelization Layer**:
   - Optimize with CUDA and batch processing
   - Implement memory offloading strategies

## Next Steps

1. **Architecture Integration**:
   - Clone and integrate NVIDIA's Cosmos repository
   - Match our model weights with their architecture

2. **Offloading Implementation**:
   - Implement the recommended offloading strategy:
     - Guardrails & T5 encoder & Diffusion decoder & Tokenizer
   - Target 28.8GB VRAM usage

3. **API Development**:
   - Create FastAPI endpoints for video/image processing
   - Implement async processing for better throughput

## References
- [NVIDIA Cosmos GitHub](https://github.com/NVIDIA/Cosmos)
- [Model Technical Paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)

## Additional Notes for setting up the environment using NVIDIA's official container in RunPod

This template uses NVIDIA's official container which has all the following pre-installed:
NeMo Framework
Megatron-LM
transformer_engine
All CUDA/cuDNN dependencies
PyTorch Lightning
Other required libraries
Your A40 with 48GB VRAM is sufficient for running the 5B Cosmos models (which require ~40GB VRAM)
Make sure RunPod's network allows outbound connections to Hugging Face for downloading models
This template configuration will help you avoid all the dependency issues we encountered earlier!
# NVIDIA NeMo for Cosmos Models

This template includes the official NVIDIA NeMo container with all dependencies pre-installed
for running Cosmos Autoregressive World Foundation Models.

## Getting Started

1. Clone the Cosmos repository:

```
cd /workspace
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
```


## Important Notes
- Uses NVIDIA's official NeMo container with all dependencies pre-installed
- Requires a Hugging Face access token to download models
- Recommended GPU: A100-80GB or H100-80GB (minimum 40GB VRAM)
3. Run inference:

 
```
torchrun --nproc-per-node=1 cosmos1/models/autoregressive/nemo/inference/video2world.py \
--input_type video \
--input_image_or_video_path /path/to/your/video.mp4 \
--prompt "A detailed and realistic scene" \
--ar_model_dir nvidia/Cosmos-1.0-Autoregressive-5B-Video2World \
--video_save_name /workspace/generated_video.mp4
```