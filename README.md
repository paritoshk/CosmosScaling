# 🌌 CosmosScaling

> Custom world creation with NVIDIA's Cosmos Video2World model for advanced frame prediction and simulated environment generation.

![Cosmos Banner](https://img.shields.io/badge/NVIDIA-Cosmos-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Model](https://img.shields.io/badge/Model-5B%20Parameters-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-NVIDIA%20Open-orange?style=for-the-badge)
![Education](https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/)

## 🚀 Project Overview

This project harnesses NVIDIA's Cosmos-1.0-Autoregressive-5B-Video2World model for state-of-the-art video generation and prediction capabilities. The model delivers impressive results:

- 📊 Extends 9-frame video inputs to 33 frames (generating 24 new frames)
- 🖼️ Generates 32 frames from a single image input
- 🎞️ Creates high-quality 3D videos at 1024×640 resolution

## ⚙️ Environment Setup

### 🖥️ RunPod Configuration

For optimal performance with the Cosmos-1.0-Autoregressive-5B-Video2World model, we recommend:

Using ```nvcr.io/nvidia/nemo:24.12``` version of NeMO since RunPod has outdated NVDA drivers. 

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A40 (48GB) or A100 (80GB) |
| vCPUs | 8-16 cores |
| RAM | 64GB minimum |
| Storage | 100GB+ SSD |
| Container | `nvcr.io/nvidia/nemo:25.02.rc1` |

> ⚠️ **Note**: The model requires significant VRAM (41.3GB with offloading) as per NVIDIA's documentation.

### 🔧 Container Configuration

#### Basic Dependencies

```bash
# Update system and install prerequisites
apt-get update && apt-get install -y \
  ffmpeg \
  libsm6 \
  libxext6 \
  git \
  wget \
  && rm -rf /var/lib/apt/lists/
```

#### Python Dependencies

```bash
# Install required Python packages
pip install --no-cache-dir \
  transformers \
  huggingface_hub \
  opencv-python \
  numpy \
  python-dotenv \
  fastapi \
  uvicorn
```

#### Model Download

```bash
# Create directory for model
mkdir -p /workspace/models/Cosmos

# Download model weights
huggingface-cli download nvidia/Cosmos-1.0-Autoregressive-5B-Video2World \
  --local-dir /workspace/models/Cosmos \
  --include ".pt" ".json" ".md"
```

## 📥 Repository Setup

```bash
# Clone this repository
git clone https://github.com/paritoshk/CosmosScaling.git
cd CosmosScaling
```

## 🧠 Model Architecture

Our investigation revealed:

```python
# Load model directly with PyTorch
model_path = "/workspace/models/Cosmos/model.pt"
model_dict = torch.load(model_path, map_location="cuda")
```

**Key Findings:**
- The model loads as a Python dictionary (8.54 GB on GPU)
- No callable modules were found in the model dictionary
- NVIDIA provides the architecture code separately via [Cosmos GitHub Repository](https://github.com/NVIDIA/Cosmos)

### 📊 Model Specifications

- 🧠 VRAM: 41.3GB with partial offloading
- ⏱️ Generation Time: ~73 seconds on H100 GPU
- 🎯 Error Rate: 2% with 9-frame video input

## 🏗️ System Architecture

![System Architecture](https://img.shields.io/badge/Architecture-4--Layer-success?style=for-the-badge)

1. **Model Loading Layer** 📥
   - Load model weights from file
   - Integrate with NVIDIA's Cosmos architecture

2. **API Layer** 🌐
   - FastAPI service for inference requests
   - Support for both image and video inputs

3. **Video Processing Layer** 🎬
   - Extract frames from input videos
   - Process 9-frame sequences for best results
   - Generate 24 new frames (extending to 33 total)

4. **Parallelization Layer** ⚡
   - Optimize with CUDA and batch processing
   - Implement memory offloading strategies

## 🛣️ Next Steps

1. **Architecture Integration**
   - Clone and integrate NVIDIA's Cosmos repository
   - Match model weights with their architecture

2. **Offloading Implementation**
   - Implement recommended offloading strategy:
     - Guardrails & T5 encoder & Diffusion decoder & Tokenizer
   - Target 28.8GB VRAM usage

3. **API Development**
   - Create FastAPI endpoints for video/image processing
   - Implement async processing for better throughput

## 📋 Quick Start Guide for RunPod

This template uses NVIDIA's official NeMo container with all dependencies pre-installed:

1. Clone the Cosmos repository:

```bash
cd /workspace
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
```

2. Set up environment variables:

```bash
export HF_TOKEN="your_hugging_face_token"
export HF_HOME="/workspace/hf_cache"
```

3. Run inference:

```bash
torchrun --nproc-per-node=1 cosmos1/models/autoregressive/nemo/inference/video2world.py \
  --input_type video \
  --input_image_or_video_path /path/to/your/video.mp4 \
  --prompt "A detailed and realistic scene" \
  --ar_model_dir nvidia/Cosmos-1.0-Autoregressive-5B-Video2World \
  --video_save_name /workspace/generated_video.mp4
```

## 📚 References

- [NVIDIA Cosmos GitHub](https://github.com/NVIDIA/Cosmos)
- [Model Technical Paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)

---

<p align="center">
  <b>Developed with ❤️ for advanced AI video generation</b>
</p>
