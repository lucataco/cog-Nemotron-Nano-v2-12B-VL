# NVIDIA Nemotron Nano VL 12B V2 - Replicate Cog Implementation

A [Replicate](https://replicate.com) implementation of NVIDIA's Nemotron Nano VL 12B V2 vision-language model using [Cog](https://github.com/replicate/cog). This implementation supports single and multi-image inference (up to 4 images), video understanding with automatic frame extraction, and 128K context length.

## Features

- **Multi-Image Support**: Process 1-4 images simultaneously for comparative analysis and multi-page document processing
- **Video Understanding**: Automatic video frame extraction with configurable FPS and pruning rate
- **Flexible Input**: Support for both images (JPEG/PNG) and videos (MP4)
- **Advanced Generation**: Temperature, top-p sampling, and configurable output length
- **Optimized Processing**: Automatic parameter defaults based on input type (images vs video)
- **Production Ready**: Clean output extraction with automatic response parsing

## Quick Start

### Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- NVIDIA GPU with CUDA support (tested on A100)
- Docker

### Test Locally

**Single Image:**
```bash
cog predict -i images=@demo.jpg -i prompt="Describe this image in detail"
```

**Multiple Images:**
```bash
cog predict \
  -i images=@page1.jpg \
  -i images=@page2.jpg \
  -i prompt="Compare these two images"
```

**Video:**
```bash
cog predict \
  -i video=@demo.mp4 \
  -i prompt="Summarize what happens in this video" \
  -i video_fps=2
```

## Project Structure

```
.
├── predict.py              # Main Cog predictor implementation
├── cog.yaml               # Cog configuration
├── requirements.txt       # Python dependencies
├── checkpoints/           # Model files (downloaded on first run)
│   ├── modeling.py        # VLM model implementation
│   ├── processing.py      # Multi-modal processor
│   ├── video_io.py        # Video frame handling utilities
│   └── ...                # Additional model components
├── C_RADIOv2_H/          # Vision encoder (downloaded on first run)
├── REPLICATE.md          # User-facing model documentation
├── CLAUDE.md             # Development documentation
└── MODEL.md              # Model card and specifications
```

## Implementation Details

### Multi-Image Processing

The implementation handles 1-4 images by:
1. Collecting non-None image inputs into a list
2. Creating multiple image placeholders in the message content
3. Passing all images to the processor in a single batch
4. Using the same inference pipeline as single-image mode

### Video Processing

Videos are processed automatically:
1. Frames extracted using ffmpeg at specified FPS
2. Frames saved to temporary directory
3. Loaded and processed with metadata (FPS, duration)
4. Temporary files cleaned up after inference
5. Efficient Video Sampling (EVS) applied based on pruning rate

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | List[Path] | None | Input images (1-4 supported) |
| `video` | Path | None | Input video (MP4) |
| `prompt` | String | "Describe what you see in detail." | Text prompt |
| `video_fps` | Integer | 1 | Frames per second to extract (1-30) |
| `video_pruning_rate` | Float | 0.75 | Video efficiency vs quality (0.0-1.0) |
| `max_new_tokens` | Integer | Auto | Max output tokens (1-2048) |
| `temperature` | Float | 0.0 | Sampling temperature (0.0-2.0) |
| `top_p` | Float | 1.0 | Nucleus sampling (0.0-1.0) |
| `system_prompt` | String | "/no_think" | System instruction |

**Note**: Images and video are mutually exclusive. Auto-defaults: 1024 tokens for images, 512 for videos.

## Development

### Building the Model

```bash
# Build the Cog container
cog build

# Test the build
cog predict -i images=@test.jpg -i prompt="Test prompt"
```

### Pushing to Replicate

```bash
# Login to Replicate
cog login

# Push to your Replicate account
cog push r8.im/your-username/nemotron-nano-vl-12b-v2
```

### Model Weights

Model weights are automatically downloaded on first run using `pget`:
- **Main Model**: Downloaded to `./checkpoints/`
- **Vision Encoder**: Downloaded to `./C_RADIOv2_H/`

For local development with pre-downloaded weights, place them in the respective directories.

### Testing Different Scenarios

**Document Intelligence:**
```bash
cog predict \
  -i images=@invoice.jpg \
  -i prompt="Extract all line items with quantities and prices"
```

**Multi-Page Document:**
```bash
cog predict \
  -i images=@contract_p1.jpg \
  -i images=@contract_p2.jpg \
  -i images=@contract_p3.jpg \
  -i prompt="Summarize the key terms and obligations in this contract"
```

**Video with Custom Settings:**
```bash
cog predict \
  -i video=@tutorial.mp4 \
  -i prompt="List the main steps shown in this tutorial" \
  -i video_fps=3 \
  -i video_pruning_rate=0.6 \
  -i max_new_tokens=1024
```

## Model Information

**Base Model**: [NVIDIA Nemotron Nano VL 12B V2](https://catalog.ngc.nvidia.com/)
- **Parameters**: 12.6B
- **Context Length**: 128K tokens
- **Vision Encoder**: C-RADIOv2-H
- **Language Model**: Nemotron Nano V2 (H-architecture)
- **Supported Languages**: English, German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese

**Optimizations**:
- Multi-image support via batched processing
- Automatic video frame extraction with ffmpeg
- Efficient Video Sampling (EVS) for reduced computation
- Clean output parsing with automatic response extraction

## Technical Details

### Image Requirements
- **Resolution**: 32×32 to 3072×1024 pixels
- **Format**: JPEG, PNG (RGB only)
- **Max Images**: 4 per inference
- **Tiling**: 12-tile layout (512×512 per tile)

### Video Requirements
- **Format**: MP4
- **Processing**: Automatic frame extraction via ffmpeg
- **FPS Range**: 1-30 (recommended: 1-2 for long videos, 2-5 for short clips)
- **Pruning**: Configurable via `video_pruning_rate` parameter

### System Requirements
- **GPU**: NVIDIA GPU with CUDA 12.2 support
- **VRAM**: Sufficient for 12.6B parameter model (~26GB recommended)
- **System Packages**: ffmpeg, libgl1-mesa-glx, libglib2.0-0
- **Python**: 3.11

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch==2.4.0` - PyTorch framework
- `transformers>=4.53,<4.54` - HuggingFace Transformers
- `mamba-ssm==2.2.5` - Mamba SSM for hybrid architecture
- `causal-conv1d==1.4.0` - Causal convolutions
- `timm>=0.9.0` - Vision model components
- `open_clip_torch>=2.20.0` - CLIP integration

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

`video_io.py` comes from [nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16), which is Apache 2.0.

The model weights are not covered by this license. [NVIDIA Nemotron Nano 12B v2 VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) is released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), which allows commercial use. `MODEL.md` is a pre-release model card; the current terms are on the Hugging Face model card.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All features are tested with `cog predict`
- Documentation is updated for new features

## Support

For issues with this Replicate/Cog implementation, please open a GitHub issue.

For questions about the base model, refer to [NVIDIA NGC](https://catalog.ngc.nvidia.com/).

## Acknowledgments

- Model developed by NVIDIA
- Cog framework by Replicate
- Implementation integrates NVIDIA's official model code from NGC
