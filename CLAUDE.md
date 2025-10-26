# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Overview

This repository contains **NVIDIA Nemotron Nano VL 12B V2**, a vision-language model (VLM) for multi-modal document intelligence. The model supports:
- Single and multi-image inference (up to 4 images)
- Video understanding via pre-extracted frames
- 128K context length
- Document analysis, visual Q&A, and summarization
- Multiple languages (English, German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese)

**Architecture:**
- Vision Encoder: C-RADIOv2-H (located in `checkpoints/configuration_radio.py`)
- Language Model: NVIDIA-Nemotron-Nano-12B-v2 (H-architecture in `checkpoints/modeling_nemotron_h.py`)
- Main VLM implementation: `checkpoints/modeling.py`

## Installation

```bash
pip install causal_conv1d "transformers>4.53,<4.54" torch timm "mamba-ssm==2.2.5" accelerate open_clip_torch numpy pillow
```

## Testing the Model

### Single Image Inference

```bash
python checkpoints/quick_test_image.py --model_path /path/to/model --device cuda:0
```

This script tests both single and multi-image inference using example images from the `images/` directory.

### Video Inference

Video inference requires **pre-extracted frames**. First, extract frames using ffmpeg:

```bash
# Extract 1 frame per second
ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg
```

Then run the test script:

```bash
python checkpoints/quick_test_video.py \
    --model_path /path/to/model \
    --frames_dir frames \
    --video_fps 1 \
    --device cuda:0
```

**Important:** The model does not perform video decoding internally. You must extract frames beforehand.

## Key Components

### Core Model Files (in `checkpoints/`)

- `modeling.py`: Main VLM model (`NemotronH_Nano_VL_V2` class)
  - Combines vision encoder and language model
  - Handles image/video token injection and processing
  - Uses Efficient Video Sampling (EVS) for video understanding

- `processing.py`: Multi-modal processor (`NemotronNanoVLV2Processor`)
  - Processes text, images, and videos into model inputs
  - Handles special tokens (`<image>`, `<video>`, `<img>`, `</img>`)
  - Converts video frames to appropriate format with metadata

- `video_io.py`: Video frame handling utilities
  - `load_frames_from_directory()`: Load frames from disk
  - `frames_to_data_urls_with_metadata()`: Convert frames to data URLs with FPS metadata
  - No built-in video decoding - frames must be pre-extracted

- `configuration.py`: Model configuration (`NemotronH_Nano_VL_V2_Config`)
  - Defines vision and language model configs
  - Image tiling parameters (12-tile layout, 512×512 per tile)
  - Downsample ratio, projector settings

- `evs.py`: Efficient Video Sampling module
  - Optimizes video token processing
  - Controlled by `video_pruning_rate` parameter (default 0.75)

- `image_processing.py`: Image preprocessing and tiling logic

### Configuration Files

- `config.json`: Main model configuration
- `preprocessor_config.json`: Image preprocessing settings
- `tokenizer_config.json`: Tokenizer configuration with special tokens

## Message Format

The model uses a chat template with special system tokens:

```python
messages = [
    {"role": "system", "content": "/no_think"},  # Disables chain-of-thought reasoning
    {
        "role": "user",
        "content": [
            {"type": "image", "image": ""},  # or {"type": "video", "video": ""}
            {"type": "text", "text": "Your prompt here"},
        ],
    }
]
```

For multi-image:
```python
content = [
    {"type": "image", "image": "/path/to/image1"},
    {"type": "image", "image": "/path/to/image2"},
    {"type": "text", "text": "Your prompt"},
]
```

## Image Resolution Constraints

- Minimum: 32×32 pixels
- Maximum: 12-tile layout (each tile 512×512)
  - 4×3: up to 2048×1536
  - 3×4: up to 1536×2048
  - 2×6: up to 1024×3072
  - 6×2: up to 3072×1024
- 3-channel RGB only (no alpha)

## Video Processing

Video is processed as a sequence of frames with temporal metadata:

1. Extract frames using ffmpeg at desired FPS
2. Load frames using `video_io.load_frames_from_directory()`
3. Convert to data URLs with metadata using `video_io.frames_to_data_urls_with_metadata(frames, fps)`
4. Pass to processor with `videos_kwargs={'video_metadata': metadata}`
5. Set `model.video_pruning_rate` (0.0-1.0) to control efficiency vs. quality tradeoff

## Inference Parameters

**Common parameters:**
- `max_new_tokens`: Output length (1024 for images, 128 for video)
- `do_sample`: False for deterministic output
- `torch_dtype`: Use `torch.bfloat16` for efficiency
- `device_map`: Device placement (e.g., "cuda:0")

**Video-specific:**
- `video_pruning_rate`: 0.0-1.0 (default 0.75 for efficiency)
  - Higher values = more pruning = faster but potentially less detail
  - Lower values = less pruning = slower but more detail

## Model Outputs

The processor's `batch_decode()` returns the full generation including special tokens when `skip_special_tokens=False`. For production use, consider setting `skip_special_tokens=True` to get clean output.

## Hardware Requirements

- Designed for NVIDIA GPUs (A100, H100/H200, B200)
- Inference engines: vLLM, TensorRT-LLM
- OS: Linux

## License

NVIDIA Open Model License - see MODEL.md for full details.
