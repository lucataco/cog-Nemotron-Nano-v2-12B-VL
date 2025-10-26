# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import re
import sys
import time
import torch
import shutil
import video_io
import tempfile
import subprocess
from typing import List, Optional, Tuple
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Set HuggingFace to offline mode to use local files only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/NVIDIA/Nemotron-Nano-v2-12B-VL/model.tar"
RADIO_PATH = "./C_RADIOv2_H"
RADIO_URL = "https://weights.replicate.delivery/default/NVIDIA/Nemotron-Nano-v2-12B-VL/radio.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print(f"Loading model from {MODEL_PATH}...")

        # Check if model exists locally
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)
        if not os.path.exists(RADIO_PATH):
            download_weights(RADIO_URL, RADIO_PATH)

        # Load model with local files only
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model components
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("Model loaded successfully!")

    def _extract_video_frames(self, video_path: Path, fps: int) -> Tuple[str, int]:
        """Extract frames from video using ffmpeg.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract

        Returns:
            Tuple of (temp_dir_path, actual_frame_count)
        """
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")
        output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality JPEG
            output_pattern
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract video frames: {e.stderr}")

        # Count extracted frames
        import glob
        frame_files = glob.glob(os.path.join(temp_dir, "frame_*.jpg"))

        if not frame_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("No frames were extracted from video")

        return temp_dir, len(frame_files)

    def _process_media_input(
        self,
        images: Optional[list] = None,
        video: Optional[Path] = None,
        video_fps: int = 1,
        prompt: str = "Describe what you see.",
        system_prompt: str = "/no_think"
    ) -> dict:
        """Process image(s) or video input into model-ready format.

        Args:
            images: List of Path objects to image files (mutually exclusive with video)
            video: Path to video file (mutually exclusive with images)
            video_fps: FPS for video frame extraction
            prompt: Text prompt
            system_prompt: System prompt

        Returns:
            dict: Processor kwargs ready to pass to self.processor()
        """
        temp_dir = None

        try:
            if images is not None:
                # Process single or multiple images
                pil_images = [Image.open(str(img)).convert("RGB") for img in images]

                # Create message content with multiple image placeholders
                content = [{"type": "image", "image": f"/path/to/image{i+1}"} for i in range(len(pil_images))]
                content.append({"type": "text", "text": prompt})

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                return {
                    "text": [formatted_prompt],
                    "images": pil_images,
                    "return_tensors": "pt"
                }

            else:  # video is not None
                # Extract frames from video
                temp_dir, frame_count = self._extract_video_frames(video, video_fps)
                print(f"Extracted {frame_count} frames from video at {video_fps} fps")

                # Load frames
                frames = video_io.load_frames_from_directory(temp_dir)

                # Get metadata
                _, metadata = video_io.frames_to_data_urls_with_metadata(frames, video_fps)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": ""},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                return {
                    "text": [formatted_prompt],
                    "videos": frames,
                    "videos_kwargs": {"video_metadata": metadata},
                    "return_tensors": "pt"
                }

        finally:
            # Clean up temp directory if it was created
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt or question about the media",
            default="Describe what you see in detail."
        ),
        images: List[Path] = Input(
            description="List of input images (1-4 images supported, provide either images or video)",
            default=None
        ),
        video: Optional[Path] = Input(
            description="Input video file (provide either images or video, not both)",
            default=None
        ),
        video_fps: int = Input(
            description="Frames per second to extract from video (only used for video input)",
            default=1,
            ge=1,
            le=30
        ),
        video_pruning_rate: float = Input(
            description="Video pruning rate for efficiency (0.0=no pruning, 1.0=max pruning, only used for video)",
            default=0.75,
            ge=0.0,
            le=1.0
        ),
        max_new_tokens: Optional[int] = Input(
            description="Maximum number of tokens to generate (default: 1024 for images, 512 for videos)",
            default=None,
            ge=1,
            le=2048
        ),
        temperature: float = Input(
            description="Sampling temperature (0 for greedy decoding)",
            default=0.0,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Nucleus sampling top-p",
            default=1.0,
            ge=0.0,
            le=1.0
        ),
        system_prompt: str = Input(
            description="System prompt (/no_think disables chain-of-thought reasoning)",
            default="/no_think"
        ),
    ) -> str:
        """Run a single prediction on the model with image(s) or video input"""

        # Validate inputs
        if not images and video is None:
            raise ValueError("Must provide at least one image or a video input")
        if images and video is not None:
            raise ValueError("Cannot provide both images and video inputs. Choose one.")
        if images and len(images) > 4:
            raise ValueError("Maximum of 4 images supported")
        if images and len(images) < 1:
            raise ValueError("Must provide at least one image")

        # Determine if processing video
        is_video = video is not None

        # Set default max_new_tokens based on input type
        if max_new_tokens is None:
            max_new_tokens = 512 if is_video else 1024

        # Process media input and get processor kwargs
        processor_kwargs = self._process_media_input(
            images=images if images else None,
            video=video,
            video_fps=video_fps,
            prompt=prompt,
            system_prompt=system_prompt
        )

        # Process inputs through processor
        inputs = self.processor(**processor_kwargs).to(self.device)

        # Set video pruning rate for video inputs
        if is_video:
            self.model.video_pruning_rate = video_pruning_rate

        # Determine sampling strategy
        do_sample = temperature > 0.0

        # Prepare generate kwargs
        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add pixel_values for image or video
        if is_video:
            generate_kwargs["pixel_values_videos"] = inputs.pixel_values_videos
        else:
            generate_kwargs["pixel_values"] = inputs.pixel_values

        # Add sampling parameters if temperature > 0
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        # Generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(**generate_kwargs)

        # Decode output (skip special tokens for clean output)
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Extract only the assistant's response (remove input prompt and conversation structure)
        # Try different patterns to extract the assistant's response
        if "Assistant\n" in output_text:
            output_text = output_text.split("Assistant\n", 1)[-1]
        elif "assistant\n" in output_text:
            output_text = output_text.split("assistant\n", 1)[-1]
        elif "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant", 1)[-1]

        # Remove thinking tags if chain-of-thought reasoning is disabled
        if system_prompt == "/no_think":
            output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)

        return output_text.strip()
