# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import base64
from PIL import Image
from transformers.video_utils import VideoMetadata


def encode_pil_to_jpeg_data_url(pil_image):
    from io import BytesIO
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def frames_to_data_urls_with_metadata(frames, video_fps):
    """
    Convert a list of PIL Image frames to data URLs with metadata.
    
    Args:
        frames: List of PIL.Image objects (pre-extracted video frames)
        video_fps: The frame rate used when extracting these frames
    
    Returns:
        tuple: (frame_data_urls, metadata)
        - frame_data_urls: List of base64-encoded frame images
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of frames
            - fps: Frame rate of the frames
            - duration: Duration covered by the frames (in seconds)
            - video_backend: Backend used for video processing (None for pre-extracted frames)
    """
    if not frames:
        raise ValueError("frames list cannot be empty")
    
    # Convert frames to data URLs
    frame_urls = [encode_pil_to_jpeg_data_url(frame) for frame in frames]
    
    # Calculate metadata
    num_frames = len(frames)
    
    # Duration is calculated based on number of frames and fps
    if num_frames > 1 and video_fps > 0:
        # Duration = (num_frames - 1) / fps
        # The duration represents the time span from first to last frame
        sampled_duration = (num_frames - 1) / video_fps
        sampled_fps = video_fps
    else:
        # Single frame case or no fps provided
        sampled_duration = None
        sampled_fps = None
    
    metadata = VideoMetadata(
        total_num_frames=num_frames,
        fps=sampled_fps,
        duration=sampled_duration,
        video_backend=None,
    )
    
    return frame_urls, metadata


def load_frames_from_directory(frames_dir, sort_key=None):
    """
    Load frames from a directory of images.
    
    Args:
        frames_dir: Path to directory containing frame images
        sort_key: Optional function to sort frame filenames (default: natural sort by filename)
    
    Returns:
        List of PIL.Image objects
    """
    import glob
    
    # Support common image formats
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    frame_paths = []
    for pattern in patterns:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, pattern)))
    
    if not frame_paths:
        raise ValueError(f"No image frames found in directory: {frames_dir}")
    
    # Sort frames (by default, sort by filename)
    if sort_key is None:
        frame_paths.sort()
    else:
        frame_paths.sort(key=sort_key)
    
    # Load all frames
    frames = [Image.open(fp).convert('RGB') for fp in frame_paths]
    return frames
