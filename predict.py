# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Set HuggingFace to offline mode to use local files only
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/NVIDIA/Nemotron-Nano-v2-12B-VL/model.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print(f"Loading model from {MODEL_PATH}...")

        # Check if model exists locally
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please ensure the checkpoints folder contains all model files."
            )

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

    def predict(
        self,
        image: Path = Input(description="Input image for visual question answering"),
        prompt: str = Input(
            description="Text prompt or question about the image",
            default="Describe the image in detail."
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=1024,
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
        """Run a single prediction on the model"""

        # Load image
        input_image = Image.open(str(image)).convert("RGB")

        # Prepare messages following the model's chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Generate prompt using chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[formatted_prompt],
            images=[input_image],
            return_tensors="pt"
        ).to(self.device)

        # Determine sampling strategy
        do_sample = temperature > 0.0

        # Generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (skip special tokens for clean output)
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Extract only the assistant's response (remove input prompt)
        # The output includes the full conversation, so we extract the last response
        if "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant")[-1]
        elif "assistant\n" in output_text:
            output_text = output_text.split("assistant\n")[-1]

        return output_text.strip()
