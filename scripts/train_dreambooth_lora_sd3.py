import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from peft import LoraConfig
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig

logger = get_logger(__name__)

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    """
    Encodes the prompt with CLIP text encoder.
    
    Args:
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        prompt: The input prompt to encode
        device: Device to put the embeddings on
        text_input_ids: Optional pre-computed input ids
        num_images_per_prompt: Number of images per prompt
    
    Returns:
        tuple(torch.Tensor, torch.Tensor): prompt_embeds, pooled_prompt_embeds
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    """
    Encodes the prompt with T5 text encoder.
    
    Args:
        text_encoder: T5 text encoder
        tokenizer: T5 tokenizer
        max_sequence_length: Maximum sequence length for tokenization
        prompt: The input prompt to encode
        num_images_per_prompt: Number of images per prompt
        device: Device to put the embeddings on
        text_input_ids: Optional pre-computed input ids
    
    Returns:
        torch.Tensor: prompt_embeds
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    """
    Encodes the prompt using multiple text encoders (CLIP and T5) and combines their embeddings.
    
    Args:
        text_encoders: List of text encoders (CLIP and T5)
        tokenizers: List of tokenizers (CLIP and T5)
        prompt: The input prompt to encode
        max_sequence_length: Maximum sequence length for T5 tokenization
        device: Device to put the embeddings on
        num_images_per_prompt: Number of images per prompt
        text_input_ids_list: Optional list of pre-computed input ids
    
    Returns:
        tuple(torch.Tensor, torch.Tensor): Combined prompt embeddings and pooled embeddings
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    # Pad CLIP embeddings to match T5 embedding dimension
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    
    # Combine CLIP and T5 embeddings
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizers,
        size=1024,
        center_crop=False,
        class_data_root=None,
        class_prompt=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        
        # Load instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
            
        self.instance_images = []
        for path in self.instance_data_root.iterdir():
            image = Image.open(path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.instance_images.append(image)

        # Prepare transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Process instance images
        self.instance_pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            self.instance_pixel_values.append(self.transforms(image))
        self.instance_pixel_values = torch.stack(self.instance_pixel_values)
        
        # Handle class images if provided
        self.class_pixel_values = None
        if class_data_root:
            class_path = Path(class_data_root)
            if not class_path.exists():
                raise ValueError("Class data root doesn't exist.")
            
            class_images = []
            for path in class_path.iterdir():
                image = Image.open(path)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                class_images.append(self.transforms(image))
            self.class_pixel_values = torch.stack(class_images)

        self.num_instance_images = len(self.instance_pixel_values)
        self.num_class_images = len(self.class_pixel_values) if self.class_pixel_values is not None else 0
        self._length = max(self.num_instance_images, self.num_class_images)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {
            "instance_pixel_values": self.instance_pixel_values[index % self.num_instance_images],
            "instance_prompt": self.instance_prompt,
        }
        
        if self.class_pixel_values is not None:
            example["class_pixel_values"] = self.class_pixel_values[index % self.num_class_images]
            example["class_prompt"] = self.class_prompt
            
        return example
    
def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizers and model
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )

    # Initialize LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
        ],
    )
    pipeline.transformer.add_adapter(lora_config)

    # Create dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizers=[tokenizer_one, tokenizer_two, tokenizer_three],
        size=args.resolution,
        center_crop=args.center_crop,
        class_data_root=args.class_data_dir,
        class_prompt=args.class_prompt,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        pipeline.transformer.parameters(),
        lr=args.learning_rate,
    )

    # Prepare text encoders and encode prompts
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device)

    # Encode prompts
    instance_prompt_embeds = encode_prompt(
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        prompt=args.instance_prompt,
        max_sequence_length=77,
        device=accelerator.device,
        num_images_per_prompt=1
    )

    if args.class_prompt:
        class_prompt_embeds = encode_prompt(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            prompt=args.class_prompt,
            max_sequence_length=77,
            device=accelerator.device,
            num_images_per_prompt=1
        )

    # Move text encoders back to CPU
    for text_encoder in text_encoders:
        text_encoder.to('cpu')

    # Training loop
    pipeline.transformer.train()
    pipeline.vae.eval()

    for step in range(args.max_train_steps):
        total_loss = 0.0
        
        # Train on instance images
        for idx in range(len(train_dataset.instance_pixel_values)):
            instance_pixel_values = train_dataset.instance_pixel_values[idx:idx+1].to(accelerator.device)
            
            # Get latents
            latents = pipeline.vae.encode(instance_pixel_values).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=accelerator.device)
            
            # Get sigmas
            sigmas = pipeline.scheduler.sigmas[timesteps].flatten()
            while len(sigmas.shape) < len(latents.shape):
                sigmas = sigmas.unsqueeze(-1)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

            # Get model prediction
            model_pred = pipeline.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=instance_prompt_embeds[0],
                pooled_projections=instance_prompt_embeds[1],
                return_dict=False,
            )[0]

            # Calculate loss
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme="logit_normal",
                sigmas=pipeline.scheduler.sigmas[timesteps]
            )
            
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - noise.float()) ** 2).reshape(latents.shape[0], -1),
                1,
            ).mean()

            # Backward pass
            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(pipeline.transformer.parameters(), args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().item()

        # Log progress
        logs = {"loss": total_loss / len(train_dataset), "step": step}
        logger.info(f"Step {step}: {logs}")

    # Save the trained model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline.transformer.save_pretrained(args.output_dir)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3-base",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)