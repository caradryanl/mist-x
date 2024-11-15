import argparse
import copy
import itertools
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
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
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
from huggingface_hub.utils import insecure_hashlib

logger = get_logger(__name__)

class PromptDataset:
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class PreprocessedDreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizers,
        text_encoders,
        vae,
        target_image_path=None,  # Added target image path
        class_data_root=None,
        class_prompt=None,
        size=1024,
        center_crop=False,
        device="cuda",
    ):
        self.size = size
        self.center_crop = center_crop
        self.device = device
        vae = vae.to(device)
        text_encoders = [text_encoder.to(device) for text_encoder in text_encoders]
        
        # Process instance images and convert to latents
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
            
        # Prepare image transforms
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Process instance images to latents
        logger.info("Processing instance images to latents...")
        instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
        self.instance_latents = []
        for image in tqdm(instance_images):
            processed_image = self._preprocess_image(image, train_resize, train_crop, train_transforms)
            with torch.no_grad():
                latent = vae.encode(processed_image.unsqueeze(0).to(device)).latent_dist.sample()
                self.instance_latents.append((latent * vae.config.scaling_factor).cpu())
        self.instance_latents = torch.cat(self.instance_latents)
        
        # Process instance prompt to embeddings
        logger.info("Processing instance prompt to embeddings...")
        self.instance_embeddings = self._get_text_embeddings(
            instance_prompt,
            tokenizers,
            text_encoders,
            device
        )
        
        # Process class data if provided
        self.class_latents = None
        self.class_embeddings = None
        if class_data_root is not None and class_prompt is not None:
            self.class_data_root = Path(class_data_root)
            if not self.class_data_root.exists():
                raise ValueError("Class images root doesn't exist.")
                
            logger.info("Processing class images to latents...")
            class_images = [Image.open(path) for path in list(Path(class_data_root).iterdir())]
            class_latents = []
            for image in tqdm(class_images):
                processed_image = self._preprocess_image(image, train_resize, train_crop, train_transforms)
                with torch.no_grad():
                    latent = vae.encode(processed_image.unsqueeze(0).to(device)).latent_dist.sample()
                    class_latents.append((latent * vae.config.scaling_factor).cpu())
            self.class_latents = torch.cat(class_latents)
            
            logger.info("Processing class prompt to embeddings...")
            self.class_embeddings = self._get_text_embeddings(
                class_prompt,
                tokenizers,
                text_encoders,
                device
            )
        
        self.target_latents = None
        if target_image_path is not None:
            logger.info("Processing target image to latents...")
            target_image = Image.open(target_image_path)
            processed_target = self._preprocess_image(target_image, train_resize, train_crop, train_transforms)
            with torch.no_grad():
                target_latent = vae.encode(processed_target.unsqueeze(0).to(device)).latent_dist.sample()
                self.target_latents = (target_latent * vae.config.scaling_factor).cpu()

        self.num_instance_images = len(self.instance_latents)
import argparse
import copy
import itertools
import logging
import math
import os
from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
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
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast

logger = get_logger(__name__)

class PromptDataset:
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {
            "prompt": self.prompt,
            "index": index
        }
        return example

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
            self.instance_pixel_values.append(self.transforms(image))
        self.instance_pixel_values = torch.stack(self.instance_pixel_values)
        
        # Handle class images if provided
        self.class_pixel_values = None
        if class_data_root is not None:
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

def generate_class_images(args, accelerator):
    """Generate class images using the base SD3 model if needed."""
    if not args.with_prior_preservation:
        return
        
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    
    cur_class_images = len(list(class_images_dir.iterdir()))
    
    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        pipeline.set_progress_bar_config(disable=True)
        
        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Generating {num_new_images} class images...")
        
        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=args.sample_batch_size
        )
        
        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)
        
        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images
            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def pgd_attack(
    args,
    pipeline,
    noise_scheduler,
    perturbed_images,
    original_images,
    target_images=None,
    num_steps=50,
    device="cuda",
):
    """
    Perform PGD attack on the pixel level while using the full SD3 pipeline.
    """
    # Move models to device and set to eval mode
    pipeline.to(device)
    pipeline.transformer.eval()
    pipeline.vae.eval()
    for encoder in pipeline.text_encoders:
        encoder.eval()

    # Initialize variables
    perturbed_images = perturbed_images.clone().detach().to(device)
    original_images = original_images.clone().detach().to(device)
    if target_images is not None:
        target_images = target_images.to(device)

    batch_size = args.train_batch_size
    num_images = len(perturbed_images)
    attacked_images = []

    # Process in batches
    for idx in tqdm(range(0, num_images, batch_size), desc="PGD attack"):
        batch_end = min(idx + batch_size, num_images)
        
        # Get current batch
        perturbed_batch = perturbed_images[idx:batch_end].clone()
        original_batch = original_images[idx:batch_end]
        target_batch = target_images[idx:batch_end] if target_images is not None else None
        
        for step in range(num_steps):
            perturbed_batch.requires_grad_(True)
            
            # Get latent representation
            latents = pipeline.vae.encode(perturbed_batch).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            # Encode text prompt
            prompt_embeds = pipeline._encode_prompt(
                args.instance_prompt,
                device,
                1,
                do_classifier_free_guidance=False
            )

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get model prediction
            model_pred = pipeline.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds[0],
                pooled_projections=prompt_embeds[1]
            ).sample

            # Calculate loss based on attack type
            if target_batch is not None:
                # Targeted attack: minimize distance to target latents
                target_latents = pipeline.vae.encode(target_batch).latent_dist.sample()
                target_latents = target_latents * pipeline.vae.config.scaling_factor
                loss = F.mse_loss(model_pred, target_latents)
            else:
                # Untargeted attack: maximize distance from original noise
                loss = -F.mse_loss(model_pred, noise)

            # Calculate gradients
            grad = torch.autograd.grad(loss, perturbed_batch)[0]

            # Update perturbed images with PGD step
            with torch.no_grad():
                # For targeted attack, move towards target
                if target_batch is not None:
                    perturbed_batch = perturbed_batch - args.pgd_alpha * grad.sign()
                else:
                    # For untargeted attack, move away from original
                    perturbed_batch = perturbed_batch + args.pgd_alpha * grad.sign()
                
                # Project back to epsilon ball and valid image range
                delta = perturbed_batch - original_batch
                delta = torch.clamp(delta, -args.pgd_eps, args.pgd_eps)
                perturbed_batch = torch.clamp(original_batch + delta, -1, 1)

        attacked_images.append(perturbed_batch.detach().cpu())

    # Combine all batches
    return torch.cat(attacked_images)

def train_one_step(
    transformer,
    text_encoders,
    vae,
    noise_scheduler,
    optimizer,
    pixel_values,
    prompt_embeds,
    args,
    device,
    is_instance=True
):
    """Train one step on either instance or class image."""
    # Get latents
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Get model prediction
    model_pred = transformer(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds[0],
        pooled_projections=prompt_embeds[1]
    ).sample

    # Calculate weighting for loss
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme="logit_normal",
        sigmas=noise_scheduler.sigmas[timesteps]
    )
    
    # Get loss
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - noise.float()) ** 2).reshape(latents.shape[0], -1),
        1,
    ).mean()

    # Apply prior preservation weight if it's a class image
    if not is_instance:
        loss = loss * args.prior_loss_weight

    # Backward pass
    loss.backward()
    
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.detach().item()

def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Generate class images if needed
    if args.with_prior_preservation:
        generate_class_images(args, accelerator)

    # Load pipeline and models
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    
    # Initialize LoRA
    transformer_lora_config = LoraConfig(
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
    pipeline.transformer.add_adapter(transformer_lora_config)

    # Create dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizers=pipeline.tokenizers,
        size=args.resolution,
        center_crop=args.center_crop,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt if args.with_prior_preservation else None,
    )

    # Get original images
    original_images = train_dataset.instance_pixel_values.clone()
    perturbed_images = original_images.clone()

    target_images = None
    if args.target_image_path:
        target_image = Image.open(args.target_image_path).convert("RGB")
        target_transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        target_images = target_transform(target_image).unsqueeze(0)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipeline.transformer.parameters(),
        lr=args.learning_rate,
    )

    # Outer loop for epochs
    for epoch in range(args.max_train_steps):
        logger.info(f"Starting epoch {epoch}")
        
        # 1. One step of adversarial attack
        perturbed_images = pgd_attack(
            args,
            pipeline,
            pipeline.scheduler,
            perturbed_images,
            original_images,
            target_images,
            num_steps=1,  # Only one PGD step per epoch
            device=accelerator.device
        )
        
        # 2. Multiple steps of LoRA training
        pipeline.transformer.train()
        pipeline.vae.eval()
        
        # Cache class prompt embeddings if using prior preservation
        if args.with_prior_preservation:
            class_prompt_embeds = pipeline._encode_prompt(
                args.class_prompt,
                accelerator.device,
                1,
                do_classifier_free_guidance=False
            )

        # Inner loop for multiple training steps per epoch
        for f_step in range(args.max_f_train_steps):
            total_loss = 0.0
            
            # Train on instance image
            instance_idx = f_step % len(perturbed_images)
            instance_image = perturbed_images[instance_idx:instance_idx+1].to(accelerator.device)
            instance_prompt_embeds = pipeline._encode_prompt(
                args.instance_prompt,
                accelerator.device,
                1,
                do_classifier_free_guidance=False
            )
            
            instance_loss = train_one_step(
                pipeline.transformer,
                pipeline.text_encoders,
                pipeline.vae,
                pipeline.scheduler,
                optimizer,
                instance_image,
                instance_prompt_embeds,
                args,
                accelerator.device,
                is_instance=True
            )
            total_loss += instance_loss

            # Train on class image if using prior preservation
            if args.with_prior_preservation:
                class_idx = f_step % len(train_dataset.class_pixel_values)
                class_image = train_dataset.class_pixel_values[class_idx:class_idx+1].to(accelerator.device)
                
                class_loss = train_one_step(
                    pipeline.transformer,
                    pipeline.text_encoders,
                    pipeline.vae,
                    pipeline.scheduler,
                    optimizer,
                    class_image,
                    class_prompt_embeds,
                    args,
                    accelerator.device,
                    is_instance=False
                )
                total_loss += class_loss

            logs = {
                "total_loss": total_loss,
                "instance_loss": instance_loss,
                "epoch": epoch,
                "f_step": f_step
            }
            if args.with_prior_preservation:
                logs["class_loss"] = class_loss
            
            logger.info(f"Epoch {epoch}, F-Step {f_step}: {logs}")

    # Save final results
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:        
        # Save final perturbed images
        for idx, perturbed_image in enumerate(perturbed_images):
            image = (perturbed_image * 0.5 + 0.5).clamp(0, 1)
            image = transforms.ToPILImage()(image)
            save_path = os.path.join(args.output_dir, f"perturbed_{idx}.png")
            image.save(save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial attack script for SD3.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/weights/stable-diffusion-3-medium",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        type=bool,
        default=True,
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=50,
        help="Minimal class images for prior preservation loss.",
    )
    parser.add_argument(
        "--target_image_path",
        type=str,
        default=None,
        help="Path to target image for targeted attack.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model and images will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=5,
        help="Total number of steps for adversarial training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after warmup) to use.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=2.0/255.0,
        help="Step size for PGD attack.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=4.0/255.0,
        help="Maximum perturbation size for PGD attack.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1453,
        help="A seed for reproducible training.",
    )
    
    args = parser.parse_args()
    
    # Sanity checks
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
            
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)