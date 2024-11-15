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
        self.num_class_images = len(self.class_latents) if self.class_latents is not None else 0
        self._length = max(self.num_instance_images, self.num_class_images)

    def _preprocess_image(self, image, resize, crop, transforms):
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = resize(image)
        return transforms(image)

    def _get_text_embeddings(self, prompt, tokenizers, text_encoders, device):
        tokens_one = tokenizers[0](
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        
        tokens_two = tokenizers[1](
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        
        tokens_three = tokenizers[2](
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            prompt_embeds_one = text_encoders[0](tokens_one, output_hidden_states=True)
            prompt_embeds_two = text_encoders[1](tokens_two, output_hidden_states=True)
            prompt_embeds_three = text_encoders[2](tokens_three)[0]
            
            prompt_embeds = torch.cat([
                prompt_embeds_one.hidden_states[-2],
                prompt_embeds_two.hidden_states[-2],
                prompt_embeds_three
            ], dim=-2)
            
            pooled_prompt_embeds = torch.cat([
                prompt_embeds_one[0],
                prompt_embeds_two[0]
            ], dim=-1)

        return {
            'prompt_embeds': prompt_embeds.cpu(),
            'pooled_prompt_embeds': pooled_prompt_embeds.cpu()
        }

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {
            "instance_latents": self.instance_latents[index % self.num_instance_images],
            "instance_embeddings": {
                k: v.clone() for k, v in self.instance_embeddings.items()
            },
            "is_instance": True,
        }
        
        if self.num_class_images > 0:
            example["class_latents"] = self.class_latents[index % self.num_class_images]
            example["class_embeddings"] = {
                k: v.clone() for k, v in self.class_embeddings.items()
            }
            example["is_instance"] = False
            
        return example

def generate_class_images(args, accelerator):
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16

            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, 
                desc="Generating class images",
                disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def optimized_pgd_attack(
    args,
    transformer,
    noise_scheduler,
    data_latents,
    embeddings,
    original_latents,
    target_latents,
    num_steps,
    device
):
    transformer.to(device)
    transformer.requires_grad_(False)

    data_latents = data_latents.detach().clone()
    num_latents = len(data_latents)
    latent_list = []
    batch_size = args.train_batch_size

    prompt_embeds = embeddings['prompt_embeds'].to(device)
    pooled_prompt_embeds = embeddings['pooled_prompt_embeds'].to(device)

    # Move target latents to device if provided
    if target_latents is not None:
        target_latents = target_latents.to(device)

    for k in tqdm(range(num_latents // batch_size), desc="PGD attack"):
        id = k * batch_size
        end_id = min((k + 1) * batch_size, num_latents)
        perturbed_latents = data_latents[id:end_id, :].to(device).detach().clone()
        perturbed_latents.requires_grad = True
        original_latent = original_latents[id:end_id, :].to(device)

        for step in range(num_steps):
            perturbed_latents.requires_grad = True
            
            # Sample noise and timesteps
            noise = torch.randn_like(perturbed_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (perturbed_latents.shape[0],), device=device)
            
            # Add noise
            noisy_latents = noise_scheduler.add_noise(perturbed_latents, noise, timesteps)

            # Predict noise
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            if target_latents is not None:
                # For targeted attack: minimize distance to target latents
                loss = F.mse_loss(model_pred, target_latents)
            else:
                # For untargeted attack: maximize distance from original noise
                loss = F.mse_loss(model_pred, noise)
            
            loss.backward()

            if step % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                # For targeted attack: move towards target
                if target_latents is not None:
                    adv_latents = perturbed_latents - args.pgd_alpha * perturbed_latents.grad.sign()
                else:
                    # For untargeted attack: move away from original
                    adv_latents = perturbed_latents + args.pgd_alpha * perturbed_latents.grad.sign()
                
                eta = torch.clamp(adv_latents - original_latent, min=-args.pgd_eps, max=args.pgd_eps)
                perturbed_latents = torch.clamp(original_latent + eta, min=-1, max=1).detach_()
                perturbed_latents.requires_grad = True

        latent_list.extend([latent.detach().cpu() for latent in perturbed_latents])

    return torch.stack(latent_list)


def process_batch_latents(
    latents,
    embeddings,
    transformer,
    noise_scheduler,
    args,
    device
):
    # Move data to device
    latents = latents.to(device)
    prompt_embeds = embeddings['prompt_embeds'].to(device)
    pooled_prompt_embeds = embeddings['pooled_prompt_embeds'].to(device)

    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
    )

    # Add noise
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get model prediction
    model_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]

    # Calculate loss weighting
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme="logit_normal",
        sigmas=noise_scheduler.sigmas[timesteps]
    )

    # Calculate loss
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - noise.float()) ** 2).reshape(noise.shape[0], -1),
        1,
    ).mean()

    return loss

def main(args):
    logging_dir = Path(args.output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # First load the models needed for class image generation
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    
    logger.info("Generating class images if needed...")
    generate_class_images(args, accelerator)

    # Now load all models for training
    logger.info("Loading models and tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")

    text_encoder_one = transformers.CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder_three = transformers.T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3"
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Create preprocessed dataset with all images (including generated class images)
    logger.info("Creating preprocessed dataset...")
    train_dataset = PreprocessedDreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizers=[tokenizer_one, tokenizer_two, tokenizer_three],
        text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
        vae=vae,
        target_image_path=args.target_image_path,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt if args.with_prior_preservation else None,
        size=args.resolution,
        center_crop=args.center_crop,
        device=accelerator.device,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Configure LoRA
    transformer_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with accelerator
    transformer, text_encoder_one, text_encoder_two, text_encoder_three, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, text_encoder_one, text_encoder_two, text_encoder_three, optimizer, train_dataloader, lr_scheduler
    )

    # Get the original instance and class data
    instance_data = torch.stack([example["instance_images"] for example in train_dataset if "instance_images" in example])
    perturbed_instance_data = instance_data.clone()

    
    class_data = None
    if args.class_data_dir:
        class_data = torch.stack([example["class_images"] for example in train_dataset if "class_images" in example])

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # Training loop setup
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    # Training loop modifications
    for epoch in range(args.num_train_epochs):
        transformer.train()

        # Perform PGD attack on instance latents with target
        logger.info("Performing PGD attack on instance latents...")
        perturbed_instance_latents = optimized_pgd_attack(
            args=args,
            transformer=transformer,
            noise_scheduler=noise_scheduler,
            data_latents=train_dataset.instance_latents,
            embeddings=train_dataset.instance_embeddings,
            original_latents=train_dataset.instance_latents,
            target_latents=train_dataset.target_latents,  # Now properly passing target latents
            num_steps=args.max_adv_train_steps,
            device=accelerator.device
        )

        # Training loop with preprocessed latents and embeddings
        logger.info(f"Training model for {args.max_f_train_steps} steps...")
        for f_step in range(args.max_f_train_steps):
            # Randomly select one instance latent
            instance_idx = torch.randint(0, len(perturbed_instance_latents), (1,)).item()
            instance_latent = perturbed_instance_latents[instance_idx:instance_idx+1]
            instance_embeddings = train_dataset.instance_embeddings
            
            # Randomly select one class latent if available
            class_latent = None
            class_embeddings = None
            if train_dataset.class_latents is not None:
                class_idx = torch.randint(0, len(train_dataset.class_latents), (1,)).item()
                class_latent = train_dataset.class_latents[class_idx:class_idx+1]
                class_embeddings = train_dataset.class_embeddings

            # First step: train on single instance latent
            optimizer.zero_grad()
            with accelerator.accumulate(transformer):
                instance_loss = process_batch_latents(
                    instance_latent,
                    instance_embeddings,
                    transformer,
                    noise_scheduler,
                    args,
                    accelerator.device
                )
                instance_loss = instance_loss / args.gradient_accumulation_steps
                accelerator.backward(instance_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Second step: train on single class latent if available
            if class_latent is not None:
                with accelerator.accumulate(transformer):
                    class_loss = process_batch_latents(
                        class_latent,
                        class_embeddings,
                        transformer,
                        noise_scheduler,
                        args,
                        accelerator.device
                    )
                    class_loss = (class_loss * args.prior_preservation_weight) / args.gradient_accumulation_steps
                    accelerator.backward(class_loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            # Logging
            if global_step % 100 == 0:
                total_loss = instance_loss.detach().item()
                if class_latent is not None:
                    total_loss += class_loss.detach().item()
                logger.info(f"Step {global_step}: loss: {total_loss}")

            logs = {
                "instance_loss": instance_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            if class_latent is not None:
                logs["class_loss"] = class_loss.detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save perturbed instance images at the end of the last epoch
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode latents back to images for saving
        vae.to(accelerator.device)
        with torch.no_grad():
            for idx, latent in enumerate(perturbed_instance_latents):
                # Scale and decode the image latents with vae
                latent = 1 / vae.config.scaling_factor * latent
                image = vae.decode(latent.unsqueeze(0).to(accelerator.device)).sample
                
                # Convert to PIL image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8"))
                image.save(
                    os.path.join(args.output_dir, f"adversarial_instance_{idx}.png")
                )

        accelerator.wait_for_everyone()

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized adversarial attack script for SD3.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/weights/stable-diffusion-3-medium-diffusers/",
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
        help="A folder containing the class images for prior preservation.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=50,
        help="Number of class images to generate if class_data_dir is empty.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        type=bool,
        default=True,
        help="Flag to enable prior preservation.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size for class image generation.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--target_image_path",
        type=str,
        default="data/targets/NIPS.png",
        help="Path to target image for targeted attack.",
    )
    
    # outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/outputs",
        help="The output directory where the perturbed images will be written.",
    )

    # basic setup
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
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=5,
        help="Total number of steps to train surrogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=50,
        help="Total number of steps for adversarial training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after warmup) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=5e-3,
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
        "--prior_preservation_weight",
        type=float,
        default=1.0,
        help="Weight for prior preservation loss.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--precompute_text_embeddings",
        action="store_true",
        help="Precompute and cache text embeddings to save memory",
    )
    parser.add_argument(
        "--precompute_latents",
        action="store_true",
        help="Precompute and cache image latents to save memory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to store precomputed embeddings and latents",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)