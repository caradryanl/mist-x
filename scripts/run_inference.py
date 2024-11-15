from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline, 
    FlowMatchEulerDiscreteScheduler,
)
import torch
from PIL import Image
from pathlib import Path
import argparse
import os
from tqdm import tqdm
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_images_from_path(path: str, resolution: int = 1024) -> List[Image.Image]:
    """
    Get images under the path with default resolution for SD3
    
    Args:
        path (str): Path to the directory containing images
        resolution (int, optional): Target resolution for the images. Defaults to 1024.
        
    Returns:
        List[Image.Image]: List of loaded and processed PIL images
    """
    images = []
    if not os.path.exists(path):
        raise ValueError(f"Input path {path} does not exist")
        
    for root, _, files in os.walk(path):
        try:
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
        except (ValueError, KeyError):
            logger.warning("Files could not be sorted numerically, using default sorting")
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(os.path.join(root, file)).convert('RGB')
                    # Preserve aspect ratio while resizing
                    ratio = min(resolution / image.width, resolution / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Could not load image {file}: {str(e)}")
                    
    if not images:
        raise ValueError(f"No valid images found in {path}")
    
    return images

def load_model_with_lora(
    pipeline_class: Union[StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline],
    model_id: str,
    lora_path: Optional[str] = None,
    **kwargs
) -> Union[StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline]:
    """
    Load SD3 model with optional LoRA weights
    
    Args:
        pipeline_class: The pipeline class to instantiate
        model_id (str): Path or HuggingFace Hub ID of the base model
        lora_path (str, optional): Path to LoRA weights
        **kwargs: Additional arguments passed to from_pretrained()
        
    Returns:
        The loaded pipeline instance
    """
    try:
        pipe = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            **kwargs
        ).to("cuda")
        
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        if lora_path:
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA path {lora_path} does not exist")
            pipe.load_lora_weights(lora_path)
            logger.info(f"Loaded LoRA weights from {lora_path}")
            
        return pipe
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_images(
    pipe: Union[StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline],
    prompt: str,
    mode: str = "t2i",
    input_image: Optional[Image.Image] = None,
    prompt_2: Optional[str] = None,
    prompt_3: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    negative_prompt_3: Optional[str] = None,
    num_inference_steps: int = 50,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    num_images: int = 1,
    joint_attention_kwargs: Optional[dict] = None
) -> List[Image.Image]:
    """
    Generate images using either text-to-image or image-to-image mode
    
    Args:
        pipe: The pipeline instance to use
        prompt (str): Main prompt
        mode (str): Either "t2i" or "i2i"
        input_image (Optional[Image.Image]): Input image for i2i mode
        **kwargs: Additional generation parameters
        
    Returns:
        List[Image.Image]: Generated images
    """
    generation_kwargs = {
        "prompt": prompt,
        "prompt_2": prompt_2,
        "prompt_3": prompt_3,
        "negative_prompt": negative_prompt,
        "negative_prompt_2": negative_prompt_2,
        "negative_prompt_3": negative_prompt_3,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images,
        "output_type": "pil",
        "joint_attention_kwargs": joint_attention_kwargs
    }
    
    try:
        with torch.no_grad():
            if mode == "i2i":
                if input_image is None:
                    raise ValueError("Input image required for image-to-image generation")
                generation_kwargs["image"] = input_image
                generation_kwargs["strength"] = strength
                
            output = pipe(**generation_kwargs)
            return output.images
            
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with Stable Diffusion 3")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/weights/stable-diffusion-3-medium-diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Main prompt")
    parser.add_argument("--prompt_2", type=str, help="Second prompt for CLIP2")
    parser.add_argument("--prompt_3", type=str, help="Third prompt for T5")
    parser.add_argument("--negative_prompt", type=str, help="Main negative prompt")
    parser.add_argument("--negative_prompt_2", type=str, help="Second negative prompt")
    parser.add_argument("--negative_prompt_3", type=str, help="Third negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--strength", "-s", type=float, default=0.6)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--samples_per_image", "-spi", type=int, default=1)
    parser.add_argument("--lora_path", "-lp", type=str, help="Path to LoRA weights")
    parser.add_argument("--input_path", "-ip", type=str, help="Path to input images for i2i")
    parser.add_argument("--output_path", "-op", type=str, required=True)
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="t2i",
        choices=["i2i", "t2i"],
        help="Pipeline mode"
    )
    parser.add_argument("--resolution", "-r", type=int, default=1024)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "i2i" and not args.input_path:
        parser.error("--input_path is required for image-to-image mode")
        
    return args

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load appropriate pipeline
    pipeline_class = StableDiffusion3Img2ImgPipeline if args.mode == "i2i" else StableDiffusion3Pipeline
    pipe = load_model_with_lora(pipeline_class, args.pretrained_model_name_or_path, args.lora_path)
    
    if not args.verbose:
        pipe.set_progress_bar_config(disable=True)
    
    if args.mode == "i2i":
        logger.info("Loading input images...")
        input_images = get_images_from_path(args.input_path, args.resolution)
        
        for idx, image in enumerate(tqdm(input_images, desc="Processing images")):
            outputs = generate_images(
                pipe=pipe,
                mode="i2i",
                input_image=image,
                prompt=args.prompt,
                prompt_2=args.prompt_2,
                prompt_3=args.prompt_3,
                negative_prompt=args.negative_prompt,
                negative_prompt_2=args.negative_prompt_2,
                negative_prompt_3=args.negative_prompt_3,
                num_inference_steps=args.num_inference_steps,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_images=args.samples_per_image
            )
            
            for sample_idx, output_image in enumerate(outputs):
                output_path = output_dir / f"{idx}_{sample_idx}.png"
                output_image.save(output_path)
                
    else:  # t2i mode
        outputs = generate_images(
            pipe=pipe,
            mode="t2i",
            prompt=args.prompt,
            prompt_2=args.prompt_2,
            prompt_3=args.prompt_3,
            negative_prompt=args.negative_prompt,
            negative_prompt_2=args.negative_prompt_2,
            negative_prompt_3=args.negative_prompt_3,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images=args.samples_per_image
        )
        
        for idx, image in enumerate(outputs):
            output_path = output_dir / f"{idx}.png"
            image.save(output_path)
    
    logger.info(f"Generated images saved to {args.output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise