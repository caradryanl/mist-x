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
from math import ceil

def get_images_from_path(path: str, resolution=1024) -> list[Image.Image]:
    """
    Get images under the path with default resolution for SD3
    """
    for root, dirs, files in os.walk(path):
        images = []
        try:
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
        except:
            print("Warning: files are not sorted in numerical order")
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                images.append(Image.open(os.path.join(root, file)).convert('RGB').resize((resolution, resolution), Image.Resampling.LANCZOS))
        return images

def get_model(model_id: str, lora_path: str = None) -> StableDiffusion3Pipeline:
    """
    Load SD3 model with optional LoRA weights
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA if specified
    if lora_path:
        pipe.load_lora_weights(lora_path)
    
    return pipe

def get_i2i_model(model_id: str, lora_path: str = None) -> StableDiffusion3Img2ImgPipeline:
    """
    Load SD3 image-to-image model with optional LoRA weights
    """
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if lora_path:
        pipe.load_lora_weights(lora_path)
    
    return pipe

def t2i(
    pipe: StableDiffusion3Pipeline, 
    prompt: str,
    prompt_2: str = None,
    prompt_3: str = None,
    negative_prompt: str = None,
    negative_prompt_2: str = None,
    negative_prompt_3: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    samples: int = 1
) -> list[Image.Image]:
    """
    Text-to-image generation with SD3
    """
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=samples,
            output_type="pil"
        )
    return output.images

def i2i(
    pipe: StableDiffusion3Img2ImgPipeline,
    images: list[Image.Image],
    prompt: str,
    prompt_2: str = None,
    prompt_3: str = None,
    negative_prompt: str = None,
    negative_prompt_2: str = None,
    negative_prompt_3: str = None,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    samples_per_image: int = 1
) -> list[Image.Image]:
    """
    Image-to-image generation with SD3
    """
    all_output_images = []
    
    for image in images:
        output = pipe(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            strength=strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=samples_per_image,
            output_type="pil"
        )
        all_output_images.extend(output.images)
    
    return all_output_images

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/weights/sd3_medium_incl_clips_t5xxlfp16.safetensors",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--prompt", '-p', type=str, default="")
    parser.add_argument("--prompt_2", type=str, default=None, help="Second prompt for CLIP2")
    parser.add_argument("--prompt_3", type=str, default=None, help="Third prompt for T5") 
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt_2", type=str, default=None)
    parser.add_argument("--negative_prompt_3", type=str, default=None)
    parser.add_argument("--strength", '-s', type=float, default=0.6)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_per_img", '-spi', type=int, default=1)
    parser.add_argument("--lora_path", '-lp', type=str, default=None)
    parser.add_argument("--input_path", '-ip', type=str, default="data/input")
    parser.add_argument("--output_path", '-op', type=str, default="data/output")
    parser.add_argument(
        '--mode',
        '-m',
        type=str,
        default='t2i',
        choices=['i2i', 't2i'],
        help='mode of the pipeline'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default=False,
        help='whether to output information'
    )
    parser.add_argument(
        '--resolution',
        '-r',
        type=int,
        default=1024,
        help='resolution of the image'
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output path {args.output_path}")
    return args

@torch.no_grad()
def I2IPipeline(args: argparse.Namespace):
    pipe = get_i2i_model(args.pretrained_model_name_or_path, args.lora_path)
    if args.verbose is False:
        pipe.set_progress_bar_config(disable=True)
    
    print("Loading images")
    images = get_images_from_path(args.input_path, args.resolution)
    
    output_images = i2i(
        pipe=pipe,
        images=images,
        prompt=args.prompt,
        prompt_2=args.prompt_2,
        prompt_3=args.prompt_3,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt_2,
        negative_prompt_3=args.negative_prompt_3,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        samples_per_image=args.sample_per_img
    )
    
    for i, img in enumerate(output_images):
        img.save(f"{args.output_path}/{i}.png")

@torch.no_grad()
def T2IPipeline(args: argparse.Namespace):
    pipe = get_model(args.pretrained_model_name_or_path, args.lora_path)
    if args.verbose is False:
        pipe.set_progress_bar_config(disable=True)
    
    images = t2i(
        pipe=pipe,
        prompt=args.prompt,
        prompt_2=args.prompt_2,
        prompt_3=args.prompt_3,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt_2,
        negative_prompt_3=args.negative_prompt_3,
        guidance_scale=args.guidance_scale,
        samples=args.sample_per_img
    )
    
    for i, img in enumerate(images):
        img.save(f"{args.output_path}/{i}.png")

if __name__ == "__main__":
    args = parseargs()
    if args.mode == 'i2i':
        I2IPipeline(args)
    elif args.mode == 't2i':
        T2IPipeline(args)