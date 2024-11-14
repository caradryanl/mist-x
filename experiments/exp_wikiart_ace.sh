#!/bin/bash

# First phase: Generate adversarial perturbations
accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/wikiart/Fauvism_henri-matisse \
    --output_dir data/outputs/wikiart/ace/Fauvism_henri-matisse \
    --instance_prompt "a photo of a sks painting" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/wikiart/Impressionism_claude-monet \
    --output_dir data/outputs/wikiart/ace/Impressionism_claude-monet \
    --instance_prompt "a photo of a sks painting" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/wikiart/Pointillism_paul-signac \
    --output_dir data/outputs/wikiart/ace/Pointillism_paul-signac \
    --instance_prompt "a photo of a sks painting" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/wikiart/Post_Impressionism-van-gogh \
    --output_dir data/outputs/wikiart/ace/Post_Impressionism-van-gogh \
    --instance_prompt "a photo of a sks painting" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/wikiart/Rococo_canaletto \
    --output_dir data/outputs/wikiart/ace/Rococo_canaletto \
    --instance_prompt "a photo of a sks painting" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

# Second phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/wikiart/ace/Fauvism_henri-matisse \
    --output_dir models/loras/wikiart/ace/Fauvism_henri-matisse \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/wikiart/ace/Impressionism_claude-monet \
    --output_dir models/loras/wikiart/ace/Impressionism_claude-monet \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/wikiart/ace/Pointillism_paul-signac \
    --output_dir models/loras/wikiart/ace/Pointillism_paul-signac \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/wikiart/ace/Post_Impressionism-van-gogh \
    --output_dir models/loras/wikiart/ace/Post_Impressionism-van-gogh \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/wikiart/ace/Rococo_canaletto \
    --output_dir models/loras/wikiart/ace/Rococo_canaletto \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

# Third phase: Run evaluation
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/ace/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/ace/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/ace/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/ace/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/ace/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/wikiart/ace/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/wikiart/ace/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/wikiart/ace/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/wikiart/ace/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/wikiart/ace/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/wikiart/ace/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/wikiart/ace/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/wikiart/ace/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/wikiart/ace/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/wikiart/ace/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting



