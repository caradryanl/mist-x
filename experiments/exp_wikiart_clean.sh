#!/bin/bash

# First phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/wikiart/Fauvism_henri-matisse \
    --output_dir models/loras/wikiart/clean/Fauvism_henri-matisse \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/wikiart/Impressionism_claude-monet \
    --output_dir models/loras/wikiart/clean/Impressionism_claude-monet \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/wikiart/Pointillism_paul-signac \
    --output_dir models/loras/wikiart/clean/Pointillism_paul-signac \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/wikiart/Post_Impressionism-van-gogh \
    --output_dir models/loras/wikiart/clean/Post_Impressionism-van-gogh \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/wikiart/Rococo_canaletto \
    --output_dir models/loras/wikiart/clean/Rococo_canaletto \
    --class_data_dir data/class/wikiart/lora \
    --instance_prompt "a photo of a sks painting" \
    --class_prompt "a photo of a painting" \
    --mixed_precision bf16 \
    --resolution 1024

# Second phase: Run inference
python scripts/run_inference.py.py -m t2i -lp models/loras/wikiart/clean/Fauvism_henri-matisse/lora_weight.safetensors -op outputs/wikiart/clean_t2i/Fauvism_henri-matisse -spi 100 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m t2i -lp models/loras/wikiart/clean/Impressionism_claude-monet/lora_weight.safetensors -op outputs/wikiart/clean_t2i/Impressionism_claude-monet -spi 100 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m t2i -lp models/loras/wikiart/clean/Pointillism_paul-signac/lora_weight.safetensors -op outputs/wikiart/clean_t2i/Pointillism_paul-signac -spi 100 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m t2i -lp models/loras/wikiart/clean/Post_Impressionism-van-gogh/lora_weight.safetensors -op outputs/wikiart/clean_t2i/Post_Impressionism-van-gogh -spi 100 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m t2i -lp models/loras/wikiart/clean/Rococo_canaletto/lora_weight.safetensors -op outputs/wikiart/clean_t2i/Rococo_canaletto -spi 100 -p "a photo of a sks painting"

python scripts/run_inference.py.py -m i2i -ip data/wikiart/Fauvism_henri-matisse -op outputs/wikiart/clean_i2i/Fauvism_henri-matisse -spi 1 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m i2i -ip data/wikiart/Impressionism_claude-monet -op outputs/wikiart/clean_i2i/Impressionism_claude-monet -spi 1 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m i2i -ip data/wikiart/Pointillism_paul-signac -op outputs/wikiart/clean_i2i/Pointillism_paul-signac -spi 1 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m i2i -ip data/wikiart/Post_Impressionism-van-gogh -op outputs/wikiart/clean_i2i/Post_Impressionism-van-gogh -spi 1 -p "a photo of a sks painting"
python scripts/run_inference.py.py -m i2i -ip data/wikiart/Rococo_canaletto -op outputs/wikiart/clean_i2i/Rococo_canaletto -spi 1 -p "a photo of a sks painting"

# Third phase: Run evaluation
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/clean/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/clean/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/clean/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/clean/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/wikiart/clean/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python scripts/eval_attacks.py -m MSSSIM --path data/wikiart/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/wikiart/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/wikiart/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/wikiart/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m MSSSIM --path data/wikiart/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python scripts/eval_attacks.py -m CLIPI2I --path data/wikiart/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/wikiart/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/wikiart/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/wikiart/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python scripts/eval_attacks.py -m CLIPI2I --path data/wikiart/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting