#!/bin/bash

# First phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/121 \
    --output_dir models/loras/celeba/clean/121 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1135 \
    --output_dir models/loras/celeba/clean/1135 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1422 \
    --output_dir models/loras/celeba/clean/1422 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1499 \
    --output_dir models/loras/celeba/clean/1499 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1657 \
    --output_dir models/loras/celeba/clean/1657 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1000 \
    --resolution 1024

# Second phase: Run inference
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/121/lora_weight.safetensors -op outputs/celeba/clean_t2i/121 -spi 100 -p "a photo of a sks person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1135/lora_weight.safetensors -op outputs/celeba/clean_t2i/1135 -spi 100 -p "a photo of a sks person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1422/lora_weight.safetensors -op outputs/celeba/clean_t2i/1422 -spi 100 -p "a photo of a sks person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1499/lora_weight.safetensors -op outputs/celeba/clean_t2i/1499 -spi 100 -p "a photo of a sks person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1657/lora_weight.safetensors -op outputs/celeba/clean_t2i/1657 -spi 100 -p "a photo of a sks person"

python scripts/run_inference.py -m i2i -ip data/outputs/celeba/clean/121 -op outputs/celeba/clean_i2i/121 -spi 1 -p "a photo of a sks person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/clean/1135 -op outputs/celeba/clean_i2i/1135 -spi 1 -p "a photo of a sks person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/clean/1422 -op outputs/celeba/clean_i2i/1422 -spi 1 -p "a photo of a sks person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/clean/1499 -op outputs/celeba/clean_i2i/1499 -spi 1 -p "a photo of a sks person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/clean/1657 -op outputs/celeba/clean_i2i/1657 -spi 1 -p "a photo of a sks person"

# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/clean/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/clean/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/clean/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/clean/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/clean/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/celeba/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/celeba/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/celeba/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/celeba/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/celeba/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/celeba/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/celeba/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/celeba/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/celeba/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/celeba/1657 --std_path data/celeba/1657 -c person



