#!/bin/bash

# First phase: Generate adversarial perturbations
accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/121 \
    --output_dir data/outputs/celeba/ace/121 \
    --instance_prompt "a photo of a sks person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1135 \
    --output_dir data/outputs/celeba/ace/1135 \
    --instance_prompt "a photo of a sks person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1422 \
    --output_dir data/outputs/celeba/ace/1422 \
    --instance_prompt "a photo of a sks person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1499 \
    --output_dir data/outputs/celeba/ace/1499 \
    --instance_prompt "a photo of a sks person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1657 \
    --output_dir data/outputs/celeba/ace/1657 \
    --instance_prompt "a photo of a sks person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --resolution 1024

# Second phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/121 \
    --output_dir models/loras/celeba/ace/121 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1135 \
    --output_dir models/loras/celeba/ace/1135 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1422 \
    --output_dir models/loras/celeba/ace/1422 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1499 \
    --output_dir models/loras/celeba/ace/1499 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1657 \
    --output_dir models/loras/celeba/ace/1657 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --resolution 1024

# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/1657 --std_path data/celeba/1657 -c person



