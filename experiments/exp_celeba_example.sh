#!/bin/bash

# First phase: Generate adversarial perturbations
accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/121 \
    --output_dir data/outputs/celeba/ace/121 \
    --class_data_dir data/class/celeba/ace \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1 \
    --max_f_train_steps 1 \
    --resolution 1024

# Second phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/121 \
    --output_dir models/loras/celeba/ace/121 \
    --class_data_dir data/class/celeba/lora \
    --instance_prompt "a photo of a sks person" \
    --class_prompt "a photo of a person" \
    --mixed_precision bf16 \
    --max_train_steps 1 \
    --resolution 1024



# Third phase: Run inference
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/121/pytorch_lora_weights.safetensors -op outputs/celeba/ace_t2i/121 -spi 100 -p "a photo of a sks person"


python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/121 -op outputs/celeba/ace_i2i/121 -spi 1 -p "a photo of a sks person"


# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path models/loras/celeba/ace/121 --std_path data/celeba/121 -c person


python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace/121 --std_path data/celeba/121 -c person


python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace/121 --std_path data/celeba/121 -c person
