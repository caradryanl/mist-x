#!/bin/bash

# First phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/121 \
    --output_dir models/loras/celeba/clean/121 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1135 \
    --output_dir models/loras/celeba/clean/1135 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1422 \
    --output_dir models/loras/celeba/clean/1422 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1499 \
    --output_dir models/loras/celeba/clean/1499 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/celeba/1657 \
    --output_dir models/loras/celeba/clean/1657 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

# Second phase: Run inference
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/121/pytorch_lora_weights.safetensors -op data/outputs/celeba/clean_t2i/121 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1135/pytorch_lora_weights.safetensors -op data/outputs/celeba/clean_t2i/1135 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1422/pytorch_lora_weights.safetensors -op data/outputs/celeba/clean_t2i/1422 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1499/pytorch_lora_weights.safetensors -op data/outputs/celeba/clean_t2i/1499 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/clean/1657/pytorch_lora_weights.safetensors -op data/outputs/celeba/clean_t2i/1657 -spi 100 -p "a photo of a specific person"

python scripts/run_inference.py -m i2i -ip data/celeba/121 -op data/outputs/celeba/clean_i2i/121 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/celeba/1135 -op data/outputs/celeba/clean_i2i/1135 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/celeba/1422 -op data/outputs/celeba/clean_i2i/1422 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/celeba/1499 -op data/outputs/celeba/clean_i2i/1499 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/celeba/1657 -op data/outputs/celeba/clean_i2i/1657 -spi 1 -p "a photo of a specific person"

# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/clean_t2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/clean_i2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/clean_i2i/1657 --std_path data/celeba/1657 -c person



# First phase: Generate adversarial perturbations
accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/121 \
    --output_dir data/outputs/celeba/ace/121 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --max_f_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1135 \
    --output_dir data/outputs/celeba/ace/1135 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --max_f_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1422 \
    --output_dir data/outputs/celeba/ace/1422 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --max_f_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1499 \
    --output_dir data/outputs/celeba/ace/1499 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --max_f_train_steps 5 \
    --resolution 1024

accelerate launch attacks/ace_sd3.py \
    --instance_data_dir data/celeba/1657 \
    --output_dir data/outputs/celeba/ace/1657 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --max_f_train_steps 5 \
    --resolution 1024

# Second phase: Train LoRA on perturbed images
accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/121 \
    --output_dir models/loras/celeba/ace/121 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1135 \
    --output_dir models/loras/celeba/ace/1135 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1422 \
    --output_dir models/loras/celeba/ace/1422 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1499 \
    --output_dir models/loras/celeba/ace/1499 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

accelerate launch scripts/train_dreambooth_lora_sd3.py \
    --instance_data_dir data/outputs/celeba/ace/1657 \
    --output_dir models/loras/celeba/ace/1657 \
    --instance_prompt "a photo of a specific person" \
    --mixed_precision bf16 \
    --max_train_steps 2000 \
    --resolution 1024

# Third phase: Run inference
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/121/pytorch_lora_weights.safetensors -op data/outputs/celeba/ace_t2i/121 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/1135/pytorch_lora_weights.safetensors -op data/outputs/celeba/ace_t2i/1135 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/1422/pytorch_lora_weights.safetensors -op data/outputs/celeba/ace_t2i/1422 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/1499/pytorch_lora_weights.safetensors -op data/outputs/celeba/ace_t2i/1499 -spi 100 -p "a photo of a specific person"
python scripts/run_inference.py -m t2i -lp models/loras/celeba/ace/1657/pytorch_lora_weights.safetensors -op data/outputs/celeba/ace_t2i/1657 -spi 100 -p "a photo of a specific person"

python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/121 -op data/outputs/celeba/ace_i2i/121 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/1135 -op data/outputs/celeba/ace_i2i/1135 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/1422 -op data/outputs/celeba/ace_i2i/1422 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/1499 -op data/outputs/celeba/ace_i2i/1499 -spi 1 -p "a photo of a specific person"
python scripts/run_inference.py -m i2i -ip data/outputs/celeba/ace/1657 -op data/outputs/celeba/ace_i2i/1657 -spi 1 -p "a photo of a specific person"

# Third phase: Run scripts/eval_attacks
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPT2I --path data/outputs/celeba/ace_t2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m MSSSIM --path data/outputs/celeba/ace_i2i/1657 --std_path data/celeba/1657 -c person

python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/121 --std_path data/celeba/121 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1135 --std_path data/celeba/1135 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1422 --std_path data/celeba/1422 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1499 --std_path data/celeba/1499 -c person
python scripts/eval_attacks.py -m CLIPI2I --path data/outputs/celeba/ace_i2i/1657 --std_path data/celeba/1657 -c person