export MODEL_NAME="stabilityai/stable-diffusion-2-base"
export dataset_name="/home/dsedov/Dev/dreambooth/dataset/storefront"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --dataset_name="/home/dsedov/Dev/dreambooth/dataset/storefront" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=20000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/home/dsedov/Dev/dreambooth/models/dystorefront" 