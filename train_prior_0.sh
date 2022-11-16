accelerate launch train_dreambooth.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --instance_data_dir="C:/Users/dsedov/training/dataset/dog" --class_data_dir="C:/Users/dsedov/training/class_dir" --output_dir="C:/Users/dsedov/training/save_model" --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a photo of sks dog" --class_prompt="a photo of dog" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant"  --lr_warmup_steps=0  --num_class_images=200  --max_train_steps=800