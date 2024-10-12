export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=1
export VAE_PATH=None

accelerate launch --num_processes 1 concept_training/diffusers_training_xl_new.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=/home/gihyun/benchmark_dataset/pet_dog1 \
          --output_dir=/home/gihyun/checkpoint_custom/pet_dog1_test \
          --instance_prompt="photo of a <dog1> dog"  \
          --resolution=512 \
          --train_batch_size=1  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=201 \
          --num_class_images=100 \
          --save_steps=200 \
          --scale_lr --hflip  \
          --modifier_token "<dog1>" \
          --gradient_checkpointing \
          --use_8bit_adam \
          --gradient_accumulation_steps=4 \


# using lora 
# accelerate launch --num_processes 1 concept_training/diffusers_training_xl_lora.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --instance_data_dir=/home/gihyun/benchmark_dataset/pet_dog1 \
#           --output_dir=/home/gihyun/checkpoint_custom/pet_dog1_lora_test \
#           --instance_prompt="photo of a <dog1> dog"  \
#           --resolution=512 \
#           --train_batch_size=1  \
#           --learning_rate=1e-5  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=1001 \
#           --num_class_images=100 \
#           --save_steps=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<dog1>" \
#           --gradient_checkpointing \
#           --use_8bit_adam \
#           --gradient_accumulation_steps=4 \