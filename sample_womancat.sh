
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=0

## GPU for text-guided segmentation 
## we recommend to use different GPU from generation to avoid memory issue
SEG_GPU=1

## background must comes last
## prompt for concept 1 + prompt for concept 2 + prompt for concept 3(background)
PROMPT="zoomed photo of a cat, lighthouse background+zoomed photo of a woman, lighthouse background+ lighthouse background"
## prompt for multiple concepts 
PROMPT_ORIG="zoomed photo of a cat and a woman, lighthouse background"

export RESULT_PATH="./test_out_woman"
export SEED=3856

## concept order must be the same as the prompt
CONCEPTS="cat+woman+lighthouse"
MODIFIER="<cat1>+<woman1>+<lighthouse1>"
## concepts for text-guided segmentation. background concept must not be included
SEG_CONCEPTS="a cat+a woman"

## guidance_scale = CFG weight 0<=guidance_scale<=1
## t_cond = timestep to start multiconcept sampling
## jumping_steps = number of steps to sampling from intermediate tweedie
## resampling_steps = number of steps to multi-concept resampling

## -----------when using custom diffusion weights-----------

# PERSONAL_CHECKPOINT="../checkpoint_custom/pet_cat5/delta-100.bin+../checkpoint_custom/spring_woman/delta-400.bin+../checkpoint_custom/scene_lighthouse/delta-200.bin"

# python fusion_generation/fusion_sampling.py \
# --guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
# --output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --concepts "$CONCEPTS" --modifier_token $MODIFIER --resolution_h 1024 --resolution_w 1024 \
# --prompt_orig "$PROMPT_ORIG" --seed $SEED --t_cond 0.2 --seg_concepts="$SEG_CONCEPTS" --negative_prompt '' --seg_gpu $SEG_GPU


## -----------when using lora weights-----------

PERSONAL_CHECKPOINT="../checkpoint_custom/pet_cat5_lora/delta-1000.bin+../checkpoint_custom/spring_woman_lora/delta-1000.bin+../checkpoint_custom/scene_lighthouse_lora/delta-1000.bin"

python fusion_generation/fusion_sampling_lora.py \
--guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
--output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --concepts "$CONCEPTS" --modifier_token $MODIFIER --resolution_h 1024 --resolution_w 1024 \
--prompt_orig "$PROMPT_ORIG" --seed $SEED --t_cond 0.2 --t_stop 0.8 --seg_concepts="$SEG_CONCEPTS" --negative_prompt '' --seg_gpu $SEG_GPU