import torch
import random
from video_gen.pipeline_i2vgen_xl import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

## image source for I2V generation
image_path = ("test_out/photo of a cat and a dog running, mountain background_3821.png")
image = load_image(image_path).convert("RGB")

## text promtp for I2V generation
prompt_orig = "A cat and a dog running, mountain background"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

## I2V generation
## you can change the seed to generate different videos
## using higher injection_timestep and interp_ratio will generate videos which are closer to initial image

# random_integers = [random.randint(1, 10000) for _ in range(20)]
random_integers = [6425]
for i in random_integers:
    pipeline = I2VGenXLPipeline.from_pretrained(
"ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
    pipeline.enable_model_cpu_offload()
    generator = torch.manual_seed(i)

    frames = pipeline(
    prompt=prompt_orig,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9,
    generator=generator,
    height=512,
    width=512,
    target_fps=8,
    injection_timestep=0.02,
    interp_ratio=0.7
    ).frames

    video_path = export_to_gif(frames[0], f"output_i2v_catdog_seed_{str(i)}.gif")