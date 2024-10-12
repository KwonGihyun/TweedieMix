import sys
import argparse
import hashlib
import itertools
import logging
import math
import os
from pathlib import Path
from typing import Optional
import torch
import json
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version

import transformers
import diffusers
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, is_wandb_available

sys.path.append('./')
from concept_training.diffusers_model_pipeline_xl_new import CustomDiffusionAttnProcessor, set_use_memory_efficient_attention_xformers
from concept_training.diffusers_data_pipeline_xl import CustomDiffusionDataset, PromptDataset, collate_fn
from concept_training.diffusers_model_pipeline_xl_new import CustomDiffusionPipeline
from concept_training import retrieve

check_min_version("0.14.0")

logger = get_logger(__name__)

def save_checkpoint(save_path, freeze_model="crossattn_kv", save_text_encoder=False, all=False, save_weight=True,unet=None,text_encoder=None,text_encoder_2=None,modifier_token=None,modifier_token_id=None,modifier_token_id_2=None):
    if all:
        super().save_pretrained(save_path)
    else:
        delta_dict = {'unet': {}, 'modifier_token': {}, 'modifier_token_2': {}}
        if modifier_token is not None:
            for i in range(len(modifier_token_id)):
                learned_embeds = text_encoder.get_input_embeddings().weight[modifier_token_id[i]]
                learned_embeds_2 = text_encoder_2.get_input_embeddings().weight[modifier_token_id_2[i]]
                delta_dict['modifier_token'][modifier_token[i]] = learned_embeds.detach().cpu()
                delta_dict['modifier_token_2'][modifier_token[i]] = learned_embeds_2.detach().cpu()
        if save_text_encoder:
            delta_dict['text_encoder'] = text_encoder.state_dict()
        if save_weight:
            for name, params in unet.named_parameters():
                if freeze_model == "crossattn":
                    if 'attn2' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif freeze_model == "crossattn_kv":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                else:
                    raise ValueError(
                            "freeze_model argument only supports crossattn_kv or crossattn"
                        )
        torch.save(delta_dict, save_path)

def create_custom_diffusion(unet, freeze_model):
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False
        elif freeze_model == "crossattn_kv":
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False
        else:
            raise ValueError(
                    "freeze_model argument only supports crossattn_kv or crossattn"
                )

    # change attn class
    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == Attention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(layer, layer.__class__)
                setattr(layer, 'set_use_memory_efficient_attention_xformers', bound_method)
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


def freeze_params(params):
    for param in params:
        param.requires_grad = False


# def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
#     text_encoder_config = PretrainedConfig.from_pretrained(
#         pretrained_model_name_or_path,
#         subfolder="text_encoder",
#         revision=revision,
#     )
#     model_class = text_encoder_config.architectures[0]

#     if model_class == "CLIPTextModel":
#         from transformers import CLIPTextModel

#         return CLIPTextModel
#     elif model_class == "RobertaSeriesModelWithTransformation":
#         from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

#         return RobertaSeriesModelWithTransformation
#     else:
#         raise ValueError(f"{model_class} is not supported.")
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids

# def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
#     text_encoder_config = PretrainedConfig.from_pretrained(
#         pretrained_model_name_or_path,
#         subfolder="text_encoder",
#         revision=revision,
#     )
#     model_class = text_encoder_config.architectures[0]

#     if model_class == "CLIPTextModel":
#         from transformers import CLIPTextModel

#         return CLIPTextModel
#     elif model_class == "RobertaSeriesModelWithTransformation":
#         from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

#         return RobertaSeriesModelWithTransformation
#     elif model_class == "T5EncoderModel":
#         from transformers import T5EncoderModel

#         return T5EncoderModel
#     else:
#         raise ValueError(f"{model_class} is not supported.")
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default='crossattn_kv',
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default='ktn+pll+ucd', help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                if accelerator.is_main_process:
                    if not Path(os.path.join(class_images_dir, 'images')).exists() or len(list(Path(os.path.join(class_images_dir, 'images')).iterdir())) < args.num_class_images:
                        retrieve.retrieve(concept['class_prompt'], class_images_dir, args.num_class_images)
                concept['class_prompt'] = os.path.join(class_images_dir, 'caption.txt')
                concept['class_data_dir'] = os.path.join(class_images_dir, 'images.txt')
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(concept['class_prompt'], num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"], num_inference_steps=50, guidance_scale=6., eta=1.).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id

            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.tokenizer_name,
    #         revision=args.revision,
    #         use_fast=False,
    #     )
    # elif args.pretrained_model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         subfolder="tokenizer",
    #         revision=args.revision,
    #         use_fast=False,
    #     )

    # import correct text encoder class
    # text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    # text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
                    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # text_encoder = text_encoder_cls.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, use_safetensors=True
    # )
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder and args.modifier_token is None:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    unet = create_custom_diffusion(unet, args.freeze_model)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # if accelerator.mixed_precision != "fp16":
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            # text_encoder_one.enable_xformers_memory_efficient_attention()
            # text_encoder_two.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ## check this##
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder or args.modifier_token is not None:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate*2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    print(optimizer_class)
    # tokenizers = [tokenizer_one, tokenizer_two]
    # text_encoders = [text_encoder_one, text_encoder_two]
    # def compute_text_embeddings(prompt, text_encoders, tokenizers):
    #     with torch.no_grad():
    #         prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
    #         prompt_embeds = prompt_embeds.to(accelerator.device)
    #         pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
    #     return prompt_embeds, pooled_prompt_embeds
    
    instance_time_ids = compute_time_ids()
    # if not args.train_text_encoder:
    #     instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
    #         args.instance_prompt, text_encoders, tokenizers
    #     )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        class_time_ids = class_time_ids.to(accelerator.device)
        # if not args.train_text_encoder:
        #     class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
        #         args.class_prompt, text_encoders, tokenizers
        #     )

    # Clear the memory here.
    # if not args.train_text_encoder:
    #     del tokenizers, text_encoders
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # Pack the statically computed variables appropriately. This is so that we don't
    # have to pass them to the dataloader.
    add_time_ids = instance_time_ids.to(accelerator.device)
    
    if args.with_prior_preservation:
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)

    # if not args.train_text_encoder:
    #     prompt_embeds = instance_prompt_hidden_states
    #     unet_add_text_embeds = instance_pooled_prompt_embeds
    #     if args.with_prior_preservation:
    #         prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
    #         unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
    # else:
    #     tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt)
    #     tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt)
    #     if args.with_prior_preservation:
    #         class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt)
    #         class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt)
    #         tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
    #         tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id_one = []
    modifier_token_id_two = []
    initializer_token_id_one = []
    initializer_token_id_two = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split('+')
        args.initializer_token = args.initializer_token.split('+')
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(args.modifier_token, args.initializer_token[:len(args.modifier_token)]):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer_one.add_tokens(modifier_token)
            num_added_tokens = tokenizer_two.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids_one = tokenizer_one.encode([initializer_token], add_special_tokens=False)
            token_ids_two = tokenizer_two.encode([initializer_token], add_special_tokens=False)
            # print(token_ids_two)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids_one) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id_one.append(token_ids_one[0])
            initializer_token_id_two.append(token_ids_two[0])
            modifier_token_id_one.append(tokenizer_one.convert_tokens_to_ids(modifier_token))
            modifier_token_id_two.append(tokenizer_two.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder_one.resize_token_embeddings(len(tokenizer_one))
        text_encoder_two.resize_token_embeddings(len(tokenizer_two))
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds_one = text_encoder_one.get_input_embeddings().weight.data
        token_embeds_two = text_encoder_two.get_input_embeddings().weight.data
        for (x,y) in zip(modifier_token_id_one,initializer_token_id_one):
            token_embeds_one[x] = token_embeds_one[y]
        for (x,y) in zip(modifier_token_id_two,initializer_token_id_two):
            token_embeds_two[x] = token_embeds_two[y]
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder_one.text_model.encoder.parameters(),
            text_encoder_one.text_model.final_layer_norm.parameters(),
            text_encoder_one.text_model.embeddings.position_embedding.parameters(),
            text_encoder_two.text_model.encoder.parameters(),
            text_encoder_two.text_model.final_layer_norm.parameters(),
            text_encoder_two.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

        if args.freeze_model == 'crossattn':
            params_to_optimize = itertools.chain( text_encoder.get_input_embeddings().parameters() , [x[1] for x in unet.named_parameters() if 'attn2' in x[0]] )
        else:
            params_to_optimize = itertools.chain( text_encoder_one.get_input_embeddings().parameters(),text_encoder_two.get_input_embeddings().parameters() , [x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])] )

    ########################################################
    ########################################################
    else:
        if args.freeze_model == 'crossattn':
            params_to_optimize = (
                itertools.chain([x[1] for x in unet.named_parameters() if 'attn2' in x[0]], text_encoder.parameters() if args.train_text_encoder else [] ) 
            )
        else:
            params_to_optimize = (
                itertools.chain([x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])], text_encoder.parameters() if args.train_text_encoder else [] ) 
            )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder or args.modifier_token is not None:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one,text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    first_epoch = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder or args.modifier_token is not None:
            text_encoder_one.train()
            text_encoder_two.train()
        for step, batch in enumerate(train_dataloader):
            # if epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                # print(latents.shape)
                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                elems_to_repeat = bsz // 2 if args.with_prior_preservation else bsz
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat, 1)}
                # with accelerator.autocast():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat, 1)})
                prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat, 1, 1)
                model_pred = unet(
                    noisy_latents, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                ).sample
                
                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # encoder_hidden_states = encode_prompt(
                #         text_encoder,
                #         batch["input_ids"],
                #         batch["attention_mask"],
                #         text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                #     )
                # print(encoder_hidden_states.shape)
                
#                 prompt_embeds = text_encoder(
#                     batch["input_ids"],
#                     output_hidden_states=True,
#                 )

#                 # We are only ALWAYS interested in the pooled output of the final text encoder
#                 pooled_prompt_embeds = prompt_embeds[0]
#                 prompt_embeds = prompt_embeds.hidden_states[-2]
                # print(prompt_embeds.shape)
                # Predict the noise residual
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0]
                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])).mean()

                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder_one = text_encoder_one.module.get_input_embeddings().weight.grad
                        grads_text_encoder_two = text_encoder_two.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder_one = text_encoder_one.get_input_embeddings().weight.grad
                        grads_text_encoder_two = text_encoder_two.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero_one = torch.arange(len(tokenizer_one)) != modifier_token_id_one[0]
                    index_grads_to_zero_two = torch.arange(len(tokenizer_two)) != modifier_token_id_two[0]
                    for i in range(len(modifier_token_id_one[1:])):
                        index_grads_to_zero_one = index_grads_to_zero_one & (torch.arange(len(tokenizer_one)) != modifier_token_id_one[i])
                    grads_text_encoder_one.data[index_grads_to_zero_one, :] = grads_text_encoder_one.data[index_grads_to_zero_one, :].fill_(0)
                    for i in range(len(modifier_token_id_two[1:])):
                        index_grads_to_zero_two = index_grads_to_zero_two & (torch.arange(len(tokenizer_two)) != modifier_token_id_two[i])
                    grads_text_encoder_two.data[index_grads_to_zero_two, :] = grads_text_encoder_two.data[index_grads_to_zero_two, :].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])], text_encoder_one.parameters(), text_encoder_two.parameters())
                        if (args.train_text_encoder or args.modifier_token is not None)
                        else itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])]) 
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # pipeline = CustomDiffusionPipeline.from_pretrained(
                        #     args.pretrained_model_name_or_path,
                        #     unet=accelerator.unwrap_model(unet),
                        #     text_encoder=accelerator.unwrap_model(text_encoder_one),
                        #     text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        #     tokenizer=tokenizer_one,
                        #     tokenizer_2=tokenizer_two,
                        #     revision=args.revision,
                        #     modifier_token=args.modifier_token,
                        #     modifier_token_id=modifier_token_id_one,
                        #     modifier_token_id_2=modifier_token_id_two,
                        # )
                        save_path = os.path.join(args.output_dir, f"delta-{global_step}.bin")
                        # pipeline.save_pretrained(save_path, freeze_model=args.freeze_model)
                        save_checkpoint(save_path, freeze_model=args.freeze_model,unet=accelerator.unwrap_model(unet)
                                                 ,text_encoder=accelerator.unwrap_model(text_encoder_one),
                                                 text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                                                 modifier_token=args.modifier_token,
                                                 modifier_token_id=modifier_token_id_one,
                                                 modifier_token_id_2=modifier_token_id_two)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    #     # create pipeline
    #     unet = unet.to(torch.float32)
    #     pipeline = CustomDiffusionPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         unet=accelerator.unwrap_model(unet),
    #         text_encoder=accelerator.unwrap_model(text_encoder),
    #         tokenizer=tokenizer,
    #         revision=args.revision,
    #         modifier_token=args.modifier_token,
    #         modifier_token_id=modifier_token_id,
    #     )
    #     save_path = os.path.join(args.output_dir, f"delta.bin")
    #     pipeline.save_pretrained(save_path, freeze_model=args.freeze_model)
    #     if args.validation_prompt is not None:
    #         logger.info(
    #             f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
    #             f" {args.validation_prompt}."
    #         )
    #         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    #         pipeline = pipeline.to(accelerator.device)
    #         pipeline.set_progress_bar_config(disable=True)

    #         # run inference
    #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    #         images = [
    #             pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
    #             for _ in range(args.num_validation_images)
    #         ]

    #         for tracker in accelerator.trackers:
    #             if tracker.name == "tensorboard":
    #                 np_images = np.stack([np.asarray(img) for img in images])
    #                 tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
    #             if tracker.name == "wandb":
    #                 tracker.log(
    #                     {
    #                         "validation": [
    #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                             for i, image in enumerate(images)
    #                         ]
    #                     }
    #                 )

    #     del pipeline
    #     torch.cuda.empty_cache()


    #     if args.push_to_hub:
    #         repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
