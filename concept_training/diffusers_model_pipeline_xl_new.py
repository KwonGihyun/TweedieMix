
from typing import Callable, Optional
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer,CLIPTextModelWithProjection
from accelerate.logging import get_logger

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# from diffusers.pipelines.stable_diffusion import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available
# sys.path.append('./')
from diffusers.models.attention_processor import Attention
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
logger = get_logger(__name__)


def set_use_memory_efficient_attention_xformers(
    self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
):
    if use_memory_efficient_attention_xformers:
        if self.added_kv_proj_dim is not None:
            # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
            # which uses this type of cross attention ONLY because the attention mask of format
            # [0, ..., -10.000, ..., 0, ...,] is not supported
            raise NotImplementedError(
                "Memory efficient attention with `xformers` is currently not supported when"
                " `self.added_kv_proj_dim` is defined."
            )
        elif not is_xformers_available():
            raise ModuleNotFoundError(
                (
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                    " xformers"
                ),
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                " only available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e

        processor = CustomDiffusionXFormersAttnProcessor(attention_op=attention_op)
    else:
        processor = CustomDiffusionAttnProcessor()

    self.set_processor(processor)


class CustomDiffusionAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CustomDiffusionXFormersAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :]*0.
            key = detach*key + (1-detach)*key.detach()
            value = detach*value + (1-detach)*value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomDiffusionPipeline(StableDiffusionXLPipeline):
    r"""
    Pipeline for custom diffusion model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        modifier_token: list of new modifier tokens added or to be added to text_encoder
        modifier_token_id: list of id of new modifier tokens added or to be added to text_encoder
    """
    # _optional_components = ["safety_checker", "feature_extractor", "modifier_token"]
    _optional_components = ["force_zeros_for_empty_prompt", "add_watermarker", "modifier_token","modifier_token_id","modifier_token_id_2"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2:CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        force_zeros_for_empty_prompt: bool = True,
        modifier_token: list = [],
        modifier_token_id: list = [],
        modifier_token_id_2: list = [],
    ):
        super().__init__(vae,
                         text_encoder,
                         text_encoder_2,
                         tokenizer,
                         tokenizer_2,
                         unet,
                         scheduler,
                         force_zeros_for_empty_prompt
                         
                         )

        # change attn class
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id
        self.modifier_token_id_2 = modifier_token_id_2
    def add_token(self, initializer_token):
        initializer_token_id = []
        initializer_token_id_2 = []
        for modifier_token_, initializer_token_ in zip(self.modifier_token, initializer_token):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(modifier_token_)
            num_added_tokens_2 = self.tokenizer_2.add_tokens(modifier_token_)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token_}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode([initializer_token_], add_special_tokens=False)
            token_ids_2 = self.tokenizer_2.encode([initializer_token_], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            self.modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token_))
            initializer_token_id.append(token_ids[0])
            self.modifier_token_id_2.append(self.tokenizer_2.convert_tokens_to_ids(modifier_token_))
            initializer_token_id_2.append(token_ids_2[0])
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for (x, y) in zip(self.modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]
        token_embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data
        for (x, y) in zip(self.modifier_token_id_2, initializer_token_id_2):
            token_embeds_2[x] = token_embeds_2[y]

    def save_pretrained(self, save_path, freeze_model="crossattn_kv", save_text_encoder=False, all=False, save_weight=True,unet=None,text_encoder=None,text_encoder_2=None,modifier_token=None,modifier_token_id=None,modifier_token_id_2=None):
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
    
    def find_disc(self,embed,embed2):
        with torch.no_grad():
            token_embedding = self.text_encoder.get_input_embeddings()
            token_embedding2 = self.text_encoder_2.get_input_embeddings()

            embedding_matrix = token_embedding.weight
            # embedding_matrix = normalize_embeddings(embedding_matrix)
            # print(embedding_matrix.shape)
            embedding_matrix2 = token_embedding2.weight
            # embedding_matrix2 = normalize_embeddings(embedding_matrix2)
            # embed = normalize_embeddings(embed.unsqueeze(0))
            # embed2 = normalize_embeddings(embed2.unsqueeze(0))
            
            embed = embed.unsqueeze(0)
            embed2 = embed2.unsqueeze(0)
            hits = semantic_search(embed, embedding_matrix.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)
            hits2 = semantic_search(embed2, embedding_matrix2.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)
            # if print_hits:
            # all_hits = []
            # for hit in hits:
            #     all_hits.append(hit[0]["score"])
            # print(f"mean hits:{mean(all_hits)}")
            
            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=embed.device)
            nn_indices2 = torch.tensor([hit[0]["corpus_id"] for hit in hits2], device=embed.device)
            print(self.tokenizer.decode(nn_indices.item()))
            # print(self.tokenizer.decode(16666))
            print(self.tokenizer_2.decode(nn_indices2.item()))
    def load_model(self, save_path, compress=False,load_weight=True):
        st = torch.load(save_path)
        
        # print(embedding_matrix2.shape)
        # hits = semantic_search(curr_embeds, embedding_matrix, 
        #                         query_chunk_size=curr_embeds.shape[0], 
        #                         top_k=1,
        #                         score_function=dot_score)
        
        # all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

        
        if 'text_encoder' in st:
            self.text_encoder.load_state_dict(st['text_encoder'])
        if 'modifier_token' in st:
            modifier_tokens = list(st['modifier_token'].keys())
            modifier_token_id = []
            modifier_token_id_2 = []
            
            self.find_disc(st['modifier_token'][modifier_tokens[0]],st['modifier_token_2'][modifier_tokens[0]])
            print(modifier_tokens[0])
            for modifier_token in modifier_tokens:
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)
                num_added_tokens = self.tokenizer_2.add_tokens(modifier_token)
                
                # token_embedding = self.text_encoder.get_input_embeddings()
                # token_embedding2 = self.text_encoder_2.get_input_embeddings()
                
                if num_added_tokens == 0:
                    raise ValueError(
                        f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                        " `modifier_token` that is not already in the tokenizer."
                    )
                modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
                modifier_token_id_2.append(self.tokenizer_2.convert_tokens_to_ids(modifier_token))
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data
            for i, (id_,id_2) in enumerate(zip(modifier_token_id,modifier_token_id_2)):
                
                token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]
            # for i, id_ in enumerate(modifier_token_id_2):
                token_embeds_2[id_2] = st['modifier_token_2'][modifier_tokens[i]]
                
                
        if load_weight:
            for name, params in self.unet.named_parameters():
                if 'attn2' in name:
                    if compress and ('to_k' in name or 'to_v' in name):
                        params.data += st['unet'][name]['u']@st['unet'][name]['v']
                    elif name in st['unet']:
                        # print(name)
                        params.data.copy_(st['unet'][f'{name}'])
