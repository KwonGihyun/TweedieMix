import torch
import os
import random
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[0]
    setattr(conv_module, 't', t)
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)

    conv_module = model.unet.mid_block.resnets[0]
    setattr(conv_module, 't', t)
    conv_module = model.unet.mid_block.resnets[1]
    setattr(conv_module, 't', t)
    
    # down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    # up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # for res in up_res_dict:
    #     for block in up_res_dict[res]:
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         setattr(module, 't', t)
    #         # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
    #         # setattr(module, 'masks', mask)
    #         # setattr(module, 't', t)
            
    # for res in down_res_dict:
    #     for block in down_res_dict[res]:
    #         module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         setattr(module, 't', t)
    #         # module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
    #         # setattr(module, 'masks', mask)

            
    # module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    # setattr(module, 't', t)
   
    

# def load_source_latents_t(t, latents_path):
#     latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
#     assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
#     latents = torch.load(latents_t_path)
#     return latents

# def register_attention_control_efficient(model, injection_schedule):
#     def sa_forward(self):
#         # to_out = self.to_out
#         # if type(to_out) is torch.nn.modules.container.ModuleList:
#         #     print('sig')
#         #     to_out = self.to_out[0]
#         # else:
#         #     to_out = self.to_out

#         def forward(x, encoder_hidden_states=None, attention_mask=None):
            
#             # masks = cross_attention_kwargs['masks']
#             input_ndim = x.ndim
#             # print(input_ndim)
#             # import pdb; pdb.set_trace()
#             batch_size, sequence_length, dim = x.shape
            
            
#             h = self.heads
#             # print(self.group_norm)
#             # print(self.group_norm)
#             is_cross = encoder_hidden_states is not None
#             encoder_hidden_states = encoder_hidden_states if is_cross else x
#             if not is_cross and self.injection_schedule is not None and (
#                     self.t in self.injection_schedule or self.t == 1000):
#             # if not is_cross and self.injection_schedule is not None and (
#             #         self.t not in self.injection_schedule):
                
                
#                 if encoder_hidden_states is None:
#                     encoder_hidden_states = hidden_states
                    
#                 elif self.norm_cross:
                    
#                     encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
                

#                 # print(encoder_hidden_states.shape)
#                 # bt,hw,channels=encoder_hidden_states.shape
#                 # encoder_hidden_states_kv = rearrange(encoder_hidden_states, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)

#                 # first_frame = encoder_hidden_states_kv[:,:1,:,:]
#                 # first_frame_mean = first_frame.mean(dim=2, keepdim=True)
#                 # first_frame_std = first_frame.std(dim=2, keepdim=True)
#                 # first_frame_norm = (first_frame-first_frame_mean)/(first_frame_std+1e-5)
#                 # other_frame_mean = encoder_hidden_states_kv[:,1:,:,:].mean(dim=2, keepdim=True)
#                 # other_frame_std = encoder_hidden_states_kv[:,1:,:,:].std(dim=2, keepdim=True)

#                 # encoder_hidden_states_kv_other = (encoder_hidden_states_kv[:,1:,:,:] - other_frame_mean) / (other_frame_std + 1e-5)
#                 # # encoder_hidden_states_kv[:,1:,:,:] = encoder_hidden_states_kv[:,1:,:,:]*first_frame_std + first_frame_mean
#                 # first_frame_norm=first_frame_norm.repeat(1,15,1,1)
#                 # first_frame = first_frame.repeat(1,15,1,1)
#                 # encoder_hidden_states_kv_other = rearrange(encoder_hidden_states_kv_other, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#                 # encoder_hidden_states_other = rearrange(encoder_hidden_states_kv[:,1:,:,:], 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#                 # first_frame_norm = rearrange(first_frame_norm, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#                 # first_frame =  rearrange(first_frame, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#                 # qo = self.to_q(encoder_hidden_states_kv_other)
#                 # ka = self.to_k(first_frame_norm)
#                 # # vs = self.to_v(first_frame)
#                 # Msqrt2 = F.scaled_dot_product_attention(
#                 #     qo, ka, first_frame**2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#                 # )

#                 # M = F.scaled_dot_product_attention(
#                 #     qo, ka, first_frame, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#                 # )

#                 # S = (Msqrt2-M**2).sqrt()
#                 # encoder_hidden_states_other = encoder_hidden_states_kv_other*S+M
#                 # # print(S.shape)
#                 # encoder_hidden_states_other = rearrange(encoder_hidden_states_other, '(b t) hw c -> b t hw c', b=2,t=15, hw=hw, c=channels)

#                 # encoder_hidden_states = rearrange(encoder_hidden_states, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)
#                 # encoder_hidden_states[:,1:,:,:] =encoder_hidden_states_other
#                 # encoder_hidden_states = rearrange(encoder_hidden_states, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)

#                 # print(encoder_hidden_states.shape)
#                 # bt,hw,channels=encoder_hidden_states.shape
#                 # encoder_hidden_states_kv = rearrange(encoder_hidden_states, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)

#                 # first_frame = encoder_hidden_states_kv[:,:1,:,:]
#                 # first_frame_mean = first_frame.mean(dim=2, keepdim=True)
#                 # first_frame_std = first_frame.std(dim=2, keepdim=True)

#                 # other_frame_mean = encoder_hidden_states_kv[:,1:,:,:].mean(dim=2, keepdim=True)
#                 # other_frame_std = encoder_hidden_states_kv[:,1:,:,:].std(dim=2, keepdim=True)

#                 # encoder_hidden_states_kv[:,1:,:,:] = (encoder_hidden_states_kv[:,1:,:,:] - other_frame_mean) / (other_frame_std + 1e-5)
#                 # encoder_hidden_states_kv[:,1:,:,:] = encoder_hidden_states_kv[:,1:,:,:]*first_frame_std + first_frame_mean

#                 # encoder_hidden_states_kv = rearrange(encoder_hidden_states_kv, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)
#                 q = self.to_q(encoder_hidden_states)
#                 k = self.to_k(encoder_hidden_states)
#                 v = self.to_v(encoder_hidden_states)
                
#                 inner_dim = k.shape[-1]
#                 head_dim = inner_dim // self.heads

#                 # source_batch_size = int(q.shape[0] // 2)
#                 # print(q.shape)
#                 # inject unconditional
#                 # q[1:2] = q[:1]
#                 # k[1:2] = k[:1]
#                 # q[2:] = q[:1]
#                 # k[2:] = k[:1]
#                 # print(x.shape)
#                 # bt,hw,channels=k.shape
#                 # k = rearrange(k, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)
#                 # v = rearrange(v, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)

#                 # first_frame_k = k[:,:1,:,:]
#                 # first_frame_kmean = first_frame_k.mean(dim=2, keepdim=True)
#                 # first_frame_kstd = first_frame_k.std(dim=2, keepdim=True)

#                 # other_frame_kmean = k[:,1:,:,:].mean(dim=2, keepdim=True)
#                 # other_frame_kstd = k[:,1:,:,:].std(dim=2, keepdim=True)

#                 # k[:,1:,:,:] = (k[:,1:,:,:] - other_frame_kmean) / (other_frame_kstd + 1e-5)
#                 # k[:,1:,:,:] = k[:,1:,:,:]*first_frame_kstd + first_frame_kmean


#                 # first_frame_v = v[:,:1,:,:]
#                 # first_frame_vmean = first_frame_v.mean(dim=2, keepdim=True)
#                 # first_frame_vstd = first_frame_v.std(dim=2, keepdim=True)

#                 # other_frame_vmean = v[:,1:,:,:].mean(dim=2, keepdim=True)
#                 # other_frame_vstd = v[:,1:,:,:].std(dim=2, keepdim=True)

#                 # v[:,1:,:,:] = (v[:,1:,:,:] - other_frame_vmean) / (other_frame_vstd + 1e-5)
#                 # v[:,1:,:,:] = v[:,1:,:,:]*first_frame_vstd + first_frame_vmean
#                 # # hidden_states[:,1:,:,:,:] = (hidden_states[:,1:,:,:,:] - other_frame_mean) / (other_frame_std + 1e-5)

#                 # # hidden_states[:,1:,:,:,:] = hidden_states[:,1:,:,:,:]*first_frame_std + first_frame_mean
#                 # # hidden_states[:,1:,:,:,:] = hidden_states[:,:1,:,:,:].detach()

#                 # # k[:,1:,:,:] = k[:,:1,:,:].detach()
#                 # # v[:,1:,:,:] = v[:,:1,:,:].detach()
#                 # print('inj')
#                 # # print(k.shape)
#                 # k = rearrange(k, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)
#                 # v = rearrange(v, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)
                
#                 inner_dim = k.shape[-1]
#                 head_dim = inner_dim // self.heads
                
#                 q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#                 k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#                 v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                
#                 # k = rearrange(k, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)
#                 # v = rearrange(v, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)
#                 # k[:,1:,:,:] = k[:,:1,:,:].detach()
#                 # v[:,1:,:,:] = v[:,:1,:,:].detach()
                
                
#                 # inject conditional
#                 # q[2 * source_batch_size:] = q[:source_batch_size]
#                 # k[2 * source_batch_size:] = k[:source_batch_size]

#                 # q = self.head_to_batch_dim(q)
#                 # print(q.shape)
#                 # k = self.head_to_batch_dim(k)
                
#             else:
#                 q = self.to_q(x)
#                 k = self.to_k(encoder_hidden_states)
#                 v = self.to_v(encoder_hidden_states)
#                 inner_dim = k.shape[-1]
#                 head_dim = inner_dim // self.heads
                
#                 q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#                 k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#                 v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                
#                 # q = self.head_to_batch_dim(q).contiguous()
#                 # k = self.head_to_batch_dim(k).contiguous()
                
                
#             # v = self.head_to_batch_dim(v).contiguous()
            
#             out = F.scaled_dot_product_attention(
#             q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#             out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)




#             # print(torch.max(x))
#             # v_0 = self.head_to_batch_dim(v_0)
#             # v_1 = self.head_to_batch_dim(v_1)
#             # v_2 = self.head_to_batch_dim(v_2)
            
#             # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
#             # sim = self.get_attention_scores(q,k,attention_mask)

#             # if attention_mask is not None:
#             #     attention_mask = attention_mask.reshape(batch_size, -1)
#             #     max_neg_value = -torch.finfo(sim.dtype).max
#             #     attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
#             #     sim.masked_fill_(~attention_mask, max_neg_value)

#             # attention, what we cannot get enough of
#             # attn = sim.softmax(dim=-1)
#             # out = torch.bmm(sim,v)
#             # print(self.residual_connection)
#             # out = torch.einsum("b i j, b j d -> b i d", attn, v)
#             # out = self.batch_to_head_dim(out)
            
#             out = self.to_out[0](out)
#             out = self.to_out[1](out)

            
#             if not is_cross and self.injection_schedule is not None and (
#                     self.t in self.injection_schedule or self.t == 1000):
#                 print('inj_attn')
#                 bt,hw,channels=out.shape
#                 out = rearrange(out, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)

#                 first_frame = out[:,:1,:,:]
#                 first_frame_mean = first_frame.mean(dim=2, keepdim=True)
#                 first_frame_std = first_frame.std(dim=2, keepdim=True)
#                 # first_frame_norm = (first_frame-first_frame_mean)/(first_frame_std+1e-5)
#                 other_frame_mean = out[:,1:,:,:].mean(dim=2, keepdim=True)
#                 other_frame_std = out[:,1:,:,:].std(dim=2, keepdim=True)

#                 out_other = (out[:,1:,:,:] - other_frame_mean) / (other_frame_std + 1e-5)
#                 out[:,1:,:,:] = out_other*first_frame_std + first_frame_mean

#                 out = rearrange(out, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)
#             # # encoder_hidden_states_kv[:,1:,:,:] = encoder_hidden_states_kv[:,1:,:,:]*first_frame_std + first_frame_mean
#             # first_frame_norm=first_frame_norm.repeat(1,15,1,1)
#             # first_frame = first_frame.repeat(1,15,1,1)
#             # encoder_hidden_states_kv_other = rearrange(encoder_hidden_states_kv_other, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#             # encoder_hidden_states_other = rearrange(encoder_hidden_states_kv[:,1:,:,:], 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#             # first_frame_norm = rearrange(first_frame_norm, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#             # first_frame =  rearrange(first_frame, 'b t hw c -> (b t) hw c', b=2,t=15, hw=hw, c=channels)
#             # qo = self.to_q(encoder_hidden_states_kv_other)
#             # ka = self.to_k(first_frame_norm)
#             # # vs = self.to_v(first_frame)
#             # Msqrt2 = F.scaled_dot_product_attention(
#             #     qo, ka, first_frame**2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             # )

#             # M = F.scaled_dot_product_attention(
#             #     qo, ka, first_frame, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             # )

#             # S = (Msqrt2-M**2).sqrt()
#             # encoder_hidden_states_other = encoder_hidden_states_kv_other*S+M
#             # # print(S.shape)
#             # encoder_hidden_states_other = rearrange(encoder_hidden_states_other, '(b t) hw c -> b t hw c', b=2,t=15, hw=hw, c=channels)

#             # encoder_hidden_states = rearrange(encoder_hidden_states, '(b t) hw c -> b t hw c', b=2,t=16, hw=hw, c=channels)
#             # encoder_hidden_states[:,1:,:,:] =encoder_hidden_states_other
#             # encoder_hidden_states = rearrange(encoder_hidden_states, 'b t hw c -> (b t) hw c', b=2,t=16, hw=hw, c=channels)

#             # out = out / self.rescale_output_factor
#             return out

#         return forward
    # down_res_dict = {}
    # down_res_dict_rep = {0: [0, 1], 1: [0, 1],2: [0, 1] }
    # # res_dict_cross = { 2: [0, 1, 2], 3: [0, 1, 2]}
    # # res_dict_rm = {1: [0,1,2]}
    
    # res_dict_rm = { }
    # # res_dict_cross = {1: [0,1,2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # res_dict_cross = {1: [0,1,2], 2: [0, 1, 2], 3: [0, 1, 2]}
    
    # # res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # res_dict = {1: [0, 1, 2]}
    # # res_dict = { 2: [0, 1, 2], 3: [0, 1, 2]}# we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         print("res:"+str(res) + "block:" + str(block))
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.forward = sa_forward(module)
    #         setattr(module, 'injection_schedule', None)
            
    # for res in res_dict_cross:
    #     for block in res_dict_cross[res]:
    #         # if 
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
    #         module.forward = sa_forward(module)
    #         # if res != 1 and block != 0:
    #         setattr(module, 'injection_schedule', injection_schedule)
#     for res in res_dict_rm:
#         for block in res_dict_rm[res]:
#             # if 
#             module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
#             module.forward = sa_forward(module)
#             # if res != 1 and block != 0:
#             setattr(module, 'injection_schedule', None)
#             # else:
#             #     setattr(module, 'injection_schedule', None)
#     for res in down_res_dict:
#         for block in down_res_dict[res]:
            
#             module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
#             module.forward = sa_forward(module)
#             setattr(module, 'injection_schedule', None)
#     for res in down_res_dict_rep:
#         for block in down_res_dict_rep[res]:
            
#             module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
#             module.forward = sa_forward(module)
#             setattr(module, 'injection_schedule', injection_schedule)
            
# #     module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
# #     module.forward = sa_forward(module)
# #     setattr(module, 'injection_schedule', injection_schedule)
    
#     module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
#     module.forward = sa_forward(module)
#     setattr(module, 'injection_schedule', injection_schedule)
    
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet_0.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.forward = sa_forward(module)
    #         setattr(module, 'injection_schedule', injection_schedule)
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet_1.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.forward = sa_forward(module)
    #         setattr(module, 'injection_schedule', injection_schedule)
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet_2.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.forward = sa_forward(module)
    #         setattr(module, 'injection_schedule', injection_schedule)

def register_conv_control_efficient(model, injection_schedule,interp):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 2)
                # inject unconditional
                # print(hidden_states.shape)
                _,channels,h,w=output_tensor.shape 

                output_tensor = rearrange(output_tensor, '(b t) c h w -> b t c h w', b=2, t=16, h=h,w=w, c=channels)
                first_frame =output_tensor[:,:1,:,:,:]
                output_tensor[:,1:,:,:,:] = first_frame.repeat(1,15,1,1,1)
                print('inj_res')
                output_tensor = rearrange(output_tensor, 'b t c h w -> (b t) c h w', b=2,t=16, h=h,w=w, c=channels)
                
            if self.injection_schedule2 is not None and (self.t in self.injection_schedule2 or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 2)

                _,channels,h,w=output_tensor.shape 

                output_tensor = rearrange(output_tensor, '(b t) c h w -> b t c h w', b=2, t=16, h=h,w=w, c=channels)
                first_frame =output_tensor[:,:1,:,:,:]

                output_tensor[:,1:,:,:,:] = self.interp*first_frame.repeat(1,15,1,1,1) + (1-self.interp)*output_tensor[:,1:,:,:,:]
                print('inj_res')
                output_tensor = rearrange(output_tensor, 'b t c h w -> (b t) c h w', b=2,t=16, h=h,w=w, c=channels)

            return output_tensor

        return forward

    conv_module = model.unet.mid_block.resnets[0]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    setattr(conv_module, 'injection_schedule2', None)
    conv_module = model.unet.mid_block.resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    setattr(conv_module, 'injection_schedule2', None)

    conv_module = model.unet.up_blocks[1].resnets[0]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', None)
    setattr(conv_module, 'injection_schedule2', injection_schedule)
    setattr(conv_module, 'interp', interp)
