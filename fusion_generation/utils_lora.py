import torch
import os
import random
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F
import torchvision

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
   
    down_res_dict = {1: [0, 1], 2: [0, 1]}
    down_transformers_len = {1:[2,2],2:[10,10]}
    up_transformers_len = {0:[10,10,10],1:[2,2,2]}
    up_res_dict_cross = {0:[ 0, 1, 2],1: [0, 1, 2]}
    mid_transformers_len=10
    for res in up_res_dict_cross:
        for block in up_res_dict_cross[res]:
            for idx in range(up_transformers_len[res][block]):
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn1
                setattr(module, 't', t)
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2
                setattr(module, 't', t)
            
    for res in down_res_dict:
        for block in down_res_dict[res]:
            for idx in range(down_transformers_len[res][block]):
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn1
                setattr(module, 't', t)
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                setattr(module, 't', t)

    
    for idx in range(mid_transformers_len):
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn1
        setattr(module, 't', t)
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        setattr(module, 't', t)


def register_attention_control_efficient(model, t_cond,num_concepts):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            if  (self.t in self.t_cond) and encoder_hidden_states.shape[0] == 4:

                q = self.to_q(x)
                qs = [q[:1]]
                for i in range(self.num_concepts):
                    qs.append(q[i+1].unsqueeze(0)+ getattr(self, f"to_q_{i}_lora")(x[i+1].unsqueeze(0)))
                q = torch.cat(qs,dim=0)

                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)
                ks = [k[:1]]
                vs = [v[:1]]
                for i in range(self.num_concepts):
                    ks.append(k[i+1].unsqueeze(0)+ getattr(self, f"to_k_{i}_lora")(encoder_hidden_states[i+1].unsqueeze(0)))
                    vs.append(v[i+1].unsqueeze(0)+ getattr(self, f"to_v_{i}_lora")(encoder_hidden_states[i+1].unsqueeze(0)))
                k = torch.cat(ks,dim=0)
                v = torch.cat(vs,dim=0)

                q = self.head_to_batch_dim(q)
                num_batch = k.shape[0]
                k = self.head_to_batch_dim(k)
                
                # v = self.to_v(encoder_hidden_states[0].unsqueeze(0))
                # vs = [v]
                # for i in range(num_concepts):
                #     vs.append(getattr(self, f"to_v_{i}")(encoder_hidden_states[i+1].unsqueeze(0)))

                # v = torch.cat(vs,dim=0)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                num_batch = k.shape[0]
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.to_v(encoder_hidden_states)
                
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            attn = sim.softmax(dim=-1)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            out_temp = self.batch_to_head_dim(out)
            out = self.to_out[0](out_temp)
            if  (self.t in self.t_cond) and encoder_hidden_states.shape[0] == 4:
                outs = [out[:1]]
                for i in range(self.num_concepts):
                    outs.append(out[i+1].unsqueeze(0)+ getattr(self, f"to_out_{i}_lora")(out_temp[i+1].unsqueeze(0)))
                out = torch.cat(outs,dim=0)

            out = self.to_out[1](out)

            return out

        return forward

    
    down_res_dict = {1: [0, 1], 2: [0, 1]}
    down_transformers_len = {1:[2,2],2:[10,10]}
    up_transformers_len = {0:[10,10,10],1:[2,2,2]}
    up_res_dict_cross = {0:[ 0, 1, 2],1: [0, 1, 2]}
    mid_transformers_len=10
    
    for res in up_res_dict_cross:
        for block in up_res_dict_cross[res]:
            for idx in range(up_transformers_len[res][block]):
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2

                for i in range(num_concepts):
                    module_temp = getattr(model, f"unet_{i}").up_blocks[res].attentions[block].transformer_blocks[idx].attn2.processor
                    setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
                    setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
                    setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
                    setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

                module.forward = sa_forward(module)

                setattr(module, 'num_concepts', num_concepts)
                setattr(module, 't_cond', t_cond)

                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn1

                for i in range(num_concepts):
                    module_temp = getattr(model, f"unet_{i}").up_blocks[res].attentions[block].transformer_blocks[idx].attn1.processor
                    setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
                    setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
                    setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
                    setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

                module.forward = sa_forward(module)

                setattr(module, 'num_concepts', num_concepts)
                setattr(module, 't_cond', t_cond)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            for idx in range(down_transformers_len[res][block]):

                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                for i in range(num_concepts):
                    module_temp = getattr(model, f"unet_{i}").down_blocks[res].attentions[block].transformer_blocks[idx].attn2.processor
                    setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
                    setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
                    setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
                    setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

                module.forward = sa_forward(module)

                setattr(module, 't_cond', t_cond)
                setattr(module, 'num_concepts', num_concepts)

                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn1
                for i in range(num_concepts):
                    module_temp = getattr(model, f"unet_{i}").down_blocks[res].attentions[block].transformer_blocks[idx].attn1.processor
                    setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
                    setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
                    setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
                    setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

                module.forward = sa_forward(module)

                setattr(module, 't_cond', t_cond)
                setattr(module, 'num_concepts', num_concepts)
    for idx in range(mid_transformers_len):
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        for i in range(num_concepts):
            module_temp = getattr(model, f"unet_{i}").mid_block.attentions[0].transformer_blocks[idx].attn2.processor
            setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
            setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
            setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
            setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

        module.forward = sa_forward(module)

        setattr(module, 't_cond', t_cond)
        setattr(module, 'num_concepts', num_concepts)

        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn1
        for i in range(num_concepts):
            module_temp = getattr(model, f"unet_{i}").mid_block.attentions[0].transformer_blocks[idx].attn1.processor
            setattr(module, f'to_q_{i}_lora', module_temp.to_q_lora)
            setattr(module, f'to_k_{i}_lora', module_temp.to_k_lora)
            setattr(module, f'to_v_{i}_lora', module_temp.to_v_lora)
            setattr(module, f'to_out_{i}_lora', module_temp.to_out_lora)

        module.forward = sa_forward(module)

        setattr(module, 't_cond', t_cond)
        setattr(module, 'num_concepts', num_concepts)
