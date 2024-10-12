from PIL import Image
from lang_sam import LangSAM
import argparse
import torch
import os 
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--text_condition', type=str)
    parser.add_argument('--output_path', type=str)
    # parser.add_argument('--mode', type=str)
    # parser.add_argument('--bg_negative', type=str, default='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image')  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    # parser.add_argument('--fg_prompts', type=str)
    # parser.add_argument('--fg_negative', type=str, default='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image')  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0','2.1'],
    #                     help="stable diffusion version")
    # parser.add_argument('--H', type=int, default=512)
    # parser.add_argument('--W', type=int, default=512)
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--steps', type=int, default=50)
    # parser.add_argument('--bootstrapping', type=int, default=20)
    opt = parser.parse_args()
    
    model = LangSAM()
    # image_pil = Image.open("./assets/recon_bike.jpg").convert("RGB")
    image_pil = Image.open(opt.input_path).convert("RGB")
    save_folder = opt.output_path
    text_prompts = opt.text_condition.split('+')
    # print(image_pil.size)
    mask_saved = []
    mask_list = []
    mask_orig = []
    mask_names = []
    for tp in text_prompts:
        # print(tp)
        masks, boxes, phrases, logits = model.predict(image_pil, tp)
        
        mask_name = tp+'.jpg'
        # if len(mask_saved)==0:
        # print(masks.shape)
        mask_orig.append(masks[0])
        nonzero_indices = torch.nonzero(masks[0])
        # print(masks[0].shape)
        min_x = torch.min(nonzero_indices[:, 1])
        max_x = torch.max(nonzero_indices[:, 1])
        min_y = torch.min(nonzero_indices[:, 0])
        max_y = torch.max(nonzero_indices[:, 0])
        
        rectangular_mask = torch.zeros_like(masks[0])
        rectangular_mask[min_y:max_y+1, min_x:max_x+1] = 1
        mask_list.append(rectangular_mask)
        mask_names.append(mask_name)

        mask_np = nonzero_indices.cpu().numpy()
        image_np = np.array(image_pil)
        image_np[masks[0]>0] = [0,0,0]
        image_pil = Image.fromarray(image_np)

    # for rm,mo in zip(mask_orig,mask_list):
    
    overlap_region = mask_list[0] & mask_list[1]
    if torch.any(overlap_region !=0):
        
        nonzero_indices_overlap = torch.nonzero(overlap_region)

        min_x = torch.min(nonzero_indices_overlap[:, 1])
        max_x = torch.max(nonzero_indices_overlap[:, 1])
        min_y = torch.min(nonzero_indices_overlap[:, 0])
        max_y = torch.max(nonzero_indices_overlap[:, 0])

        overlap_1 = overlap_region*mask_orig[0]
        # check overlap_region and mask_orig[0] is 80% matched   
        
        
        
        overlap_2 = overlap_region*mask_orig[1]
        if torch.sum(overlap_1)/torch.sum(mask_orig[0])>0.8:
            overlap_2 = torch.zeros_like(overlap_2)
        mask_list[0][min_y:max_y+1, min_x:max_x+1] = overlap_1[min_y:max_y+1, min_x:max_x+1]
        mask_list[1][min_y:max_y+1, min_x:max_x+1] = overlap_2[min_y:max_y+1, min_x:max_x+1]
    
    
    for i,ml in enumerate(mask_list):
        image_mask = ml.cpu().numpy()
        image_mask_pil = Image.fromarray(image_mask)
        image_mask_pil.save(os.path.join(save_folder,mask_names[i]))
        # for mi,mask_single in enumerate(masks):
        #     mask_name = tp+'_'+str(mi)+'.jpg'
        #     image_mask = mask_single.cpu().numpy()
        #     image_mask_pil = Image.fromarray(image_mask)
        #     image_mask_pil.save(os.path.join(save_folder,mask_name))