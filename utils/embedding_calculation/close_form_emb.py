import torch
import copy
import ast
from tqdm import tqdm
import argparse
import os
from diffusers import StableDiffusionPipeline
import sys
from . import embedding2img, setup_seed
import pandas as pd

def close_form_emb(model, model_copy, concept, with_to_k=True, save_path=None, save_name=None, old_target_concept=None, regeular_scale=1e-3, seed=123, reg_item='both'):
    ### collect all the cross attns modules
    print(f'concept: {concept}, old_target_concept: {old_target_concept}')
    sub_nets = model.unet.named_children()
    sub_nets_copy = model_copy.unet.named_children()
    ca_layers = []
    ca_layers_copy = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    for net in sub_nets_copy:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers_copy.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers_copy.append(transformer.attn2)
    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    projection_matrices_copy = [l.to_v for l in ca_layers_copy]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers] 
    og_matrices_copy = [copy.deepcopy(l.to_v) for l in ca_layers_copy]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        projection_matrices_copy = projection_matrices_copy + [l.to_k for l in ca_layers_copy]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
        og_matrices_copy = og_matrices_copy + [copy.deepcopy(l.to_k) for l in ca_layers_copy]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[idx_ + num_ca_clip_layers])
            projection_matrices[idx_ + num_ca_clip_layers] = l.to_k
    for idx_, l in enumerate(ca_layers_copy):
        l.to_v = copy.deepcopy(og_matrices_copy[idx_])
        projection_matrices_copy[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices_copy[idx_ + num_ca_clip_layers])
            projection_matrices_copy[idx_ + num_ca_clip_layers] = l.to_k

    ### get the concept embedding
    concept_tokens =  model.tokenizer(
                    concept,
                    padding="max_length",
                    max_length=model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
    concept_ids = concept_tokens["input_ids"].to(model.device)
    concept_embedding = model.text_encoder(concept_ids)[0].squeeze(0)
    
    if old_target_concept is None:
        print(f'old_target_concept is None, using surrogate regularization with {reg_item} regularization item')
        if '1' in reg_item:
            mat1 = torch.zeros((projection_matrices[0].weight.shape[1], projection_matrices[0].weight.shape[1])).to(model.device)
            mat2 = torch.zeros((projection_matrices[0].weight.shape[1], projection_matrices[0].weight.shape[1])).to(model.device)
        elif reg_item == 'both' or '2' in reg_item:
            mat1 = torch.eye(projection_matrices[0].weight.shape[1]).to(model.device)*regeular_scale
            mat2 = torch.eye(projection_matrices[0].weight.shape[1]).to(model.device)*regeular_scale
        else: 
            raise NotImplementedError
        for idx_, l in enumerate(projection_matrices):
            if reg_item == 'both' or '1' in reg_item:
                mat1 = mat1 + torch.matmul(l.weight.T, l.weight)*(1+regeular_scale)
                mat2 = mat2 + torch.matmul(l.weight.T, projection_matrices_copy[idx_].weight) + regeular_scale*torch.matmul(l.weight.T, l.weight)
            elif '2' in reg_item:
                mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
                mat2 = mat2 + torch.matmul(l.weight.T, projection_matrices_copy[idx_].weight)
            else:
                raise NotImplementedError
        coefficent = torch.matmul(torch.inverse(mat1), mat2)
        adv_embedding = torch.matmul(concept_embedding, coefficent.T).unsqueeze(0)
    else:
        print('old_target_concept is not None, using standard regularization')
        mat1  = torch.eye(projection_matrices[0].weight.shape[1]).to(model.device)*regeular_scale
        old_target_concept_tokens =  model.tokenizer(
                    old_target_concept,
                    padding="max_length",
                    max_length=model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
        old_target_concept_ids = old_target_concept_tokens["input_ids"].to(model.device)
        old_target_concept_embedding = model.text_encoder(old_target_concept_ids)[0].squeeze(0)
        mat2_1 = old_target_concept_embedding.clone().detach()
        mat2_2 = torch.zeros((projection_matrices[0].weight.shape[1], projection_matrices[0].weight.shape[1])).to(model.device)
        for idx_, l in enumerate(projection_matrices):
            mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
            mat2_2 = mat2_2 + torch.matmul(l.weight.T, projection_matrices_copy[idx_].weight)
        mat2 = mat2_1*regeular_scale + torch.matmul(concept_embedding, mat2_2.T)
        adv_embedding = torch.matmul(mat2, torch.inverse(mat1).T).unsqueeze(0)
        

    if save_path is not None:
        print(f'Saving the embedding and image to {save_path}')
        # save the new embedding as pt
        torch.save(adv_embedding, f'{save_path}/close_new_embedding_preserved.pt')
        # create a df with column as concept and evaluation_seed
        df = pd.DataFrame(columns=['concept', 'evaluation_seed'])
        df['concept'] = [concept]
        df['evaluation_seed'] = [seed]
        
        embedding2img(adv_embedding, df, model, save_path, save_name=save_name)
    return concept_embedding.unsqueeze(0), adv_embedding

def close_form_emb_regzero(model, model_copy, concept, with_to_k=True, save_path=None, save_name=None, regeular_scale=1e-3, seed=123):
    ### collect all the cross attns modules
    print('Regularization to 0 vector')
    sub_nets = model.unet.named_children()
    sub_nets_copy = model_copy.unet.named_children()
    ca_layers = []
    ca_layers_copy = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    for net in sub_nets_copy:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers_copy.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers_copy.append(transformer.attn2)
    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    projection_matrices_copy = [l.to_v for l in ca_layers_copy]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers] 
    og_matrices_copy = [copy.deepcopy(l.to_v) for l in ca_layers_copy]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        projection_matrices_copy = projection_matrices_copy + [l.to_k for l in ca_layers_copy]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
        og_matrices_copy = og_matrices_copy + [copy.deepcopy(l.to_k) for l in ca_layers_copy]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[idx_ + num_ca_clip_layers])
            projection_matrices[idx_ + num_ca_clip_layers] = l.to_k
    for idx_, l in enumerate(ca_layers_copy):
        l.to_v = copy.deepcopy(og_matrices_copy[idx_])
        projection_matrices_copy[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices_copy[idx_ + num_ca_clip_layers])
            projection_matrices_copy[idx_ + num_ca_clip_layers] = l.to_k

    ### get the concept embedding
    concept_tokens =  model.tokenizer(
                    concept,
                    padding="max_length",
                    max_length=model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
    concept_ids = concept_tokens["input_ids"].to(model.device)
    concept_embedding = model.text_encoder(concept_ids)[0].squeeze(0)
    # ###### slice ######
    # final_token_idx = concept_tokens.attention_mask[0].sum().item()-2
    # concept_embedding = concept_embedding[final_token_idx:]
    # ###### slice ######
    
    mat1 = torch.eye(projection_matrices[0].weight.shape[1]).to(model.device)*regeular_scale
    mat2 = torch.zeros((projection_matrices[0].weight.shape[1], projection_matrices[0].weight.shape[1])).to(model.device)

    for idx_, l in enumerate(projection_matrices):
        mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
        mat2 = mat2 + torch.matmul(l.weight.T, projection_matrices_copy[idx_].weight)
    coefficent = torch.matmul(torch.inverse(mat1), mat2)
    adv_embedding = torch.matmul(concept_embedding, coefficent.T).unsqueeze(0)
        

    if save_path is not None:
        print(f'Saving the embedding and image to {save_path}')
        # save the new embedding as pt
        torch.save(adv_embedding, f'{save_path}/close_new_embedding_preserved.pt')
        # create a df with column as concept and evaluation_seed
        df = pd.DataFrame(columns=['concept', 'evaluation_seed'])
        df['concept'] = [concept]
        df['evaluation_seed'] = [seed]
        
        embedding2img(adv_embedding, df, model, save_path, save_name=save_name)
    return concept_embedding.unsqueeze(0), adv_embedding

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_path', type=str, default='/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--finetuned_model_path', type=str, default='/share/ckpt/gongchao/unified-concept-editing/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt')
    parser.add_argument('--concept', type=str, default='nudity')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reg_scale', type=float, default=1e-3)
    parser.add_argument('--with_to_k', type=bool, default=True)
    parser.add_argument('--old_target_concept', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='/share/ckpt/gongchao/SD_adv_train/close_form_new_emb/regzero')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--reg_item', type=str, default='1st')
    args = parser.parse_args()
    ori_model_path = args.ori_model_path
    finetuned_model_path = args.finetuned_model_path
    concept = args.concept
    reg_scale = args.reg_scale
    with_to_k = args.with_to_k
    save_path = f'{args.save_path}/{concept}/seed_{args.seed}/reg_{reg_scale}'
    os.makedirs(save_path, exist_ok=True)
    save_name = args.save_name
    reg_item = args.reg_item
    
    setup_seed(args.seed)
    
    model = StableDiffusionPipeline.from_pretrained(ori_model_path)
    model_copy = StableDiffusionPipeline.from_pretrained(ori_model_path)
    model.unet.load_state_dict(torch.load(finetuned_model_path))
    model = model.to('cuda')
    model_copy = model_copy.to('cuda')
    
    if 'zero' in save_path:
        old_emb, adv_emb = close_form_emb_regzero(model, model_copy, concept, with_to_k, save_path, save_name, regeular_scale=reg_scale, seed=args.seed)
    else:
        old_emb, adv_emb = close_form_emb(model, model_copy, concept, with_to_k, save_path, old_target_concept=args.old_target_concept, regeular_scale=reg_scale, seed=args.seed, reg_item=reg_item, save_name=save_name)
    