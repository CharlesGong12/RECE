import torch
import random
import pandas as pd
from PIL import Image
import pandas as pd 
import argparse
import os, glob, json
from diffusers import StableDiffusionPipeline
import abc
import copy
from functools import reduce
import operator
import time
import ast
from tqdm import tqdm
import pickle
import sys
import math

from ..execs import generate_images, compute_nudity_rate

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def edit_model(ldm_stable, old_text_, new_text_, retain_text_, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    max_bias_diff = 0.05
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    print(f'CUDA memory allocated: {torch.cuda.memory_allocated()/1e9} GB')
    ######################## START ERASING ###################################
    for layer_num in tqdm(range(len(projection_matrices)), desc='Editing layers'):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                
                
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = text_embeddings[0]
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                new_emb = text_embeddings[1]
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]
                
                context = old_emb.detach()
                
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            
                            target = new_embs - (new_emb_proj)*u 
                            values.append(target.detach()) 
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concepts', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--guided_concepts', help='Concepts to guide the erased concepts towards', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    # parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--lamb', help='lambda for TIME init', type=float, required=False, default=0.5)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=True)
    parser.add_argument('--add_prompts', help='option to add additional prompts', type=bool, required=False, default=False)
    parser.add_argument('--epochs', help='number of epochs to train uce', type=int, required=False, default=1)
    parser.add_argument('--test_csv_path', help='path to csv file with prompts', type=str, default='/vhome/gongchao/UCE_acceleration/adv-train/dataset/nudity.csv')
    parser.add_argument('--save_path', help='path to save the images', type=str, default='/share/ckpt/gongchao/unified-concept-editing/twice')
    parser.add_argument('--ddim_steps', help='number of steps for ddim', type=int, default=50)

    args = parser.parse_args()
    technique = args.technique
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb=args.lamb
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = args.concepts.split(',')
    concepts = [con.strip() for con in concepts]
    concept_type = args.concept_type
    ddim_steps = args.ddim_steps
    print_text=''
    for txt in concepts:
        print_text += txt.lower()+'_'
    print_text  = print_text[:-1]
    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']

    if 'artists' in concepts[0]:
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', ''))
        concepts = random.sample(artists,number) 

    old_texts = []
    
    additional_prompts = []
    if concept_type == 'art':
        additional_prompts.append('painting by {concept}')
        additional_prompts.append('art by {concept}')
        additional_prompts.append('artwork by {concept}')
        additional_prompts.append('picture by {concept}')
        additional_prompts.append('style of {concept}')
    elif concept_type=='object':
        additional_prompts.append('image of {concept}')
        additional_prompts.append('photo of {concept}')
        additional_prompts.append('portrait of {concept}')
        additional_prompts.append('picture of {concept}')
        additional_prompts.append('painting of {concept}')  
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length)
    
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text+=f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con]*length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
            
    assert len(new_texts) == len(old_texts)
    
    
    if preserve_concepts is None:
        if concept_type == 'art':
            df = pd.read_csv('data/artists1734_prompts.csv')

            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)
        else:
            preserve_concepts = []

    retain_texts = ['']+preserve_concepts
    if len(retain_texts) > 1:
        print_text+=f'-preserve_true'     
    else:
        print_text+=f'-preserve_false'
    if preserve_scale is None:
        # set the format to be .3f
        preserve_scale = max(0.1, 1/len(retain_texts))
        preserve_scale = round(preserve_scale, 3)
    sd14="model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    sd21='model_zoo/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_version,
        )
    ldm_stable = ldm_stable.to(device)
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text += f"-erase_{erase_scale}"
    print_text += f"-preserve_{preserve_scale}"
    print_text += f"-lamb_{lamb}"
    print_text = print_text.lower()
    print(print_text)
    
    dev_df = pd.read_csv(args.test_csv_path)
    save_path = f'{args.save_path}/{print_text}'
    os.makedirs(save_path, exist_ok=True)
    
    nudity_rates = {}
    # validation before loading UCE
    generate_images(ldm_stable, dev_df, f'{save_path}/before', ddim_steps=ddim_steps, num_samples=1)
    nudity_rate = compute_nudity_rate(f'{save_path}/before')
    print(f'Nudity rate of the original model: {nudity_rate}')
    nudity_rates['before'] = nudity_rate
    lowest_nudity_rate = nudity_rate
    
    # running time
    start = time.time()
    for epoch in range(args.epochs):
        ldm_stable = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=add_prompts, retain_text_= retain_texts, lamb=lamb, erase_scale = erase_scale, preserve_scale = preserve_scale,  technique=technique)
        if epoch ==0:
            # validation of UCE
            generate_images(ldm_stable, dev_df, f'{save_path}/uce', ddim_steps=ddim_steps, num_samples=1)
            nudity_rate = compute_nudity_rate(f'{save_path}/uce')
            print(f'Nudity rate of uce: {nudity_rate}')
            nudity_rates['uce'] = nudity_rate
            if nudity_rate < lowest_nudity_rate:
                lowest_nudity_rate = nudity_rate
        
        # validation
        generate_images(ldm_stable, dev_df, f'{save_path}/epoch-{epoch}', ddim_steps=ddim_steps, num_samples=1)
        nudity_rate = compute_nudity_rate(f'{save_path}/epoch-{epoch}')
        print(f'epoch: {epoch}, nudity rate: {nudity_rate}')
        nudity_rates[epoch] = nudity_rate
        if nudity_rate < lowest_nudity_rate:
            lowest_nudity_rate = nudity_rate
            print(f'Lowest nudity rate: {lowest_nudity_rate}. Saving model.')
            torch.save(ldm_stable.unet.state_dict(), f'{save_path}/{lowest_nudity_rate}-{epoch}.pt')
            print('Model saved.') 
        with open(f'{save_path}/nudity.json', 'w') as f:
            json.dump(nudity_rates, f)
            f.flush()

    end = time.time()
    print(f'Running time: {end-start}')
    
    save_path = '/share/ckpt/gongchao/unified-concept-editing'
    print('Saving model...')
    os.makedirs(save_path, exist_ok=True)
    torch.save(ldm_stable.unet.state_dict(), f'{save_path}/erased-{print_text}-{erase_scale}-{preserve_scale}.pt')
    print('Model saved.')