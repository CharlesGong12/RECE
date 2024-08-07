import torch
import random
import pandas as pd
import argparse
import os
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTokenizer
from functools import reduce
import operator
import time
import tqdm
import json
import numpy as np
import pickle
from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights
from PIL import Image


from erase_methods import edit_model_adversarial
from attack_methods import *
from execs import generate_images
from utils import generate_latents
from utils.embedding_calculation import close_form_emb, close_form_emb_regzero

def setup_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def image_classify(images_path, prompts_path, save_path, target_class, device='cuda', topk=1, batch_size=200):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()

    scores = {}
    categories = {}
    indexes = {}
    for k in range(1,topk+1):
        scores[f'top{k}']= []
        indexes[f'top{k}']=[]
        categories[f'top{k}']=[]

    names = os.listdir(images_path)
    names = [name for name in names if '.png' in name or '.jpg' in name]
    if len(names) == 0:
        images_path = images_path+'/imgs'
        names = os.listdir(images_path)
        names = [name for name in names if '.png' in name or '.jpg' in name]

    preprocess = weights.transforms()

    images = []
    for name in names:
        img = Image.open(os.path.join(images_path,name))
        batch = preprocess(img)
        images.append(batch)

    if batch_size == None:
        batch_size = len(names)
    if batch_size > len(names):
        batch_size = len(names)
    images = torch.stack(images)
    # Step 4: Use the model and print the predicted category
    for i in range(((len(names)-1)//batch_size)+1):
        batch = images[i*batch_size: min(len(names), (i+1)*batch_size)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim = 1)

        for k in range(1,topk+1):
            scores[f'top{k}'].extend(probs[:,k-1].detach().cpu().numpy())
            indexes[f'top{k}'].extend(class_ids[:,k-1].detach().cpu().numpy())
            categories[f'top{k}'].extend([weights.meta["categories"][idx] for idx in class_ids[:,k-1].detach().cpu().numpy()])

    if save_path is not None:
        df = pd.read_csv(prompts_path)
        df['case_number'] = df['case_number'].astype('int')
        case_numbers = []
        for i, name in enumerate(names):
            case_number = name.split('/')[-1].split('_')[0].replace('.png','').replace('.jpg','')
            case_numbers.append(int(case_number))

        dict_final = {'case_number': case_numbers}

        for k in range(1,topk+1):
            dict_final[f'category_top{k}'] = categories[f'top{k}'] 
            dict_final[f'index_top{k}'] = indexes[f'top{k}'] 
            dict_final[f'scores_top{k}'] = scores[f'top{k}'] 

        df_results = pd.DataFrame(dict_final)
        merged_df = pd.merge(df,df_results)
        merged_df.to_csv(save_path)

        # compute the accuracy of the target class and others
        target_acc = 0
        other_acc = 0
        for i in range(len(merged_df)):
            if merged_df['category_top1'][i].lower() == merged_df['class'][i] and merged_df['class'][i] == target_class:
                target_acc += 1
            elif merged_df['category_top1'][i].lower() == merged_df['class'][i] and merged_df['class'][i] != target_class:
                other_acc += 1
        target_acc /= (len(merged_df)/10.)  # imagenette has 10 classes
        other_acc /= (9*len(merged_df)/10.)
        return target_acc, other_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concepts', help='concepts to erase', type=str, required=True)
    parser.add_argument('--old_target_concept', help='old target concept ever used in UCE', type=str, required=False, default=None)
    parser.add_argument('--seed', help='random seed', type=int, required=False, default=42)
    parser.add_argument('--epochs', help='epochs to train', type=int, required=False, default=5)
    parser.add_argument('--test_csv_path', help='path to csv file with prompts', type=str, default='dataset/small_imagenet_prompts.csv')
    parser.add_argument('--guided_concepts', help='whether to use old prompts to guide', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='whether to preserve old prompts', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--target_ckpt', help='target checkpoint to load, UCE', type=str, required=False, default="ckpt2/UCE/erased-tench-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt")
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=0.1)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='scale for init', type=float, required=False, default=0.1)
    parser.add_argument('--save_path', help='path to save the model', type=str, required=False, default='ckpt2/SD_adv_train')
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=True)
    parser.add_argument('--emb_computing', help='close-form or gradient-descent, standard regularization or surrogate regularization', type=str, required=False, default='close_regzero', choices=['close_standardreg', 'close_surrogatereg', 'gd', 'close_regzero'])
    parser.add_argument('--reg_item', help='use 1st, 2nd or both items in surrogate regularization', type=str, required=False, default='1st', choices=['1st', '2nd','both'])
    parser.add_argument('--regular_scale', help='scale for regularization', type=float, required=False, default=1e-3)
    parser.add_argument('--num_samples', help='number of samples for gradient descent', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='number of steps for ddim', type=int, required=False, default=50)

    args = parser.parse_args()
    seed_shuffle=123
    setup_seed(seed_shuffle)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    concepts = args.concepts.split(',')
    concepts = [con.strip() for con in concepts]

    if args.old_target_concept is None:
        old_target_concept = [None for _ in concepts]
    else:
        old_target_concept = args.old_target_concept.split(',')
        old_target_concept = [con.strip() for con in old_target_concept]
        for idx, con in enumerate(old_target_concept):
            if con == 'none':
                old_target_concept[idx] = None
            if con == '':
                old_target_concept[idx] = ' '
    assert len(old_target_concept) == len(concepts), f'length of old_target_concept {len(old_target_concept)} should be the same as concepts {len(concepts)}'

    seed = args.seed
    epochs = args.epochs
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    technique = args.technique
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb
    preserve_number = args.preserve_number
    concept_type = args.concept_type
    emb_computing = args.emb_computing
    reg_item = args.reg_item
    regular_scale = args.regular_scale
    num_samples = args.num_samples
    ddim_steps = args.ddim_steps
    sd14="/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    sd21='/share/ckpt/gongchao/model_zoo/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_version,
        )
    ldm_stable_copy = StableDiffusionPipeline.from_pretrained(
        model_version,
        )
    ldm_stable = ldm_stable.to(device)
    ldm_stable_copy = ldm_stable_copy.to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")

    target_ckpt = args.target_ckpt
    dev_df = pd.read_csv(args.test_csv_path)

    print_text=''
    for concept in concepts:
        print_text+=f'{concept}_'

    # PROMPT CLEANING
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']

    # create a new df similar to prompts_df, using concepts and seed
    # It should contain prompt, evaluation_seed
    adv_df = pd.DataFrame(columns=['prompt', 'evaluation_seed'])
    for concept in concepts:
        adv_df = adv_df.append({'prompt':concept, 'evaluation_seed':args.seed}, ignore_index=True)


    old_texts = []
    for concept in concepts:
        old_texts.append(f'{concept}')
    
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text+=f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con] for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
            
    assert len(new_texts) == len(old_texts)
    
    
    if preserve_concepts is None:
        preserve_concepts = []
    if type(preserve_concepts) == str:
        preserve_concepts = [con.strip() for con in preserve_concepts.split(',')]
    retain_texts = ['']+preserve_concepts
    if len(retain_texts) > 1:
        print_text+=f'-preserve_true'     
    else:
        print_text+=f'-preserve_false'
    if preserve_scale is None:
        # set the format to be .3f
        preserve_scale = max(0.1, 1/len(retain_texts))
        preserve_scale = round(preserve_scale, 3)

    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text += f"-erase_{erase_scale}"
    print_text += f"-preserve_{preserve_scale}"
    print_text += f"-lamb_{lamb}"
    print_text = print_text.lower()
    print(print_text)
    
    # Below is for gradient descent
    # TODO
    # adv_train_steps = 100
    adv_train_steps = 1000
    n_samples = 50 
    start_t = 5
    sampled_t = [start_t + i * adv_train_steps // n_samples for i in range(n_samples)]

    
    
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=adv_train_steps)
    
    if 'close' in emb_computing:
        if 'surrogate' in emb_computing:
            save_path = f'{args.save_path}/{concept_type}/{emb_computing}_regitem_{reg_item}/{print_text}/regular_{regular_scale}/seed_{seed}'
        else:
            save_path = f'{args.save_path}/{concept_type}/{emb_computing}/{print_text}/regular_{regular_scale}/seed_{seed}'
    elif 'gd' in emb_computing:
        save_path = f'{args.save_path}/{concept_type}/{emb_computing}/{print_text}/seed_{seed}'
    os.makedirs(save_path, exist_ok=True)

    target_acc = {}
    other_acc = {}

    # generate_images(ldm_stable, dev_df, f'{save_path}/sd', ddim_steps=ddim_steps, num_samples=num_samples)
    if len(old_texts) == 1:
        os.makedirs(f'{save_path}/sd', exist_ok=True)
        target_acc['sd'], other_acc['sd'] = image_classify("/share_io03_ssd/ckpt2/gongchao/SD_adv_train/object/close_regzero/tench_-towards_uncond-preserve_false-sd_1_4-method_replace-erase_1-preserve_0.1-lamb_0.1/regular_0.001/seed_42/sd",
                                                            args.test_csv_path, f'{save_path}/sd/classification.csv', old_texts[0].lower())
        print(f'SD: target {target_acc["sd"]}, others {other_acc["sd"]}')
    # TODO: add the case for multiple concepts

    # load UCE model
    if target_ckpt != '':
        ldm_stable.unet.load_state_dict(torch.load(target_ckpt))
    ldm_stable.to(device)
    generate_images(ldm_stable, dev_df, f'{save_path}/uce', ddim_steps=ddim_steps, num_samples=num_samples)
    if len(old_texts) == 1:
        target_acc['uce'], other_acc['uce'] = image_classify(f'{save_path}/uce', args.test_csv_path, f'{save_path}/uce/classification.csv', old_texts[0].lower())
        print(f'UCE: target {target_acc["uce"]}, others {other_acc["uce"]}')

    start = time.time()

    for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
        adv_emb_list = []
        new_emb_list = []
        for i in range(0, len(old_texts)):
            # batch size is 1
            batch_df = adv_df.iloc[i:i+1]
            batch_concept = old_texts[i]
            batch_old_target_concept = old_target_concept[i]
            batch_new_text = new_texts[i]
            
            # tokenize
            id_concept = tokenizer(batch_concept, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            id_new_text = tokenizer(batch_new_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            
            # get embeddings
            input_embedding = ldm_stable.text_encoder(id_concept)[0]
            new_embedding = ldm_stable.text_encoder(id_new_text)[0]
            new_emb_list.append(new_embedding[0])   # squeeze the batch dimension
            
            input_ids = id_concept
            if 'close' in emb_computing:
                if 'surrogate' in emb_computing:
                    _, adv_embedding = close_form_emb(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, old_target_concept=None, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}', reg_item=reg_item)
                elif 'standard' in emb_computing:
                    _, adv_embedding = close_form_emb(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, old_target_concept=batch_old_target_concept, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}')
                elif 'regzero' in emb_computing:
                    _, adv_embedding = close_form_emb_regzero(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}')
            else:
                raise NotImplementedError
            adv_emb_list.append(adv_embedding[0])   # squeeze the batch dimension

        ldm_stable = edit_model_adversarial(ldm_stable, adv_emb_list, new_emb_list, retain_texts, technique=technique, preserve_scale=preserve_scale, erase_scale=erase_scale, lamb=lamb)
        generate_images(ldm_stable, dev_df, f'{save_path}/epoch_{epoch}', ddim_steps=ddim_steps, num_samples=num_samples)
        if len(old_texts) == 1:
            target_acc[epoch], other_acc[epoch] = image_classify(f'{save_path}/epoch_{epoch}', args.test_csv_path, f'{save_path}/epoch_{epoch}/classification.csv', old_texts[0].lower())
            if target_acc[epoch]==0:
                print(f'Epoch {epoch}: target accuracy is 0, others\' accuracy is {other_acc[epoch]}. Break now!')
                torch.save(ldm_stable.unet.state_dict(), f'{save_path}/ep_{epoch}_tar_{target_acc[epoch]}_oth_{other_acc[epoch]}.pt')
                break
            if target_acc[epoch] < target_acc['uce'] and other_acc[epoch] >= other_acc['uce']-0.1:
                print(f'Epoch {epoch}: lower target accuracy: {target_acc[epoch]}, higher other accuracy: {other_acc[epoch]}')
                torch.save(ldm_stable.unet.state_dict(), f'{save_path}/ep_{epoch}_tar_{target_acc[epoch]}_oth_{other_acc[epoch]}.pt')
            else:
                print(f'Epoch {epoch}: target {target_acc[epoch]}, others {other_acc[epoch]}')

    end = time.time()
    print(f'Running time: {end-start}')
    print(f'Running time per epoch: {(end-start)/epochs}')

    with open(f'{save_path}/target_acc.json', 'w') as f:
        json.dump(target_acc, f)
    with open(f'{save_path}/other_acc.json', 'w') as f:
        json.dump(other_acc, f)
