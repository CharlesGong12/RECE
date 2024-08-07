import argparse
import csv
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import sys
from ..utils.embedding_calculation import embedding2img
from ..execs.compute_nudity_rate import compute_nudity_rate
import pandas as pd
import random
import numpy as np
import os
import json

def ringabell(csv_file, output_path, ldm_stable, concept, num_prompts=None, seed=42, batch_size=8):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        rows = rows[1:]
        if num_prompts is not None:
            random.seed(seed)
            rows = random.sample(rows, num_prompts)

    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder.to('cuda')

    abnormal_prompts = [row[0].strip() for row in rows]
    normal_prompts = [row[1].strip() for row in rows]

    abstract_embeddings_list = []

    for i in range(0, len(abnormal_prompts), batch_size):
        batch_abnormal_prompts = abnormal_prompts[i:i+batch_size]
        batch_normal_prompts = normal_prompts[i:i+batch_size]

        abnormal_tokens = tokenizer(batch_abnormal_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        normal_tokens = tokenizer(batch_normal_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
            abnormal_embeddings = text_encoder(abnormal_tokens.input_ids.to('cuda')).last_hidden_state
            normal_embeddings = text_encoder(normal_tokens.input_ids.to('cuda')).last_hidden_state

        abstract_embeddings = abnormal_embeddings - normal_embeddings
        abstract_embeddings = abstract_embeddings.mean(dim=0)
        abstract_embeddings_list.append(abstract_embeddings.cpu().numpy())

        # Free up GPU memory
        del abnormal_tokens, normal_tokens, abnormal_embeddings, normal_embeddings
        torch.cuda.empty_cache()

    abstract_embeddings = np.stack(abstract_embeddings_list)
    abstract_embeddings = abstract_embeddings.mean(axis=0)
    abstract_embeddings = torch.from_numpy(abstract_embeddings).unsqueeze(0).to('cuda')
    os.makedirs(output_path, exist_ok=True)
    torch.save(abstract_embeddings, output_path + f"/{concept}_embedding_ring_a_bell.pt")

    return abstract_embeddings

if __name__ == "__main__":
    # read the prompts pairs from csv file, use CLIP to encode, subtract the embeddings respectively, and save the results
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="/vhome/gongchao/UCE_acceleration/adv-train/dataset/nudity-ring-a-bell.csv", help='csv file for abstract embedding calculation')
    parser.add_argument('--prompt_file', type=str, default='/vhome/gongchao/UCE_acceleration/adv-train/dataset/nudity.csv', help='the file containing the prompts')
    parser.add_argument("--output_path", type=str, default="/share/ckpt/gongchao/SD_adv_train/ring_a_bell", help='the path to save the abstract embedding')
    parser.add_argument('--img_path', type=str, default='/share/ckpt/gongchao/SD_adv_train/ring_a_bell', help='the path to save images')
    parser.add_argument('--SD_path', type=str, default='/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument('--eta', type=float, default=3, help='the eta value in the RingABell attack')   # 3 for nudity, 0.9 for Van Gogh
    parser.add_argument('--num_prompts', type=int, default=None, help='the number of prompts to be used for abstract embedding calculation')
    parser.add_argument('--concept', type=str, default='nudity', help='the concept to be attacked, used to name the output file.')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--ckpt', type=str, default=None, help='target defense SD')
    parser.add_argument('--from_case', type=int, default=0, help='the case number to start from')
    parser.add_argument('--to_case', type=int, default=None, help='the case number to end at, not included')
    parser.add_argument('--emb_path', help='path to the abstract embedding', default=None, type=str)

    args = parser.parse_args()
    csv_file = args.csv_file
    prompt_file = args.prompt_file
    output_path = args.output_path
    img_path = args.img_path
    SD_path = args.SD_path
    eta = args.eta
    num_prompts = args.num_prompts
    concept = args.concept
    seed = args.seed
    ckpt = args.ckpt
    from_case = args.from_case
    ldm_stable = StableDiffusionPipeline.from_pretrained(SD_path).to('cuda')
    if ckpt is not None:
        if ckpt.endswith('pt'):
            try:
                ldm_stable.unet.load_state_dict(torch.load(ckpt))
            except:
                ckpt_load = torch.load(ckpt)['state_dict']
                print(ckpt_load.keys())
                ldm_stable.unet.load_state_dict(ckpt_load, strict=False)
        else:
            try:
                ldm_stable.unet = UNet2DConditionModel.from_pretrained(ckpt)
            except:
                raise ValueError('edited model not found')

    if args.emb_path is None:
        abs_emb = ringabell(csv_file, output_path, ldm_stable, concept, num_prompts, seed)
    else:
        abs_emb = torch.load(args.emb_path)

    # read the prompt from the prompt file, use the abstract embedding to generate the adversarial prompt, and save the result
    df = pd.read_csv(prompt_file)
    for i in range(len(df)):
        prompt = df['prompt'][i]
        tokens = ldm_stable.tokenizer(prompt, padding="max_length", max_length=ldm_stable.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        embedding = ldm_stable.text_encoder(tokens.input_ids.to('cuda')).last_hidden_state
        adv_emb = embedding + abs_emb * eta

        row_df = df.iloc[i:i+1]
        embedding2img(adv_emb, row_df, ldm_stable, img_path, from_case=from_case, to_case=args.to_case)

    nudity_rate = compute_nudity_rate(img_path)
    print(f"nudity rate: {nudity_rate}")