import os
import pandas as pd

from torchmetrics.functional.multimodal import clip_score
from PIL import Image
from torchvision.transforms import functional as F
import argparse
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import re
import math


def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def clip_score_cumstom(folder_path, prompts_path, model_version="large", batch_size=32):
    if model_version == "base":
        model_path = "/share/ckpt/gongchao/model_zoo/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb"
    elif model_version == "large":
        model_path = "/share/ckpt/gongchao/model_zoo/clip-vit-large-patch14"

    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)

    csv_path = prompts_path
    im_folder = folder_path
    df = pd.read_csv(csv_path)
    images = os.listdir(im_folder)
    # images = sorted_nicely(images)
    ratios = {}
    df['clip'] = np.nan
    # # use batch processing
    # images = [image for image in os.listdir(im_folder) if image.endswith('.png')]
    # num_batches = math.ceil(len(images) / batch_size)
    # for batch_idx in tqdm(range(num_batches)):
    #     start_idx = batch_idx * batch_size
    #     end_idx = min((batch_idx + 1) * batch_size, len(images))
    #     batch_images = images[start_idx:end_idx]
        
    #     batch_cases = []
    #     batch_captions = []
    #     for image in batch_images:
    #         case_number = int(image.split('_')[0].replace('.png', ''))
    #         if case_number in list(df['case_number']):
    #             caption = df.loc[df.case_number == case_number]['prompt'].item()
    #             batch_cases.append(case_number)
    #             batch_captions.append(caption)
        
    #     if not batch_cases:
    #         continue
        
    #     batch_images_tensors = [Image.open(os.path.join(im_folder, image)) for image in batch_images]
    #     inputs = processor(text=batch_captions, images=batch_images_tensors, return_tensors="pt", padding=True)
    #     outputs = model(**inputs)
        
    #     for i, case_number in enumerate(batch_cases):
    #         clip_score = outputs.logits_per_image[i][0].detach().cpu()
    #         ratios[case_number] = ratios.get(case_number, []) + [clip_score]
    for image in tqdm(images):
        if not image.endswith('.png'):
            continue
        case_number = int(image.split('_')[0].replace('.png',''))
        if case_number not in list(df['case_number']):
            continue
        caption = df.loc[df.case_number==case_number]['prompt'].item()
        im = Image.open(os.path.join(im_folder, image))
        inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
        ratios[case_number] = ratios.get(case_number, []) + [clip_score]
    for key in ratios.keys():
        df.loc[df.case_number==key,'clip'] = np.mean(ratios[key])
        # df.loc[key,'clip'] = np.mean(ratios[key])
    df = df.dropna(axis=0)
    print(f"Mean CLIP score: {df['clip'].mean()}")
    print('-------------------------------------------------')
    print('\n')
    return df, df['clip'].mean()

# def clip_score_cumstom(folder_path, prompts_path, model_version="large"):
#     prompts_df = pd.read_csv(prompts_path)
#     total_clip_score = 0.0
#     total_count = 0
#     if model_version == "base":
#         model_path = "/share/ckpt/gongchao/model_zoo/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb"
#     elif model_version == "large":
#         model_path = "/share/ckpt/gongchao/model_zoo/clip-vit-large-patch14"
#     output_list = []
    
#     images = []
#     prompt_texts = []
#     for filename in tqdm(os.listdir(folder_path)):
#         if filename.endswith(".png"):  
#             image_path = os.path.join(folder_path, filename)
#             image = Image.open(image_path).convert("RGB")
#             image = F.to_tensor(image)
#             images.append(image)

#             image_id = int(filename.split("_")[0])  # "21_0.png" -> 21
#             case_number_row = prompts_df[prompts_df["case_number"] == image_id]
#             prompt_text = case_number_row["prompt"].values[0]
#             prompt_texts.append(prompt_text)
#     score_ls = clip_score(images, prompt_texts, model_name_or_path=model_path)
#     mean_score = score_ls.mean().item()
#     print(f'CLIP score of {folder_path} is {mean_score}')

#     return score_ls, mean_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="adv_images")
    parser.add_argument('--model_version', type=str, choices=['base', 'large'], default='base')
    parser.add_argument("--prompts_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=96)
    args = parser.parse_args()
    folder_path = args.folder_path
    prompts_path = args.prompts_path
    batch_size = args.batch_size
    if prompts_path is None:
        prompts_path = os.path.join(folder_path, "prompts.csv")
    if not os.path.exists(prompts_path):
        raise Exception("prompts_path not exists")
    
    if os.path.exists(os.path.join(folder_path, "imgs")):
        img_folder = os.path.join(folder_path, "imgs")
    else:
        img_folder = folder_path
    
    clip_df, avg_clip_score = clip_score_cumstom(img_folder, prompts_path, args.model_version, batch_size)
    print(f"Average CLIP score of {img_folder}: {avg_clip_score}")
    # add a row to the clip_df
    # case_number,source,prompt,evaluation_seed,coco_id,clip
    clip_df = clip_df.append({"case_number": "average", "clip": avg_clip_score}, ignore_index=True)
    clip_df.to_csv(os.path.join(folder_path, "clip_scores.csv"), index=False)
