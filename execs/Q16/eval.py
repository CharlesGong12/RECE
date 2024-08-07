from main.paper_experiments.experiments import run_model_imagefolder
import argparse
from argparse import Namespace
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Q16')
    parser.add_argument('--path', type=str, help='data path')
    args = parser.parse_args()
    path = args.path
    q16_language_model_path = "model_zoo/ViT-L-14.pt"
    # finetuned prompt which is more precise, taken from https://github.com/YitingQu/unsafe-diffusion
    q16_prompt_path = 'execs/Q16/data/ViT-L-14/prompts_finetuned.pt'
    q16_args = Namespace(
            gpu=[0],
            language_model=q16_language_model_path,
            model_type='sim',
            prompt_path=q16_prompt_path,
            only_inappropriate=False,
            input_type='img',
            )
    try:
        detect_rate = run_model_imagefolder(q16_args, f'{path}/imgs', f'{path}')
    except:
        detect_rate = run_model_imagefolder(q16_args, f'{path}', f'{path}')
    print(f'overall_detect_rate: {detect_rate}')
    # columns: non inappropriate,0,0.58,0_0.png
    inappropriate_path = f'{path}/inappropriate/ViT-L-14/sim_prompt_tuned_data/inappropriate_images.csv'
    # columns: Unnamed: 0,prompt,categories,hard,inappropriate_percentage,nudity_percentage,q16_percentage,sd_safety_percentage,prompt_toxicity,lexica_url,evaluation_seed,evaluation_guidance,sd_image_width,sd_image_height,sd_model,case_number
    prompts_path = f'{path}/prompts.csv'
    # compute inappropriate rate of hate, harassment, violence, self-harm, sexual, shocking, and illegal activity, respectively
    inappropriate_rate = {}
    try:
        prompts_file = pd.read_csv(prompts_path)
    except:
        prompts_file = pd.read_csv('dataset/i2p.csv')
    inappropriate_file = pd.read_csv(inappropriate_path, header=None)
    for cls in ['hate', 'harassment', 'violence', 'self-harm', 'sexual', 'shocking', 'illegal activity']:
        inappropriate_rate[cls] = 0
        category_num = 0
        for i in range(len(prompts_file)):
            if cls in prompts_file.loc[i, 'categories'].lower():
                inappropriate_rate[cls] = inappropriate_rate[cls] + inappropriate_file.iloc[i, 1]    # ith row, 1st column
                category_num = category_num + 1
        inappropriate_rate[cls] = inappropriate_rate[cls] / category_num if category_num != 0 else None
    print(inappropriate_rate)
    # save the results and overall detect rate
    with open(f'{path}/inappropriate/ViT-L-14/sim_prompt_tuned_data/rates.txt', 'w') as f:
        f.write(f'overall_detect_rate: {detect_rate}\n')
        for cls in inappropriate_rate:
            f.write(f'{cls}: {inappropriate_rate[cls]}\n')
