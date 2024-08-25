import torch
import random
import pandas as pd
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
from PIL import Image
from attack_methods import *
from execs import generate_images
from utils import list_Image_to_tensor, generate_latents
from execs import compute_nudity_rate

def setup_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def embedding2img(embeddings, df, ldm_stable, save_path, save_name=None, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=50, num_samples=1, from_case=0, to_case=None):
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = ldm_stable.vae
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    # 3. The UNet model for generating the latents.
    unet = ldm_stable.unet
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    folder_path = save_path
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f'{folder_path}/emb2imgs', exist_ok=True)
    
    repeated_rows = []
    for i in range(len(embeddings)):
        row = df.iloc[i]
        seed = row.evaluation_seed if hasattr(row,'evaluation_seed') else row.sd_seed
        case_number = row.case_number if hasattr(row,'case_number') else i
        repeated_rows.extend([row]*num_samples)
        if case_number<from_case:
            continue
        if to_case is not None and case_number>=to_case:
            break
        
        height = row.sd_image_height if hasattr(row, 'sd_image_height') else image_size # default height of Stable Diffusion
        width = row.sd_image_width if hasattr(row, 'sd_image_width') else image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        # TODO: different attributes name for different datasets
        guidance_scale = row.evaluation_guidance if hasattr(row, 'evaluation_guidance') else guidance_scale            # Scale for classifier-free guidance

        generator = torch.cuda.manual_seed(seed)        # Seed generator to create the inital latent noise

        batch_size = num_samples

        text_embeddings = [embeddings[i]]*num_samples
        text_embeddings = torch.stack(text_embeddings)

        max_length = embeddings.shape[1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,device=torch_device
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            if save_name is None:
                im.save(f"{folder_path}/emb2imgs/{case_number}_{num}.png")
            else:
                im.save(f"{folder_path}/emb2imgs/{save_name}_{num}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--epochs', help='epochs to train', type=int, required=False, default=2)
    parser.add_argument('--batch_size', help='batch size', type=int, required=False, default=1)
    parser.add_argument('--concepts', help='target concepts', type=str, required=False, default=None)
    parser.add_argument('--seed', help='seed should be assigned if concepts are not None', type=int, required=False, default=123)
    parser.add_argument('--num_adv_prompts', help='number of advesarial prompts to use', type=int, required=False, default=0)
    parser.add_argument('--csv_path', help='path to csv file with prompts', type=str, default='/vhome/gongchao/UCE_acceleration/adv-train/dataset/sexual_all_i2p_dropUnlearnDiff.csv')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--target_ckpt', help='target checkpoint to load, according to version --base', type=str, required=False, default='/share/ckpt/gongchao/unified-concept-editing/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt')
    parser.add_argument('--save_path', help='path to save the images', type=str, required=False, default='/share/ckpt/gongchao/advtrain_execs/adv_embedding_images')
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=True)
    parser.add_argument('--loss_type', help='type of loss function', type=str, required=False, default='unlearndiff', choices=['unlearndiff', 'p4d'])
    parser.add_argument('--ddim_steps', help='number of steps for ddim', type=int, required=False, default=50)

    args = parser.parse_args()
    seed_shuffle=123
    setup_seed(seed_shuffle)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs
    batch_size = args.batch_size
    if args.num_adv_prompts != 0:
        df = pd.read_csv(args.csv_path)
        df = df.sample(frac=1, random_state=seed_shuffle).reset_index(drop=True)
        all_concepts = df['prompt'].tolist()
        all_concepts = [con.strip() for con in all_concepts]
        adv_concepts = all_concepts[:args.num_adv_prompts]
        adv_df = df.iloc[:args.num_adv_prompts]
    else:
        adv_concepts = args.concepts
        adv_concepts = adv_concepts.split(',')
        adv_concepts = [con.strip() for con in adv_concepts]
        seed = args.seed
        # create a dataframe
        adv_df = pd.DataFrame(columns=['prompt', 'evaluation_seed'])
        adv_df['prompt'] = adv_concepts
        adv_df['evaluation_seed'] = seed

    concept_type = args.concept_type
    ddim_steps = args.ddim_steps
    sd14="/vhome/gongchao/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    sd21='/vhome/gongchao/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
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
    # copy an independent ldm_stable
    ldm_stable_copy = StableDiffusionPipeline.from_pretrained(
        model_version,
        )
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")

    target_ckpt = args.target_ckpt

    print_text=''
    if args.num_adv_prompts != 0:
        print_text+=f'fromcsv_{concept_type}'
    else:
        print_text+=f'assigned_{adv_concepts[0]}'
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text = print_text.lower()
    print(print_text)
    # running time
    start = time.time()
    
    # TODO
    # adv_train_steps = 1000
    # n_samples = 20
    # start_t = 10
    adv_train_steps = 1000
    n_samples = 50
    start_t = 5
    sampled_t = [start_t + i * adv_train_steps // n_samples for i in range(n_samples)]
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=adv_train_steps)
    save_path = f'{args.save_path}/{concept_type}/{print_text}'
    os.makedirs(save_path, exist_ok=True)
    adv_df.to_csv(f'{save_path}/adv_df.csv', index=False)

    # nudity_rates = {}
    # # validation before loading UCE
    # generate_images(ldm_stable, adv_df, f'{save_path}/before', ddim_steps=ddim_steps, num_samples=1)
    # nudity_rate = compute_nudity_rate(f'{save_path}/before')
    # print(f'Nudity rate of the original model: {nudity_rate}')
    # nudity_rates['before'] = nudity_rate

    # lowest_nudity_rate = nudity_rate

    # # validation of UCE
    if target_ckpt != '':
        ldm_stable.unet.load_state_dict(torch.load(target_ckpt))
    ldm_stable.to(device)
    # generate_images(ldm_stable, adv_df, f'{save_path}/uce', ddim_steps=ddim_steps, num_samples=1)
    # nudity_rate = compute_nudity_rate(f'{save_path}/uce')
    # print(f'Nudity rate of uce: {nudity_rate}')
    # nudity_rates['uce'] = nudity_rate
    # if nudity_rate < lowest_nudity_rate:
    #     lowest_nudity_rate = nudity_rate
    if args.loss_type == 'unlearndiff':
        # adv_embeddings = pgd_attack_UnlearnDiff(ldm_stable, adv_embeddings, input_ids, x0, t, scheduler)
        # use Adam to optimize adv_embeddings
        loss_function = UnlearnDiff_loss_function
    elif args.loss_type == 'p4d':
        loss_function = P4D_loss_function

    for i in tqdm.tqdm(range(0, len(adv_concepts), batch_size), desc='Batch'):
        # batch size is 1
        batch_df = adv_df.iloc[i:i+batch_size]
        batch_df = batch_df.reset_index(drop=True)
        batch_concepts = adv_concepts[i:i+batch_size]
        # tokenize
        id_concepts = tokenizer(batch_concepts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        input_embeddings = ldm_stable.text_encoder(id_concepts)[0]
        adv_embeddings = input_embeddings.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([adv_embeddings], lr=0.01)
        input_ids = id_concepts
        with torch.no_grad():
            x0 = generate_latents(ldm_stable_copy, batch_df, device=device, num_samples=1, ddim_steps=ddim_steps)[0]
        for epoch in range(epochs):
            if epoch==epochs-1:
                sampled_t = [5,25]
            for t in sampled_t:
                for i in range(10):
                    loss = -loss_function(ldm_stable, adv_embeddings, x0, t, input_ids, scheduler)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if i==9:
                        print(f't: {t}/{adv_train_steps}, loss: {loss}')
                embedding2img(adv_embeddings, batch_df, ldm_stable, f'{save_path}/epoch_{epoch}/t_{t}', device=device, ddim_steps=ddim_steps, num_samples=1)
            # save the adv_embeddings
            torch.save(adv_embeddings, f'{save_path}/epoch_{epoch}/adv_embeddings.pt')
    end = time.time()
    print(f'Running time: {end-start}')