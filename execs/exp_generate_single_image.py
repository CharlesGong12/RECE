from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
import random

def generate_image(prompt, seed, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=50, num_samples=1,cache_dir='./ldm_pretrained', ckpt=None, base='1.4'):
    if base=='1.4':
        dir_= '/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
    elif base=='2.1':
        dir_ = '/share/ckpt/gongchao/model_zoo/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
    else:
        raise ValueError('base should be 1.4 or 2.1')
    # dir_ = "stabilityai/stable-diffusion-2-1-base"     #for SD 2.1
    # dir_ = "stabilityai/stable-diffusion-2-base"       #for SD 2.0
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae",cache_dir=cache_dir)
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer",cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder",cache_dir=cache_dir)
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet",cache_dir=cache_dir)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    if ckpt is not None:
        unet.load_state_dict(torch.load(ckpt, map_location=device))

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    prompt = [prompt]*num_samples
    if seed is None:
        seed = random.randint(0, 2^32-1)
    print(seed)

    
    height =image_size # default height of Stable Diffusion
    width =image_size                         # default width of Stable Diffusion

    num_inference_steps = ddim_steps           # Number of denoising steps

    guidance_scale = guidance_scale            # Scale for classifier-free guidance

    generator = torch.cuda.manual_seed(seed)       # Seed generator to create the inital latent noise

    batch_size = len(prompt)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,device=torch_device
    )
    # latents = latents.to(torch_device)

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
    images = (image * 255).astype("uint8")
    # save the image
    pil_images = [Image.fromarray(image) for image in images]
    save_path = f'/share_io03_ssd/ckpt2/gongchao/SD_adv_train/advtrain_execs/execs_images/{prompt[0]}' if len(prompt[0])<15 else f'/share_io03_ssd/ckpt2/gongchao/SD_adv_train/advtrain_execs/execs_images/{prompt[0][:15]}'
    if ckpt is not None:
        save_path += '/uce'
    os.makedirs(save_path, exist_ok=True)
    for num, im in enumerate(pil_images):
        im.save(f"{save_path}/{num}_seed{seed}.png")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default=None, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run on')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='scale for classifier-free guidance')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--ddim_steps', type=int, default=50, help='number of denoising steps')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples')
    parser.add_argument('--cache_dir', type=str, default='./ldm_pretrained', help='path to cache dir')
    parser.add_argument('--ckpt', type=str, default=None, help='path to target checkpoint')
    parser.add_argument('--base', type=str, default='1.4', help='base version of SD')
    args = parser.parse_args()
    prompt = args.prompt
    seed = args.seed
    generate_image(prompt, seed, args.device, args.guidance_scale, args.image_size, args.ddim_steps, args.num_samples, args.cache_dir, args.ckpt, args.base)
    # seeds = [random.randint(0, 1000000) for _ in range(10)]
    # for seed in seeds:
    #     generate_image(prompt, seed, args.device, args.guidance_scale, args.image_size, args.ddim_steps, args.num_samples, args.cache_dir, args.ckpt, args.base)