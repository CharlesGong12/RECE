import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms

import pandas as pd
from PIL import Image
import pandas as pd
import os
from diffusers import LMSDiscreteScheduler


def generate_latents(model, df, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=1, from_case=0):
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = model.vae
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    # 3. The UNet model for generating the latents.
    unet = model.unet
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    all_latents = []
    
    repeated_rows = []
    for i, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed if hasattr(row,'evaluation_seed') else row.sd_seed
        case_number = row.case_number if hasattr(row,'case_number') else i
        repeated_rows.extend([row]*num_samples)
        if case_number<from_case:
            continue
        
        height = row.sd_image_height if hasattr(row, 'sd_image_height') else image_size # default height of Stable Diffusion
        width = row.sd_image_width if hasattr(row, 'sd_image_width') else image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        # TODO: different attributes name for different datasets
        guidance_scale = row.evaluation_guidance if hasattr(row, 'evaluation_guidance') else guidance_scale            # Scale for classifier-free guidance

        generator = torch.cuda.manual_seed(seed)        # Seed generator to create the inital latent noise

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
            
        # append the latents
        # (batch_size, channels, h, w)
        all_latents.append(latents[0])
        
    return torch.stack(all_latents)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def list_Image_to_tensor(list_image):
    transform = get_transform()
    list_tensor = []
    for image in list_image:
        image = transform(image)
        list_tensor.append(image)
    return torch.stack(list_tensor)