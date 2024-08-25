import torch
import pandas as pd
from PIL import Image
import pandas as pd
import os
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
import argparse
import sys
from execs import compute_nudity_rate

def generate_images(model, df, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=50, num_samples=1, from_case=0):
        
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
    folder_path = f'{save_path}'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f'{folder_path}/imgs', exist_ok=True)
    
    repeated_rows = []
    for i, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples if not hasattr(row,'adv_prompt') else [str(row.adv_prompt)]*num_samples
        try:
            seed = row.evaluation_seed if hasattr(row,'evaluation_seed') else row.sd_seed
        except:
            seed = 42
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

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/imgs/{case_number}_{num}.png")
    new_df = pd.DataFrame(repeated_rows)
    new_df.to_csv(os.path.join(folder_path,'prompts.csv'),index=False)

# generate a single image and return tensor
def generate_single_image(model, df, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=1):
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

    repeated_rows = []
    row = df.iloc[0]
    i = 0
    prompt = [str(row.prompt)]*num_samples
    seed = row.evaluation_seed if hasattr(row,'evaluation_seed') else row.sd_seed
    case_number = row.case_number if hasattr(row,'case_number') else i
    repeated_rows.extend([row]*num_samples)
    
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

    # return latents
    latents = 1 / 0.18215 * latents
    # resize to 512x512
    latents = torch.nn.functional.interpolate(latents, size=(512,512), mode='bilinear', align_corners=False)
    return latents

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, default='/vhome/gongchao/UCE_acceleration/adv-train/dataset/coco_30k.csv')
    parser.add_argument('--concept', help='unsafe or art, object concept', type=str, default='nudity')
    parser.add_argument('--save_path', help='folder where to save images', type=str, default='/share/ckpt/gongchao/ESD/dataset')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--cache_dir', help='cache directory', type=str, required=False, default='./.cache')
    parser.add_argument('--ckpt', help='ckpt path', type=str, required=False, default=None)
    parser.add_argument('--base', help='version of stable diffusion to use', type=str, required=False, default='1.4')
    parser.add_argument('--df_length', help='number of prompts to use', type=int, required=False, default=None)
    parser.add_argument('--df_start', help='start index of prompts to use', type=int, required=False, default=0)
    args = parser.parse_args()
    
    prompts_path = args.prompts_path
    concept = args.concept
    save_path = f'{args.save_path}/{concept}'
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    cache_dir  = args.cache_dir
    base = args.base
    ckpt = args.ckpt
    
    if base=='1.4':
        dir_= '/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
    elif base=='2.1':
        dir_ = '/share/ckpt/gongchao/model_zoo/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
    else:
        raise ValueError('base should be 1.4 or 2.1')
    
    model = StableDiffusionPipeline.from_pretrained(
        dir_,)
    if ckpt is not None:
        if ckpt.endswith('pt'):
            try:
                model.unet.load_state_dict(torch.load(ckpt))
            except:
                ckpt_load = torch.load(ckpt)['state_dict']
                print(ckpt_load.keys())
                model.unet.load_state_dict(ckpt_load, strict=False)
        else:
            try:
                model.unet = UNet2DConditionModel.from_pretrained(ckpt)
            except:
                raise ValueError('edited model not found')
    model = model.to(device)
    
    df = pd.read_csv(prompts_path)
    df = df[args.df_start:]
    if args.df_length is not None:
        df = df[:args.df_length]
    
    generate_images(model, df, save_path, device=device, ddim_steps=ddim_steps)
    if concept == 'nudity':
        nudity_rate = compute_nudity_rate(save_path)
        print(f'Nudity Rate: {nudity_rate}')