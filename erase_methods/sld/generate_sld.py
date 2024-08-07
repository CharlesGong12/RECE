import argparse
import os
import torch
import pandas as pd
from diffusers import LMSDiscreteScheduler, StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig


config_cases = {
    "weak": SafetyConfig.WEAK,
    "medium": SafetyConfig.MEDIUM,
    "strong": SafetyConfig.STRONG,
    "max": SafetyConfig.MAX,
}


def generate_SLD(sld_strength, prompts_path, save_path, device="cuda:0", guidance_scale=7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, till_case=1000000, base="1.4", safety_concept=None):
    # Generates Images with SLD pipeline
    if base == "1.4":
        model_version = "/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
    elif base == "2.1":
        model_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        model_version = "CompVis/stable-diffusion-v1-4"
    scheduler = LMSDiscreteScheduler.from_pretrained(model_version, subfolder="scheduler")
    pipeline = StableDiffusionPipelineSafe.from_pretrained(model_version, scheduler=scheduler, safety_checker=None).to(device)
    if safety_concept is not None:
        pipeline.safety_concept = safety_concept

    folder_path = f"{save_path}/sld_{sld_strength}"
    os.makedirs(folder_path, exist_ok=True)

    df = pd.read_csv(prompts_path)
    for i, row in df.iterrows():
        prompt = [str(row.prompt)] * num_samples if not hasattr(row, "adv_prompt") else [str(row.adv_prompt)] * num_samples
        try:
            seed = row.evaluation_seed if hasattr(row,'evaluation_seed') else row.sd_seed
        except:
            seed = 42
        case_number = row.case_number if hasattr(row,'case_number') else i
        if "unsafe" in prompts_path:
            guidance_scale = row.evaluation_guidance
        elif "object" in prompts_path:
            class_name = row.class_name
            if not os.path.exists(os.path.join(folder_path, class_name)):
                os.makedirs(os.path.join(folder_path, class_name), exist_ok=True)

        if not (case_number >= from_case and case_number <= till_case):
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.cuda.manual_seed(seed)    # Seed generator to create the inital latent noise

        pil_images = pipeline(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, **config_cases[sld_strength])[0]

        for num, im in enumerate(pil_images):
            if "unsafe" in prompts_path:
                im.save(f"{folder_path}/unsafe/{case_number}_{num}.png")
            elif "object" in prompts_path:
                im.save(f"{folder_path}/{class_name}/{case_number}_{num}.png")
            else:
                im.save(f"{folder_path}/{case_number}_{num}.png")


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="generateImages", description="Generate Images using Diffusers Code")
    parser.add_argument("--prompts_path", help="path to csv file with prompts", type=str, required=True)
    parser.add_argument("--save_path", help="folder where to save images", type=str, required=True)
    parser.add_argument("--device", help="cuda device to run on", type=str, required=False, default="cuda:0")
    parser.add_argument("--base", help="version of stable diffusion to use", type=str, required=False, default="1.4")
    parser.add_argument("--guidance_scale", help="guidance to run eval", type=float, required=False, default=7.5)
    parser.add_argument("--image_size", help="image size used to train", type=int, required=False, default=512)
    parser.add_argument("--till_case", help="continue generating till case_number", type=int, required=False, default=1000000)
    parser.add_argument("--from_case", help="continue generating from case_number", type=int, required=False, default=0)
    parser.add_argument("--num_samples", help="number of samples per prompt", type=int, required=False, default=1)
    parser.add_argument("--ddim_steps", help="ddim steps of inference used to train", type=int, required=False, default=50)
    parser.add_argument("--sld_strength", help="Type of SLD", type=str, required=False, choices=["weak", "medium", "strong", "max"], default="medium")
    parser.add_argument('--safety_concept', help='New safety concept to use', type=str, required=False, default=None)
    args = parser.parse_args()
    
    sld_strength = args.sld_strength
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    base = args.base
    safety_concept = args.safety_concept
    generate_SLD(sld_strength=sld_strength, prompts_path=prompts_path, save_path=save_path, device=device, guidance_scale=guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, till_case=till_case, base=base, safety_concept=safety_concept)