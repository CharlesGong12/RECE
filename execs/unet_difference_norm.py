from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--edited', type=str, default=None, help='path to the edited checkpoint')

args = parser.parse_args()
pt1 = args.edited
sd14="/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

sd = StableDiffusionPipeline.from_pretrained(sd14)
sd_edited = StableDiffusionPipeline.from_pretrained(sd14)
if pt1.endswith('.pt'):
    sd_edited.unet.load_state_dict(torch.load(pt1))
else:
    try:
        sd_edited.unet = UNet2DConditionModel.from_pretrained(pt1)
        # torch.save(sd_edited.unet.state_dict(), pt1+'.pt')
    except:
        raise ValueError('edited model not found')

sd.to('cuda:0')
sd_edited.to('cuda:0')

difference = []

for p1, p2 in zip(sd.unet.named_parameters(), sd_edited.unet.named_parameters()):
    difference.append((p1[1]-p2[1]).norm().item())

print(f'difference: {sum(difference):.4f}')