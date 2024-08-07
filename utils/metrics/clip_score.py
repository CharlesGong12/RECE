import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")


def calculate_clip_score(images, prompts,device):
    clip_score = clip_score_fn(torch.from_numpy(images).to(device), prompts).detach()
    return round(float(clip_score), 4)
