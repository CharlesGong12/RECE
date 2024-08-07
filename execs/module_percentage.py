import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

# Initialize counters for cross-attention and non-cross-attention parameters
cross_attention_kv_params = 0
non_cross_attention_params = 0
cross_attention_params = 0

model_version = '/share/ckpt/gongchao/model_zoo/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
ldm_stable = StableDiffusionPipeline.from_pretrained(
        model_version,
        )
ldm_stable = ldm_stable.to('cuda')

# Iterate through the U-Net's named children
for name, module in ldm_stable.unet.named_children():
    # Check if the module is an up or down block
    if 'up' in name or 'down' in name:
        # Iterate through the blocks in the module
        for block in module:
            # Check if the block is a cross-attention block
            if 'Cross' in block.__class__.__name__:
                # Iterate through the attentions in the cross-attention block
                for attn in block.attentions:
                    # Iterate through the transformer blocks in the attention
                    for transformer in attn.transformer_blocks:
                        # Accumulate parameters from key and value modules in cross-attention
                        cross_attention_kv_params += sum(p.numel() for p in [transformer.attn2.to_k.weight, transformer.attn2.to_v.weight])
                        cross_attention_params += sum(p.numel() for p in transformer.attn2.parameters())
    elif 'mid' in name:
        # Iterate through the attentions in the mid block
        for attn in module.attentions:
            # Iterate through the transformer blocks in the attention
            for transformer in attn.transformer_blocks:
                # Accumulate parameters from key and value modules
                cross_attention_kv_params += sum(p.numel() for p in [transformer.attn2.to_k.weight, transformer.attn2.to_v.weight])
                cross_attention_params += sum(p.numel() for p in transformer.attn2.parameters())

# Calculate the total parameters in the U-Net
total_params = sum(p.numel() for p in ldm_stable.unet.parameters())

# Calculate the percentage of cross-attention and non-cross-attention parameters
percentage_cross_attention_kv = (cross_attention_kv_params / total_params) * 100
percentage_cross_attention = (cross_attention_params / total_params) * 100

for name, param in ldm_stable.unet.named_parameters():
    if name.startswith("conv_out.") or ("time_embed" in name) or ('attn2' in name):
        continue
    non_cross_attention_params += param.numel()
    
percentage_non_cross_attention = (non_cross_attention_params / total_params) * 100

print(f"Percentage of Cross-Attention KV Parameters: {percentage_cross_attention_kv:.2f}%")
print(f"Percentage of Cross-Attention Parameters: {percentage_cross_attention:.2f}%")
print(f"Percentage of Non-Cross-Attention Parameters: {percentage_non_cross_attention:.2f}%")