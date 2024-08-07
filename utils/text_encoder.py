import torch
from typing import Any, Optional, Tuple, Union
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

class CustomTextEncoder(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.embedding = text_encoder.text_model.embeddings
        self.encoder = text_encoder.text_model.encoder
        self.final_layer_norm = text_encoder.text_model.final_layer_norm
        self.config = text_encoder.text_model.config
        self.eos_token_id = self.config.eos_token_id
    def get_all_embedding(self):
        return self.embedding.token_embedding.weight
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds : Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds  is None:
            raise ValueError("You have to specify input_embds")
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        hidden_states = self.embedding(inputs_embeds=inputs_embeds , position_ids=position_ids)
        input_shape = input_ids.size()
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


if __name__ == "__main__":
    dir_ = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer", cache_dir="../ldm_pretrained/")
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder", cache_dir="../ldm_pretrained/")
    print(text_encoder.text_model.config)
    model = CustomTextEncoder(text_encoder)

    model.eval()
    text_encoder.eval()
    all_embeddings = model.get_all_embedding()
    print(all_embeddings.shape)
    text = ["a photo of a cat"]
    
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    orig_prompt_len = input_ids.shape[1]
    print(input_ids)
    out = torch.nn.functional.one_hot(input_ids.view(-1), num_classes=all_embeddings.shape[0])
    input_embeds = out.float() @ all_embeddings
    prompt_num = 5
    #### test customer encoder output whether is same as text_encoder
    outputs = model(input_ids=input_ids, inputs_embeds=input_embeds)
    outputs2 = text_encoder(input_ids)
    assert torch.count_nonzero(outputs[0] - outputs2[0]) == 0 
    print(input_embeds.shape) 
    #### test customer encoder output whether change when input_ids change
    adv_embeddings = torch.zeros(prompt_num,768)
    
    
    new_embeddings = torch.cat([input_embeds[0:-1,:],adv_embeddings,torch.unsqueeze(input_embeds[-1,:],0)], dim=0)
    print(input_ids)
    new_input_ids = torch.cat([input_ids[:,0:-1], torch.zeros(1,prompt_num).long(),torch.unsqueeze(input_ids[:,-1],0)], dim=1)
    new_input_ids2 = torch.cat([input_ids[:,0:-1], torch.ones(1,prompt_num,).long(),torch.unsqueeze(input_ids[:,-1],0)], dim=1)
    print(new_input_ids)
    print(new_embeddings.shape)
    output3 = model(input_ids=new_input_ids, inputs_embeds=new_embeddings)
    output4 = model(input_ids=new_input_ids2, inputs_embeds=new_embeddings)
    assert torch.count_nonzero(output3[0] - output4[0]) == 0
    output5 = text_encoder(new_input_ids)
    assert torch.count_nonzero(output3[0] - output5[0]) != 0