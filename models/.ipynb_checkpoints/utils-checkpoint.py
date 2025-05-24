from transformers import AutoTokenizer
import math
from typing import List, Optional, Tuple, Union
import torch
import logging
import numpy as np
import json
from transformers import GenerationConfig, TextStreamer

def _prepare_4d_abc_mask(
	attention_mask: torch.Tensor,
	sequence_length: int,
	target_length: int,
	dtype: torch.dtype,
	device: torch.device,
	cache_position: torch.Tensor,
	batch_size: int,
    checkpoint_mask: torch.Tensor,
	**kwargs,
):
    assert checkpoint_mask.dim() == 2 # shape: (B, S)
    assert attention_mask is not None

	if attention_mask is not None and attention_mask.dim() == 4:
		# In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
		causal_mask = attention_mask
	else:
		min_dtype = torch.finfo(dtype).min
		causal_mask = torch.full(
			(sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
		)
		if sequence_length != 1:
			causal_mask = torch.triu(causal_mask, diagonal=1)
		causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
		causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
		if attention_mask is not None:
			causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
			mask_length = attention_mask.shape[-1]
			padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
				causal_mask.device
			)
			padding_mask = padding_mask == 0
			causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
				padding_mask, min_dtype
			)
        
        # Checkpoint mask handling (basically prefix bi-directional attention)
        ZERO = torch.zeros(1).item()
        mask_length = attention_mask.shape[-1]
        checkpoint_mask = checkpoint_mask.to(causal_mask.device)
        checkpoint_mask = (checkpoint_mask[:, :, None] * checkpoint_mask[:, None, :]) == 1
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            checkpoint_mask[:, None, :, :], ZERO # shape: (B, S, S) -> (B, 1, S, S)
        )

	return causal_mask


def generate(
    prompt, model, tokenizer,
    repetition_penalty=1.13,
    top_p=0.95,
    top_k=50,
    max_new_tokens=1024,
    temperature=0.4,
    eos_token_id=None,
    do_sample=False,
    use_cache=True,
    return_dict_in_generate=True,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    streaming=True
):
    input_ids = tokenizer(
        prompt, return_tensors="pt"
    )["input_ids"].to(model.device)
    eos_token_id = (eos_token_id if eos_token_id is not None 
                    else tokenizer.eos_token_id)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
            use_cache=use_cache,
            return_dict_in_generate=return_dict_in_generate,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer if streaming else None,
        )
        
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()
