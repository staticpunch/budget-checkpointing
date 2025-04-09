from transformers import AutoTokenizer
import math
from typing import List, Optional, Tuple, Union
import torch
import logging
import numpy as np
import json
from transformers import GenerationConfig, TextStreamer

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
