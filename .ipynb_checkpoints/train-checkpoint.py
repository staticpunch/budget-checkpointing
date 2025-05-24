#!/usr/bin/env python3
"""
Training script for reasoning models with thought tokens.
"""

import argparse
import json
import yaml
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AddedToken,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models.llama import LlamaForCausalLM

# Setup logging
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("train")


@dataclass
class TrainingConfig:
    """Configuration for training loaded from YAML."""
    # Model configuration
    model_name_or_path: str
    
    # Dataset configuration
    dataset_configs: Dict[str, int]  # Path to dataset -> number of samples
    train_split: str
    max_length: int
    
    # Training parameters
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    eval_strategy: str
    report_to: str
    remove_unused_columns: bool = False
    logging_first_step: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = False
    validation_split: Optional[str] = None
    train_on_inputs: bool = False
    lr_scheduler: str = "constant"
    preprocessing_num_workers: int = 16

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Convert learning rate to float if it's a string
            if isinstance(config_dict['learning_rate'], str):
                config_dict['learning_rate'] = float(config_dict['learning_rate'])
        return cls(**config_dict)

class DataProcessor:
    """Handles dataset loading and preprocessing."""
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        dataset_configs: Dict[str, Optional[int]], 
        split: str, 
        max_length: int, 
        train_on_inputs: bool = False
    ):
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.split = split
        self.max_length = max_length
        self.train_on_inputs = train_on_inputs

    def load_dataset(self):
        """Load and prepare the training dataset."""
        datasets_list = []
        
        for data_config, num_samples in self.dataset_configs.items():
            logger.info(f"Loading {num_samples if num_samples else 'all'} samples from {data_config}")
            new_dataset = load_dataset(data_config, split=self.split)
            
            # Randomly sample the specified number of examples
            if num_samples and num_samples < len(new_dataset):
                new_dataset = new_dataset.shuffle(seed=42).select(range(num_samples))

            datasets_list.append(new_dataset)
            
        train_dataset = concatenate_datasets(datasets_list)
        logger.info(f"Training on {len(train_dataset)} samples total.")
        return train_dataset.shuffle(seed=101)

    def tokenize(self, element):
        """Tokenize a single element and mark tokens for loss computation based on train_on_inputs."""
        effective_spans = []
        current_position = 0
        
        # Track positions of assistant messages
        for message in element["messages"]:
            message_tokens = self.tokenizer.apply_chat_template(
                [message],
                tokenize=True,
                add_generation_prompt=False
            )
            
            if message["role"] == "assistant" or (self.train_on_inputs and message["role"] == "user"):
                effective_spans.append((
                    current_position,
                    current_position + len(message_tokens)
                ))
            current_position += len(message_tokens)

        # Tokenize full conversation
        tokenized = self.tokenizer(
            self.tokenizer.apply_chat_template(
                element["messages"],
                tokenize=False,
                add_generation_prompt=False
            ),
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        # Create labels with -100 for tokens we don't want to compute loss on
        labels = [-100] * len(tokenized["input_ids"])
        for start, end in effective_spans:
            for i in range(start, min(end, len(labels))):
                labels[i] = tokenized["input_ids"][i]
                
        tokenized["labels"] = labels
        return tokenized


@dataclass 
class SFTDataCollator:
    """Data collator for supervised fine-tuning with proper padding and loss masking."""
    
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 4
    return_tensors: str = "pt"
    
    def _format_batch_log(self, batch):
        """Format the batch sample log showing colored text chunks."""
        input_ids = batch["input_ids"][0].tolist()  
        labels = batch["labels"][0].tolist()
        
        # Build chunks of tokens with same label type (-100 or non -100)
        chunks = []
        current_chunk = {"tokens": [], "is_ignored": labels[0] == -100}
        
        for token_id, label in zip(input_ids, labels):
            is_ignored = label == -100
            # If label type changes, start new chunk
            if is_ignored != current_chunk["is_ignored"]:
                chunks.append(current_chunk)
                current_chunk = {"tokens": [], "is_ignored": is_ignored}
            current_chunk["tokens"].append(token_id)
        
        # Add final chunk
        chunks.append(current_chunk)
        
        # Format output
        log_messages = []
        log_messages.append("=== Sample text chunks ===")
        # Decode and display each chunk with appropriate color
        for i, chunk in enumerate(chunks):
            text = self.tokenizer.decode(chunk["tokens"])
            color = ("\033[90m" if (i == len(chunks) - 1) and chunk["is_ignored"] # Gray for padded.
                     else "\033[91m" if chunk["is_ignored"] # Red for ignored.
                     else "\033[92m") # Green for trained.
            log_messages.append(f"{color}{text}\033[0m")
            
        log_messages.append("==========================")
        return "\n".join(log_messages)

    def __post_init__(self):
        self.first_batch = True

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch with proper padding."""
        if not isinstance(examples[0], Mapping):
            raise ValueError("Data collator only processes list of dictionaries.")
        
        # Extract input_ids and labels
        input_ids_list = []
        labels_list = []
        other_features = {}
        
        for example in examples:
            # Pop attention_mask if present (not needed for padding)
            example.pop("attention_mask", None)
            
            # Extract input_ids and labels
            input_ids_list.append({"input_ids": example.pop("input_ids")})
            labels_list.append({"input_ids": example.pop("labels")})
            
            # Collect other features
            for key, value in example.items():
                if key not in other_features:
                    other_features[key] = []
                other_features[key].append(value)
        
        # Pad input_ids
        batch = self.tokenizer.pad(
            input_ids_list,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side="right",
        )
        
        # Pad labels
        labels_batch = self.tokenizer.pad(
            labels_list,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding_side="right",
        )
        
        # Set padded positions in labels to -100 (ignore in loss)
        labels = labels_batch["input_ids"]
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        # Add other features to batch
        for key, values in other_features.items():
            if key in batch:
                raise ValueError(
                    f"`{key}` feature is already collated. "
                    "Overriding it with initial values is prohibited."
                )
            
            # Convert to tensor if all values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                batch[key] = torch.tensor(values, dtype=torch.long)
            else:
                batch[key] = values

        # All logging in a single info_once call
        if self.first_batch:
            logger.info(f"Logging first batch sample:\n{self._format_batch_log(batch)}")
            self.first_batch = False
        
        return batch

## Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
##

class SFTTrainer(Trainer):
    """Custom trainer for supervised fine-tuning with enhanced debugging."""
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def compute_entropy_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get inputs
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift logits and labels for causal LM
        # logits: [batch_size, seq_len, vocab_size]
        # We predict token i+1 from tokens 0:i
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()  # Remove last token
        shift_labels = labels[..., 1:].contiguous()      # Remove first token
        
        # Flatten for cross entropy computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))    # [batch_size * (seq_len-1), vocab_size]
        shift_labels = shift_labels.view(-1).to(shift_logits.device)  # [batch_size * (seq_len-1)]
        
        # Compute cross entropy loss (ignore_index=-100 by default)

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, 
            ignore_index=-100, reduction=reduction
        )
        if reduction == "sum":
            loss = loss / num_items_in_batch
            
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # loss = loss_fct(shift_logits, shift_labels)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Debug: Before computing loss
        for name, param in model.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                logger.warning(f"NaN detected in parameter {name}")
                if self.args.debug:
                    import pdb; pdb.set_trace()

        # with torch.no_grad():
        #     # loss_func = self.compute_entropy_loss
        #     # loss1 = loss_func(model, inputs, return_outputs, num_items_in_batch)
        #     outputs = model(**inputs)
        #     loss1 = ForCausalLMLoss(
        #         logits=outputs.logits, 
        #         labels=inputs["labels"],
        #         vocab_size=outputs.logits.size(-1),
        #         num_items_in_batch=num_items_in_batch,
        #         ignore_index=-100
        #     )
        #     loss2 = model(**inputs).loss
        #     torch.testing.assert_allclose(loss1, loss2)

        # loss = model(**inputs).loss
        # outputs = model(**inputs)
        # loss = ForCausalLMLoss(
        #     logits=outputs.logits, 
        #     labels=inputs["labels"],
        #     vocab_size=outputs.logits.size(-1),
        #     num_items_in_batch=num_items_in_batch,
        #     ignore_index=-100
        # )
        loss = self.compute_entropy_loss(model, inputs, return_outputs, num_items_in_batch)
        # Debug: After computing loss
        if torch.isnan(loss):
            logger.warning(f"Loss became NaN")
            if self.args.debug:
                import pdb; pdb.set_trace()

        return (loss, outputs) if return_outputs else loss


def setup_tokenizer(args) -> PreTrainedTokenizerBase:
    """Setup tokenizer with special tokens for reasoning."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    
    # Set pad token if not present
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add latent tokens
    latent_tokens = [
        AddedToken(f"<|latent_{i:03}|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)
        for i in range(256)
    ]

    # Add reasoning markers
    markers = ["<|begin_of_thought|>", "<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"]
    marker_tokens = [
        AddedToken(marker, rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)
        for marker in markers
    ]

    new_tokens = latent_tokens + marker_tokens
    tokenizer.add_tokens(new_tokens)
    
    logger.info(f"Added {len(new_tokens)} special tokens to tokenizer")
    return tokenizer


def setup_model(args, tokenizer: PreTrainedTokenizerBase):
    """Setup model with proper dtype and resize embeddings."""
    # Parse torch dtype
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    
    # Resize token embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to YAML config file')
    config_file = parser.parse_args().config_file
    args = TrainingConfig.from_yaml(config_file)

    # Setup tokenizer and model
    tokenizer = setup_tokenizer(args)
    model = setup_model(args, tokenizer)

    # Setup data processor
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        dataset_configs=args.dataset_configs,
        split=args.train_split,
        max_length=args.max_length,
        train_on_inputs=args.train_on_inputs
    )

    # Load and tokenize dataset
    logger.info("Loading and tokenizing dataset...")
    train_dataset = data_processor.load_dataset()
    tokenized_dataset = train_dataset.map(
        data_processor.tokenize,
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing dataset"
    ).select_columns(["input_ids", "attention_mask", "labels"])

    logger.info(f"Tokenized dataset: {tokenized_dataset}")

    # Setup data collator
    data_collator = SFTDataCollator(tokenizer=tokenizer)


    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() 
        if buffer.dtype == torch.bool
    ]
    
    # Setup training arguments and data collator
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if args.validation_split else "no",
        eval_steps=args.eval_steps if args.validation_split else None,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=args.report_to,  # Enable TensorBoard logging
        remove_unused_columns=args.remove_unused_columns,
        logging_first_step=args.logging_first_step,
        gradient_checkpointing=args.gradient_checkpointing,
        # bf16=args.bf16,
        # fp16=not args.bf16,
        ddp_find_unused_parameters=False
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    # Training
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()