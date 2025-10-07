import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import numpy as np
import json
from datasets import load_dataset, load_from_disk, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import gc
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import warnings

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress some warnings that may appear
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")

# Main training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Available GPUs: {torch.cuda.device_count()}")

base_model = "Qwen/Qwen2.5-7B-Instruct"
finetune_model = "./models/Qwen-7B-SFT-0814"

def load_long_short():
    dataset_path = "database/prompts/ALDE_long_conversational_96.json"
    dataset = json.load(open(dataset_path))
    return [{"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]} 
            for prompt, completion in zip(dataset["prompt"], dataset["completion"])]

conv_dataset = load_long_short()
print("Sample conversation:", conv_dataset[0])

dataset = Dataset.from_list(conv_dataset)
split = int(len(dataset) * 6/7)
train_dataset = dataset.select(range(split))
val_dataset = dataset.select(range(split, len(dataset)))
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Set pad token")

# Pre-tokenize dataset for group_by_length compatibility
def pretokenize_dataset(examples):
    """Pre-tokenize the dataset to add input_ids for group_by_length."""
    tokenized_examples = {"input_ids": [], "attention_mask": [], "messages": []}
    
    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )
        
        tokenized_examples["input_ids"].append(encoding["input_ids"])
        tokenized_examples["attention_mask"].append(encoding["attention_mask"])
        tokenized_examples["messages"].append(messages)
    
    return tokenized_examples

# Pre-tokenize the datasets if using group_by_length
use_group_by_length = True  # Set to True if you want to use group_by_length
if use_group_by_length:
    print("Pre-tokenizing dataset for group_by_length...")
    train_dataset = train_dataset.map(
        pretokenize_dataset,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    val_dataset = val_dataset.map(
        pretokenize_dataset,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )
    print(f"Train dataset columns: {train_dataset.column_names}")
    print(f"Sample tokenized example length: {len(train_dataset[0]['input_ids'])} tokens")

collator = DataCollatorForCompletionOnlyLM(
    response_template="<|im_start|>assistant\n",
    instruction_template="<|im_start|>user\n",
    tokenizer=tokenizer,
)
print("Using DataCollatorForCompletionOnlyLM")

# Training configuration with important settings for custom collator
training_args = SFTConfig(
    max_length=4096,
    output_dir=finetune_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
    weight_decay=0.1,
    
    # CRITICAL: These settings are needed for custom collator compatibility
    remove_unused_columns=False,  # Don't remove messages column
    ddp_find_unused_parameters=False,
    
    # Only use group_by_length if dataset is pre-tokenized
    group_by_length=use_group_by_length,
    
    # Training parameters
    num_train_epochs=2,
    eval_steps=25,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    dataloader_num_workers=0,
    disable_tqdm=False,
    report_to=None,
    
    # IMPORTANT: Skip SFTTrainer's default preprocessing to use our custom collator
    dataset_kwargs={"skip_prepare_dataset": True},
    
    # Disable eval packing which can cause issues with custom collators
    eval_packing=False,
    
    # Disable the completion-only metrics that cause shape mismatches
    assistant_only_loss=False,  # We handle this in our collator
)

# Initialize trainer
# Note: We'll override the trainer if we need custom behavior

# Use standard SFTTrainer for DataCollatorForCompletionOnlyLM
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model(finetune_model)
tokenizer.save_pretrained(finetune_model)
print(f"Model saved to {finetune_model}")