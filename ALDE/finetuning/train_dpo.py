import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import numpy as np
import pandas as pd
import json
from datasets import load_dataset, load_from_disk, Dataset
from trl import DPOConfig, DPOTrainer, apply_chat_template
import os
import gc
import sys
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Load dataset
dataset_path = "database/prompts/ALDE_long_prompts.json"
model_name = "./models/Qwen2-7B-SFT-Long-0731" #"Qwen/Qwen2.5-0.5B-Instruct"  # Start here for testing
finetune_model = "Qwen2-7B-DPO-Long-0731"

dataset = json.load(open(dataset_path))
prompt = [i["prompt"] for i in dataset]
chosen = [i["chosen"] for i in dataset]
rejected = [i["rejected"] for i in dataset]
dataset ={
    "prompt": prompt, "chosen": chosen, "rejected": rejected
}
dataset = Dataset.from_dict(dataset)


# For larger models, consider these options:
# model_name = "microsoft/DialoGPT-large"  # 774M parameters
# model_name = "EleutherAI/gpt-neo-1.3B"  # 1.3B parameters  
# model_name = "EleutherAI/gpt-neo-2.7B"  # 2.7B parameters
# model_name = "EleutherAI/gpt-j-6b"      # 6B parameters
# model_name = "meta-llama/Llama-2-7b-hf" # 7B parameters
# model_name = "meta-llama/Llama-2-13b-hf" # 13B parameters 


print(f"Loading {model_name}...")

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    padding_side="left", 
    local_files_only=True,
    trust_remote_code=True
)

# Load model optimized for large model training
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
    local_files_only=True,
    attn_implementation="flash_attention_2",
    #For very large models, consider:
)
model_ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
    local_files_only=True,
    # For very large models, consider:
    # low_cpu_mem_usage=True,
    # torch_dtype=torch.float16,  # Alternative to bfloat16
)
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

def create_task_based_split(dataset, test_fraction=1/7):
    """Split dataset for proper train/eval separation and filter prompts > 3072 tokens"""
    
    # Filter out prompts longer than 3072 tokens
    def filter_long_prompts(example):
        prompt_tokens = tokenizer(example['prompt'], add_special_tokens=False)['input_ids']
        return len(prompt_tokens) <= 1024
    
    original_size = len(dataset)
    filtered_dataset = dataset.filter(filter_long_prompts)
    filtered_size = len(filtered_dataset)
    
    removed_count = original_size - filtered_size
    removed_percentage = (removed_count / original_size) * 100
    print(f"Removed {removed_count} samples ({removed_percentage:.2f}%) with prompts > 3072 tokens")
    
    total_size = filtered_size
    split_point = int(total_size * (1 - test_fraction))
    
    train_dataset = filtered_dataset.select(range(split_point))
    eval_dataset = filtered_dataset.select(range(split_point, total_size))
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

train_dataset, eval_dataset = create_task_based_split(dataset)

# After creating your DPO dataset, inspect a few examples
for i in range(3):
    example = train_dataset[i]
    
    print(f"\nExample {i}:")
    print(example['prompt'], len(tokenizer.encode(example['prompt'])))
    print(f"Chosen {example['chosen']} length: {len(tokenizer.encode(example['chosen']))}")
    print(f"Rejected {example['rejected']} length: {len(tokenizer.encode(example['rejected']))}")

print(f"Max prompt length: {max([len(tokenizer.encode(train_dataset[i]['prompt'])) for i in range(len(train_dataset))])}")# Configure model for training
model.config.use_cache = False
model.enable_input_require_grads()

print(f"Model loaded successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")

# Training configuration optimized for large models
training_args = DPOConfig(
    output_dir=f"./models/{finetune_model}",
    hub_model_id=finetune_model,
    learning_rate=5e-7,  # Lower learning rate for full fine-tuning
    beta=0.9,
    num_train_epochs=2,
    max_grad_norm=1.0,
    # Memory optimization settings
    per_device_train_batch_size=1,      # Start with 1, increase if memory allows
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,     # High accumulation for effective batch size
    gradient_checkpointing=True,        # Essential for large models
    
    # Precision settings
    bf16=True,                          # Better than fp16 for stability
    dataloader_pin_memory=False,        # Reduce CPU memory usage
    
    # Length management parameterss
    max_prompt_length=1024,
    max_completion_length=48,
    truncation_mode="keep_end",
    
    # Training efficiency
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
    save_steps=50,
    report_to=None,
    
    # Advanced memory optimizations
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,   # Important for DDP efficiency
    
    # For very large models
    #max_grad_norm=1.0,                # Gradient clipping
    weight_decay=0.1,                # Regularization
)

# Initialize trainer
trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    ref_model=model_ref,
    eval_dataset=eval_dataset,
)

# Memory monitoring function
def print_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Max memory allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")

print("Memory stats before training:")
print_memory_stats()

# Train the model
print("\nStarting training...")
trainer.train()

print("\nMemory stats after training:")
print_memory_stats()

# Save the final model
output_dir = f"./models/{finetune_model}"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Print final statistics
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Original model: {model_name}")
print(f"Model saved to: {output_dir}")
print(f"Training approach: Full fine-tuning (no LoRA)")
print(f"Memory optimization: DeepSpeed ZeRO Stage 3 + CPU offloading")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")


print("\nTraining completed successfully!")
print("For larger models (7B+), consider:")
print("1. Increasing gradient_accumulation_steps")
print("2. Using DeepSpeed ZeRO Stage 3 with CPU offloading")
print("3. Reducing sequence length if needed")
print("4. Using fp16 instead of bf16 if experiencing instability")
