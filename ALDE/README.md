# ALDE: Active Learning with LLMs for Protein Optimization

This repository contains the implementation for benchmarking Large Language Models (LLMs) in protein active learning, comparing traditional Bayesian Optimization methods with LLM-based approaches for optimizing 4-amino acid protein motifs.

Framework and data based on:  
Yang, J., Lal, R. G., Bowden, J. C., Astudillo, R., Hameedi, M. A., Kaur, S., Hill, M., Yue, Y., & Arnold, F. H. (2025). *Active learning-assisted directed evolution*. Nature Communications, 16, Article 714. [https://www.nature.com/articles/s41467-025-55987-8](https://www.nature.com/articles/s41467-025-55987-8)
Workflow comes directly from https://github.com/jsunn-y/ALDE/tree/master

## Overview

The project evaluates different approaches to protein sequence optimization:
- **Statistical Methods**: Traditional Bayesian Optimization (BO) with various acquisition functions
- **LLM Agents**: Multi-agent systems using reasoning models for strategic sequence selection  
- **Standalone LLM Models**: Direct optimization using GPT-5, Qwen3, DeepSeek-R1 via API calls
- **Fine-tuned Models**: Custom-trained LLMs for direct sequence generation and conversational optimization


## Repository Structure

```
ALDE/
├── src/                               # Core implementation (much from original paper)
│   ├── acquisition.py                 # Acquisition functions and helpers
│   ├── agent.py                       # Multi-agent LLM workflow system
│   ├── AL_LLM.py                      # LLM integration for AL workflows
│   ├── encoding_utils.py              # Encodings (onehot, ESM2, etc.)
│   ├── implementer.py                 # Execution helpers
│   ├── models.py                      # Surrogate models and wrappers
│   ├── networks.py                    # Neural network architectures
│   ├── objectives.py                  # Protein fitness objectives (GB1, TrpB)
│   ├── optimize.py                    # Bayesian optimization implementation
│   ├── prompts.py                     # LLM prompting system
│   ├── prompts_LLMChainSelector.yaml  # Prompt templates
│   ├── sft_run.py                     # Fine-tuned model execution agent
│   ├── standalone_run.py              # Reasoning LLM agent (GPT-5, Qwen3, DeepSeek-R1)
│   ├── tools.py                       # Tooling for agents
│   └── utils.py                       # Utility functions
├── data/                              # Protein fitness datasets + synthetic
│   ├── GB1/                           # GB1 4-site epistatic region data
│   ├── TrpB/                          # TrpB enzyme activity data
│   └── artificial_ESM2_*/             # Synthetic datasets generated from ESM2
├── database/
│   └── prompts/                       # Generated SFT/DPO training prompts
├── finetuning/                        # Fine-tuning pipeline (SFT/DPO)
│   ├── gen_data.py
│   ├── gen_prompts.py
│   ├── train_dpo.py
│   ├── train_sft.py
│   └── training_utils.py
├── analysis/                          # Analysis and visualization
│   ├── tabulate_results.py            # Results processing script
│   └── visualization.ipynb            # Results analysis notebook
├── models/                            # Fine-tuned model checkpoints
├── results/                           # Output folders from experiments
├── execute_simulation.py              # Run BO baselines
├── execute_standalone.py              # Batch execution for standalone/sft/game
├── directed_evolution.py              # Directed evolution baseline
├── requirements-alde.txt              # Pip requirements (main)
├── requirements-trl.txt               # Pip requirements (training)
├── alde.yaml                          # Conda env (main)
└── trl.yaml                           # Conda env (training)
```

## Installation

### PerlMutter Prerequisites

```bash
module load conda/Miniforge3-24.7.1-0
module load cuda/12.9
```

### Environment Setup

**Option 1: Create from YAML files (recommended)**
```bash
# Main ALDE environment
conda env create -n alde_env -f ALDE/alde.yaml
# TRL training environment (for fine-tuning)
conda env create -n trl_env -f ALDE/trl.yaml
```

**Option 2: Create from pip requirements**
```bash
# Main environment
conda create -n alde_env python=3.11
conda activate alde_env
pip install -r ALDE/requirements-alde.txt
# Training environment
conda create -n trl_env python=3.10
conda activate trl_env  
pip install -r ALDE/requirements-trl.txt
```

### API Keys Setup

For LLM-based methods, you'll need API keys:
```bash
export OPENAI_API_KEY="your_openai_key"        # For GPT-5
export ANTHROPIC_API_KEY="your_anthropic_key"  # For Claude
export LAMBDA_API_KEY="your_lambda_key"        # For Qwen3 via Lambda
export LANGSMITH_API_KEY="your_langsmith_key"  # For experiment tracking
```

## Usage

### 1. Traditional Bayesian Optimization

Run baseline BO methods with different acquisition functions. Total budget = budget +  n_pseudorand_init

```bash
cd ALDE
conda activate alde_env
# Small campaigns (Good model + encoding)
python execute_simulation.py \
    --names GB1 TrpB \
    --encodings onehot \
    --model_type DNN_ENSEMBLE \
    --acq_fn UCB TS GREEDY \
    --batch_size 10 \
    --budget 50 \
    --n_pseudorand_init 10 \
    --runs 50
# Large campaign (from paper)
python execute_simulation.py \
    --names GB1 \
    --encodings onehot \
    --model_type DNN_ENSEMBLE \
    --acq_fn UCB TS GREEDY \
    --batch_size 96 \
    --budget 384 \
    --n_pseudorand_init 96 \
    --runs 50
# Bad model setup
python execute_simulation.py \
    --names GB1 \
    --encodings ESM2 \
    --model_type GP_BOTORCH \
    --acq_fn UCB TS GREEDY \
    --batch_size 10 \
    --budget 50 \
    --runs 50
```

### 2. Multi-Agent LLM Workflows

Run structured multi-agent systems within the Bayesian optimization framework:

```bash
cd ALDE
conda activate alde_env
# Multi-agent workflows integrated with BO
python execute_simulation.py \
    --names GB1 TrpB \
    --encodings onehot \
    --model_type DNN_ENSEMBLE \
    --acq_fn SIMPLEAGENT \
    --batch_size 10 \
    --budget 50 \
    --runs 10
```

### 3. Standalone LLM Optimization

Direct optimization using reasoning models (GPT-5, Qwen3, DeepSeek-R1) without Bayesian optimization framework:

#### Single Campaign Execution

```bash
cd ALDE
conda activate alde_env
# GPT-5 reasoning model
python src/standalone_run.py \
    --protein GB1 \
    --batch_size 10 \
    --total_budget 60 \
    --random_seed 0 \
    --model gpt-5

# Qwen3 via Lambda AI
python src/standalone_run.py \
    --protein TrpB \
    --batch_size 10 \
    --total_budget 60 \
    --random_seed 42 \
    --model qwen

# DeepSeek-R1 via Lambda AI  
python src/standalone_run.py \
    --protein GB1 \
    --batch_size 10 \
    --total_budget 60 \
    --random_seed 0 \
    --model deepseek
```

#### Multiple Campaign Execution

```bash
# Batch execution across multiple seeds and proteins
python execute_standalone.py \
    --names GB1 TrpB \
    --type agent \
    --model qwen \
    --batch_size 10 \
    --total_budget 60 \
    --runs 20 \
    --n_init_pseudorandom 10
    
**Available Models:**
- `gpt-5`: GPT-5 with reasoning (requires OPENAI_API_KEY)
- `qwen`: Qwen3-32B-FP8 via Lambda (requires LAMBDA_API_KEY)  
- `deepseek`: DeepSeek-R1 via Lambda (requires LAMBDA_API_KEY)
- `qwen-blind`, `deepseek-blind`, `gpt-5-blind`,: Domain-agnostic variants

### 4. Fine-tuned LLM Models

The fine-tuning pipeline creates custom models trained on synthetic active learning trajectories generated from ESM2 embeddings. This process follows Algorithm 1 from the paper and involves four main steps: synthetic data generation, BO campaigns, prompt extraction, and model training.

#### Step 1: Generate Synthetic Training Data

```bash
cd ALDE
conda activate trl_env
# Generate artificial fitness datasets from ESM2 embeddings, the TrpB dataset is most complete so we use those sequences and embeddings
python finetuning/gen_data.py \
    data/TrpB/ESM2_x.pt \
    data/TrpB/fitness.csv \
    data/artificial_ESM2 \
    14 \
    0
# This creates 14 synthetic datasets with:
# - ESM2-based fitness landscapes (Algorithm 1 from paper)
# - Log-normal fitness distributions  
# - 10% sequences randomly set to fitness=0
# - Saves to data/artificial_ESM2_0/, artificial_dataset_1/, etc.
```

**Synthetic Data Generation (Algorithm 1):**
- Sample binary mask: 20% active dimensions (p=0.2)
- Log-weights: Normal(μ=-6.5, σ=2) distribution  
- Fitness = ESM2_embeddings · masked_weights
- Log-normal noise + quantile normalization

#### Step 2: Run BO Campaigns on Synthetic Data

```bash
# Run BO campaigns on synthetic datasets to generate training trajectories
cd ALDE
conda activate alde_env
for i in {0..13}; do
    python execute_simulation.py \
        --names artificial_ESM2_$i \
        --encodings onehot \
        --model_type DNN_ENSEMBLE \
        --acq_fn TS \
        --batch_size 10 \
        --budget 400 \
        --n_pseudorand_init 10 \
        --runs 50 \
        --output_path results/synthetic_campaigns/
done
```

#### Step 3: Generate Training Prompts

```bash
cd ALDE
conda activate trl_env
# Extract prompts from top-performing BO trajectories. This matches folders as {folder_path}_* and maps them to {fitness_path}_*/fitness.csv
python gen_prompts.py \
    --folder_path results/synthetic_campaigns/artificial_ESM2 \
    --fitness_path data/artificial_ESM2 \
    --n_folders 14 \
    --n_samples 15 \
    --batch_samples 10 \
    --output_path database/prompts/ \
    --output_name ALDE_training_data
# This creates:
# - SFT training data: ALDE_long_prompts_10.json
# - Samples 15 timepoints per campaign with t^-1 probability weighting
# - Creates chosen/rejected pairs for DPO training
```

**Prompt Generation Features:**
- Evaluates campaigns by recall@0.5% metric
- Samples timepoints with t^-1 probability (favors later cycles)
- Position permutation + sequence shuffling for data augmentation
- Ground truth vs random/accumulated sequence preferences for DPO

#### Step 4: Supervised Fine-Tuning (SFT)
Requires 80GB GPU to train without further improvements to efficency.

```bash
cd ALDE
conda activate trl_env
# Train SFT model (modify paths in train_sft.py)
accelerate launch finetuning/train_sft.py
# Key parameters in script:
# - base_model: "Qwen/Qwen2.5-7B-Instruct" 
# - output: "./models/Qwen-7B-SFT-0814"
# - batch_size: 1 per device, gradient_accumulation: 8
# - learning_rate: 5e-7, epochs: 2
# - max_length: 4096 tokens
# - Uses DataCollatorForCompletionOnlyLM for response-only training
# - Uses ds_config.json deepspeed hyperparameters
```

**SFT Training Configuration:**
- **Model**: Qwen2.5-7B-Instruct (base model)
- **Precision**: bfloat16 + Flash Attention 2  
- **Memory**: Gradient checkpointing, completion-only loss
- **Data Split**: 6/7 train, 1/7 validation
- **Early Stopping**: 3 patience on eval_loss
- **Optimization**: Strong regularization (weight_decay=0.1)

#### Step 5: Direct Preference Optimization (DPO)

```bash
# Train DPO model (requires SFT checkpoint)
accelerate launch train_dpo.py
# Key parameters in script:
# - base_model: "./models/Qwen2-7B-SFT-Long-0731"
# - output: "./models/Qwen2-7B-DPO-Long-0731" 
# - batch_size: 1 per device, gradient_accumulation: 16
# - learning_rate: 5e-7, beta: 0.9, epochs: 2
# - max_prompt_length: 1024, max_completion_length: 48
```

**DPO Training Details:**
- **Preference Data**: Ground truth BO selections vs random/accumulated sequences
- **Reference Model**: Copy of SFT model (strong regularization β=0.9)
- **Memory Optimization**: Filters prompts >1024 tokens, aggressive gradient checkpointing
- **Task-based Split**: Prevents overfitting to specific campaigns

#### Running Fine-tuned Models

```bash
conda activate trl_env
cd ALDE
# Single campaign with SFT model
python src/sft_run.py \
    --protein GB1 \
    --batch_size 10 \
    --total_budget 60 \
    --random_seed 42 \
    --model ./models/Qwen2-7B-SFT-Long-0731/checkpoint-250/

# Conversational DPO model with game interaction
python src/sft_run.py \
    --protein TrpB \
    --batch_size 10 \
    --total_budget 60 \
    --random_seed 42 \
    --model ./models/Qwen2-7B-DPO-Long-0731/checkpoint-150/ \
    --game

# Batch evaluation across multiple seeds
python execute_standalone.py \
    --names GB1 TrpB \
    --type sft \
    --model ./models/Qwen2-7B-SFT-Long-0731/checkpoint-250/ \
    --batch_size 10 \
    --total_budget 60 \
    --runs 20 \
    --n_init_pseudorandom 10

# DPO model with conversational optimization
python execute_standalone.py \
    --names GB1 TrpB \
    --type game \
    --model ./models/Qwen2-7B-DPO-Long-0731/checkpoint-150/ \
    --batch_size 10 \
    --total_budget 60 \
    --runs 20
```

## Experimental Campaigns

The paper evaluates 4 main campaign types:

### Small-Good 1 (GB1, DNN-onehot)
```bash
python execute_simulation.py --names GB1 --encodings onehot --model_type DNN_ENSEMBLE \
    --batch_size 10 --budget 50 --n_pseudorand_init 10 --runs 10
```

### Small-Bad (GB1, GP-ESM2)  
```bash
python execute_simulation.py --names GB1 --encodings ESM2 --model_type GP_BOTORCH \
    --batch_size 10 --budget 50 --n_pseudorand_init 10 --runs 10
```

### Big-Good (GB1, DNN-onehot)
```bash  
python execute_simulation.py --names GB1 --encodings onehot --model_type DNN_ENSEMBLE \
    --batch_size 96 --budget 384 --n_pseudorand_init 96 --runs 50
```

### Small-Good 2 (TrpB, DNN-onehot)
```bash
python execute_simulation.py --names TrpB --encodings onehot --model_type DNN_ENSEMBLE \
    --batch_size 10 --budget 50 --n_pseudorand_init 10 --runs 10
```

## Data

### Protein Targets

- **GB1**: 4-site epistatic region (V39, D40, G41, V54) of protein G domain B1
  - Immunoglobulin-binding domain from Streptococcal bacteria
  - 160,000 possible variants assessed for IgG-Fc binding
  - Wildtype fitness ~0.1

- **TrpB**: 4-amino acid motif (V183, F184, V227, S228) of tryptophan synthase β-subunit  
  - Enzyme catalyzing L-tryptophan synthesis
  - Fitness linked to enzyme activity in E. coli Trp auxotroph
  - Wildtype fitness ~0.4

### Encodings
- **onehot**: One-hot encoding of amino acids (optimized)
- **ESM2**: ESM2 protein language model embeddings  
- **AA**: Simple amino acid encoding
- **georgiev**: Georgiev physicochemical properties

## Analysis

### Results Processing

```bash
cd ALDE/analysis
conda activate alde_env
# Tabulate results from all methods
python tabulate_results.py \
    --results-dir ../results/campaign_name \
    --output data/results_summary.csv \
    --max_seed_index 50
# Visualize and analyze
jupyter notebook visualization.ipynb
```

### Performance Metrics

- **Primary**: Highest fitness discovered across cycles
- **Secondary**: Recall at 0.5% cutoff (top sequences discovered)
- **Statistical**: Bootstrap confidence intervals and significance testing

## Method Categories

The analysis pipeline categorizes methods into:

- **Statistical**: Traditional BO (GREEDY, UCB, TS)
- **Agent**: Multi-agent LLM systems (AGENT, SIMPLEAGENT) 
- **Standalone**: Direct LLM optimization (GPT-5, Qwen3, DeepSeek-R1)
- **Finetuned**: Custom-trained models (SFT, DPO, Game)
- **Baseline**: Random/control methods

## Production Usage Guidelines

### Large-Scale Experiments

**Batch Execution:**
1. **Traditional BO**: Use `execute_simulation.py` for statistical BO baselines
2. **LLM Methods**: Use `execute_standalone.py` for standalone and fine-tuned models
3. **Set appropriate run counts**: 20+ runs for statistical significance
4. **Monitor resources**: Track API costs for cloud models, GPU usage for local models

**Environment Configuration:**
```bash
# Essential API keys for different models
export OPENAI_API_KEY="your_openai_key"        # GPT-5
export LAMBDA_API_KEY="your_lambda_key"        # Qwen3, DeepSeek-R1  
export ANTHROPIC_API_KEY="your_anthropic_key"  # Claude (for structured output)
export LANGSMITH_API_KEY="your_langsmith_key"  # Experiment tracking

# Set appropriate limits
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="ALDE_Production"
```

**Budget and Resource Planning:**
- **Small campaigns**: `batch_size=10, total_budget=60` (~6 cycles)
- **Medium campaigns**: `batch_size=20, total_budget=100` (~5 cycles)  
- **Large campaigns**: `batch_size=50, total_budget=200` (~4 cycles)
- **API costs**: ~$2-5 per GPT-5 run, ~$0.50-1 per Qwen3/DeepSeek run

**Output Organization:**
```
results/
├── {experiment_name}/
│   ├── {protein}/
│   │   ├── onehot/                    # Required for compatibility
│   │   │   ├── campaign_history.json  # Cycle-by-cycle results
│   │   │   ├── *_indices.pt          # Selected sequence indices
│   │   │   ├── cycle_*_summary.json  # Individual cycle data
│   │   │   └── graph.png             # LangGraph workflow diagram
│   │   └── execute_standalone.py     # Script copy for reproducibility
│   └── timeit.txt                    # Runtime tracking
```

### Method Selection Guidelines

**Traditional Statistical Methods:**
- **Bayesian Optimization**: Proven baseline, model-agnostic, fast execution
- **Acquisition functions**: UCB (exploration), TS (Thompson sampling), GREEDY (exploitation)

**LLM-based Methods:**
- **Multi-agent workflows**: Structured reasoning within BO framework
- **Standalone models**: Direct optimization, supports GPT-5, Qwen3, DeepSeek-R1
- **Fine-tuned models**: Custom-trained for protein optimization tasks
- **Resource considerations**: API costs vs local GPU requirements

### Performance Optimization

**For High-Throughput Experiments:**
1. **Local model deployment**: Avoid API rate limits and costs
2. **Batch size optimization**: Larger batches reduce total cycles
3. **Parallel execution**: Run multiple proteins/conditions simultaneously
4. **Resource monitoring**: Track GPU memory and API quotas

**Common Considerations:**
- **Statistical Methods**: Fast, reliable, no API dependencies
- **LLM Methods**: May require API management, higher computational costs
- **Local vs Cloud**: Trade-off between resource control and infrastructure requirements
- **Reproducibility**: Save random seeds, model versions, and environment configurations

### Statistical Analysis Preparation

**Required Runs for Publication:**
- **Preliminary experiments**: 10-15 runs per condition
- **Final benchmarks**: 20-50 runs per condition  
- **Comparative studies**: Ensure balanced sampling across methods

**Expected Output Metrics:**
- **Max fitness discovered**: Primary performance measure
- **Recall@0.5%**: Secondary diversity measure
- **Convergence rate**: Cycles to reach performance plateau
- **Runtime statistics**: Model inference and total wall time

## Key Features

### Multi-Agent System (agent.py)
- Strategy generation using reasoning LLMs
- Tool-based sequence querying with regex patterns
- BLOSUM62 similarity filtering for diversity
- Structured output validation with Pydantic
- LangGraph workflow orchestration

### Fine-tuning Pipeline
- **Synthetic Data Generation**: ESM2 embeddings → sparse weight vectors → log-normal fitness landscapes
- **Training Data Creation**: Top 10% campaigns → timepoint sampling (t^-1) → position permutation + shuffling
- **SFT Training**: Response-only loss masking, completion-only learning, gradient checkpointing
- **DPO Optimization**: Ground truth vs random preferences, strong regularization (β=0.9)
- **Memory Optimization**: Flash Attention 2, bfloat16, prompt filtering, aggressive checkpointing

### Evaluation Framework
- Standardized result formats across all methods
- Statistical significance testing with bootstrapping  
- Comprehensive visualization and analysis tools
- Campaign history tracking and reproducibility
