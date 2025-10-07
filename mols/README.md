# Active Learning Framework for Molecular Property Prediction

A modular and extensible active learning framework for molecular property prediction using SMILES strings. The framework is designed to benchmark and test the use of LLM-based selectors in active learning, with a focus on reproducibility, extensibility, and robust evaluation.

Data assembled by:  
Gorantla, R., Kubincová, A., Suutari, B., Cossins, B. P., & Mey, A. S. J. S. (2024). *Benchmarking active learning protocols for ligand-binding affinity prediction*. Journal of Chemical Information and Modeling, 64(6), 1831–1849. [https://pubs.acs.org/doi/10.1021/acs.jcim.4c00220](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00220)

## Project Structure

```
mols/
├── data/               # Dataset CSV files (TYK2, USP7, D2R, Mpro included, but code is general)
├── aloracles/          # Oracle model implementations (one per file)
├── alselectors/        # Active learning selector strategies (one per file)
├── output/             # Training results, checkpoints, and plots
├── utils/              # Base classes and utilities
└── bin/                # Command-line scripts (AL loop, benchmarking, etc.)
```

## Usage

### Running an Active Learning Experiment

```bash
python bin/AL.py \
    --model RandomForestOracle \
    --selector LLMFeaturiseQwenSelector \
    --data data/TYK2.csv \
    --batch_size 60 \
    --initial_size 60 \
    --output_dir output/training/TYK2_RF_LLMFat_60_60_0_20240601 \
    --model_kwargs '{"n_estimators": 100}' \

# LLM-based workflow selector
python bin/AL.py \
    --model RandomForestOracle \
    --selector LLMWorkflowSelector \
    --data data/TYK2.csv \
    --batch_size 60 \
    --initial_size 60 \

# UCB with specific parameters
python bin/AL.py \
    --model RandomForestOracle \
    --selector UCBSelector \
    --data data/TYK2.csv \
    --batch_size 60 \
    --initial_size 60 \
    --selector_kwargs '{"beta": 4.0, "uncertainty_method": "inverse"}' \
```

### Required Arguments

- `--model`: Oracle model class name (must be in aloracles/)
- `--selector`: Selector class name (must be in alselectors/)
- `--data`: Path to dataset CSV file

### Optional Arguments

- `--batch_size`: Samples per AL cycle (default: 60)
- `--initial_size`: Initial training set size (default: 60)
- `--output_dir`: Results directory path
- `--model_kwargs`: JSON string of model-specific parameters
- `--selector_kwargs`: JSON string of selector-specific parameters
- `--bad_start`: Whether to start  with a bad or random start (default: False)

## Data Format

Input CSV files should have the following format:
```
SMILES,label1,label2,label3,...
CC(=O)OC1=CC=CC=C1C(=O)O,0.5,0.3,0.8,...
...
```

The repository includes datasets for four protein targets (TYK2, USP7, D2R, Mpro), but the code is not limited to these and can be used with any dataset in the above format.

## Output

The framework generates:
1. Training results JSON file with metrics per AL cycle
2. Model checkpoints (for resuming experiments)
3. Training summary and UMAP plots

### Output Directory Structure

Results are saved to:
```
output/training/[DataName]/[ModelName]_[SelectorName]_[initial_batch_size]_[batch_size]_[index]_YYYYMMDD/
```
where `[index]` is incremented for each repeat with the same configuration.

### Results JSON Format

```json
{
    "experiment_config": {
        "model": "RandomForestOracle",
        "selector": "LLMFatSelector",
        "data_path": "data/TYK2.csv",
        "batch_size": 60,
        "total_budget": 600,
        "initial_size": 60,
        "random_seed": 42,
        "data_config": {"target": "affinity"},
        "hyperparameters": {"model": {...}, "selector": {...}}
    },
    "al_cycles": [
        {
            "cycle": 0,
            "samples_selected": [...],
            "total_labeled": 60,
            "performance_metrics": {...},
            "computational_time": 1.23,
            "model_state": "path/to/checkpoint"
        },
        ...
    ]
}
```

## Available Selectors

The framework includes several selector implementations:

### Traditional Selectors
- **`RandomSelector`**: Random sampling baseline
- **`ExploitSelector`**: Exploitation-only (highest predicted values)
- **`UCBSelector`**: Upper Confidence Bound with configurable parameters
- **`ThompsonSelector`**: low-rank Thompson sampling 
- **`EISelector`**: Expected improvement
**`EpsExploit`**: $\epsilon$-exploit with 5 % chance of random selection

### LLM-based Selectors  
- **`LLMFatSelector`**: Direct LLM selection with basic prompting
- **`LLMFeaturiseSelector`**: LLM with feature analysis and chemical reasoning
- **`LLMWorkflowSelector`**: Multi-agent LLM workflow with tool access (supports ablation studies)

Each selector supports different `selector_kwargs` parameters - see individual selector files for specific options.

## LLM API Key Setup

Some selectors (e.g., `LLMFatSelector`, `LLMFeaturiseSelector`, `LLMWorkflowSelector`) require access to external LLM APIs. You must set the following environment variables before running experiments that use these selectors:

- For OpenAI-based selectors:
  ```bash
  export OPENAI_API_KEY=sk-...
  ```
- For Anthropic-based selectors:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

The framework will automatically fetch these keys from the environment.

## Evaluation Metrics

The framework evaluates both regression and classification performance:
- **Regression:** RMSE, R², Spearman correlation
- **Classification:** Recall and F1 for top 2% and 5% binders

## Visualization

- **UMAP Visualization:** The framework generates UMAP plots to visualize the exploration of chemical space during active learning. These plots show both the full dataset and the progression of selected samples across cycles.
- **Training Metrics:** Plots of performance metrics over AL cycles are generated for each experiment and for benchmarks.

## Checkpointing and Resuming

Experiments can be checkpointed and resumed from any cycle. Checkpoints include the current state of the labeled/unlabeled pools and all experiment configuration.

```bash
python bin/AL.py \
    --restore_checkpoint output/training/TYK2/GPRegOracle_ExploitSelector_60_60_1_20250618/Training_results.json \
    --cycle -1 \
```

## Extending the Framework

### Adding a New Oracle Model

1. Create a new file in `aloracles/` (e.g., `MyModel.py`)
2. Inherit from `utils.Model` base class
3. Implement required methods:
   - `__init__(random_seed, **kwargs)`
   - `train(X_train, y_train, random_seed, **kwargs)`
   - `predict(X, random_seed)`

### Adding a New Selector

1. Create a new file in `alselectors/` (e.g., `MySelector.py`)
2. Inherit from `utils.Selector` base class
3. Implement required methods:
   - `__init__(batch_size, random_seed, **kwargs)`
   - `select(predictions, confidence_scores, training_data, unlabeled_data, random_seed, **kwargs)`

## UCB Sensitivity Experiments

The framework supports systematic studies of Upper Confidence Bound (UCB) parameters through the `UCBSelector`. Different beta values control the exploration-exploitation trade-off:

```bash
# UCB with different beta values
python bin/AL.py --model GPRegOracle --selector UCBSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"beta": 1.0}'
python bin/AL.py --model GPRegOracle --selector UCBSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"beta": 4.0}'
python bin/AL.py --model GPRegOracle --selector UCBSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"beta": 16.0}'
```

### UCB Parameters
- `beta`: Exploration-exploitation trade-off parameter (default: 4.0)
- `uncertainty_method`: Method to compute uncertainty from confidence scores
  - `'inverse'`: σ = 1/confidence - 1 (default)
  - `'sqrt_inverse'`: σ = sqrt(1/confidence - 1)  
  - `'log_inverse'`: σ = -log(confidence)
- `confidence_threshold`: Minimum confidence threshold (default: 0.0)

## Ablation Studies

The framework supports ablation studies to evaluate the contribution of different tools in LLM-based selectors. Use the `LLMWorkflowSelector` with disable flags:

```bash
# Disable UCB tool
python bin/AL.py --model RandomForestOracle --selector LLMWorkflowSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"disable_ucb": true}'

# Disable SMARTS substructure search
python bin/AL.py --model RandomForestOracle --selector LLMWorkflowSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"disable_smarts": true}'

# Disable Tanimoto similarity calculations
python bin/AL.py --model RandomForestOracle --selector LLMWorkflowSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"disable_tanimoto": true}'

# Multiple ablations
python bin/AL.py --model RandomForestOracle --selector LLMWorkflowSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --selector_kwargs '{"disable_ucb": true, "disable_smarts": true}'
```

### Ablation Flags
- `disable_ucb`: Disable Upper Confidence Bound functionality
- `disable_smarts`: Disable SMARTS substructure search functionality  
- `disable_tanimoto`: Disable Tanimoto similarity functionality

When tools are disabled, the LLM agent receives modified prompts indicating which tools are unavailable and must adapt its selection strategy accordingly.

## Benchmarking

To benchmark a model/selector combination, use:
```bash
python bin/benchmark.py --model RandomForestOracle --selector RandomSelector --data data/TYK2.csv --batch_size 60 --initial_size 60 --num_repeats 3
```
Each repeat uses a different random seed (probabilistic benchmarking). The script will automatically create a benchmark file of all files matching the creation conditions. So running benchmark --num_repeats 1 and then --num_repeats 2 will create a file averaging 3 trajectories.

## Output Interpretation Guide

- **Training_results.json**: Contains experiment configuration, all hyperparameters, and per-cycle metrics.
- **Plots**: UMAP and metric plots are saved in the output directory for each experiment.
- **Checkpoints**: Allow resuming experiments from any cycle.

