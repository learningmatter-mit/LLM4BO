# BO_LLM: Large Language Models for Bayesian Optimization
**AUTHOR:** Mattias Akke  
**CONTACT:** [akkem@mit.edu](mailto:akkem@mit.edu), [mattias.akke@gmail.com](mailto:mattias.akke@gmail.com)

This repo intend to benchmark several different LLM methods for Bayesian Optimisation across two different domains. 

## Project Structure

### `ALDE/`
Active Learning with LLMs for protein optimization. Benchmarks traditional Bayesian Optimization methods against LLM-based approaches for optimizing 4-amino acid protein motifs. Includes generative reasoning models (GPT-5, Qwen3, DeepSeek-R1), agentic workflows, and fine-tuned models trained on synthetic optimization trajectories. See `ALDE/README.md`.

### `mols/`
Molecular property prediction active learning framework. Modular benchmarking of LLM-based selectors vs traditional methods for molecular discovery, with multiple oracle models, selectors, and ablation tools. See `mols/README.md`.

### `art/`
Experimentall use of Auxiliary research tooling (ART) for GRPO-trained small models for BO and AL. Trains models on the abstract goal "do well" in BO. Includes standalone agents (`art/AL_LLM.py`), rollout/training scripts (`art/art_scripts/`), and fine-tuning experiments (`art/sft/`).

### Other top-level files
- `file_summaries.md`: Aggregated file summaries

Each subdirectory contains its own detailed README with installation instructions, usage examples, and technical documentation.
