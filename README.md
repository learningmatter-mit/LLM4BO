# LLM4BO: Large Language Models for Bayesian Optimization
**CONTACT:** [mattias.akke@gmail.com](mailto:mattias.akke@gmail.com)

This repository aims to benchmark several different LLM methods for Bayesian Optimization across two domains. 

## Project Structure

### `ALDE/`
Bayesian Optimization of protein motifs with LLMs. Benchmark traditional Bayesian Optimization methods against LLM-based approaches for optimizing 4-amino acid protein motifs. Includes generative reasoning models (GPT-5, Qwen3, DeepSeek-R1), agentic workflows, and fine-tuned models trained on synthetic optimization trajectories. See `ALDE/README.md`.

### `mols/`
Molecular property prediction active learning framework. Modular benchmarking of LLM-based selectors vs traditional methods for molecular discovery, with multiple oracle models, selectors, and ablation tools. See `mols/README.md`.

Each subdirectory contains its own detailed README with installation instructions, usage examples, and technical documentation.

<p align="center">
  <img src="https://raw.githubusercontent.com/learningmatter-mit/LLM4BO/refs/heads/main/llm-al_overview-1.png" alt="LLM4BO Overview" width="600"/>
</p>

