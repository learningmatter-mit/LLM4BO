#!/usr/bin/env python3
"""
Execute standalone LLM agent for protein active learning simulations.

This script runs the standalone_agent.py multiple times with different seeds
and stores the results in the same format as execute_simulation.py for
compatibility with the analysis pipeline.
"""

import argparse
import numpy as np
import torch
import random
import time

import warnings
import json
import shutil
from pathlib import Path
import sys

# Add src to path for imports (robust to CWD)
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import src.objectives as objectives
import src.utils as utils
try:
    from standalone_run import StandaloneAgent
except ImportError:
    try:
        from src.standalone_run import StandaloneAgent  # fallback if needed
    except Exception:
        print("Warning: StandaloneAgent not available")
        StandaloneAgent = None
try:
    from sft_run import sft_run
except ImportError:
    try:
        from src.sft_run import sft_run  # fallback if needed
    except Exception:
        print("Warning: sft_run not available")
        sft_run = None

def run_single_agent(args_tuple):
    """Run a single standalone agent instance."""
    protein, batch_size, total_budget, seed, output_dir, agent_type, model, n_init_pseudorandom = args_tuple
    
    
    print(f"Initializing {agent_type} agent...")
    print(f"Protein: {protein}")
    print(f"Batch size: {batch_size}")
    print(f"Total budget: {total_budget}")
    print(f"Random seed: {seed}")
    
    # Generate initial random indices like execute_simulation.py
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load objective to get initial random samples
    obj = objectives.Combo(protein, "onehot")  # Use AA encoding for agent compatibility
    n_pseudorand_init = n_init_pseudorandom  # Use batch_size as initial data size
    
    start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)
    # Initialize and run agent
    if agent_type == "sft" or agent_type == "game":
        print(agent_type=="game")
        agent = sft_run(
            protein=protein,
            batch_size=batch_size,
            total_budget=total_budget,
            random_seed=seed,
            model=model,
            output_dir=str(output_dir),
            init_data=start_indices,
            game=(agent_type == "game")
        )
    elif agent_type == "agent":
        agent = StandaloneAgent(
            protein=protein,
            batch_size=batch_size,
            total_budget=total_budget,
            random_seed=seed,
            model=model,
            output_dir=str(output_dir)
        )
    
    start_time = time.time()
    agent.run_campaign()
    end_time = time.time()
    #Append runtime to timeit.txt
    with open("timeit.txt", "a") as f:
        f.write(f"{protein},{agent_type},{model},{batch_size},{total_budget},{seed},{end_time - start_time:.2f}\n")
    print(f"Completed {agent_type} agent run with seed {seed} in {end_time - start_time:.2f} seconds")
    return seed


def main():
    """Main function to execute standalone agent runs."""
    parser = argparse.ArgumentParser(description="Execute standalone LLM agent for protein active learning")
    parser.add_argument("--names", nargs="+", default=["GB1", "TrpB"], help="Protein names to optimize")
    parser.add_argument("--encodings", nargs="+", default=["onehot"], help="Encodings to use, for compatibility with execute_simulation.py")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of sequences per cycle")
    parser.add_argument("--total_budget", type=int, default=60, help="Total experimental budget")
    parser.add_argument("--output_path", type=str, default='results/standalone_simulations/', help="Output directory")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--seed_index", type=int, default=0, help="Starting seed index")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--type", type=str, default="agent", choices=["agent", "sft", "game"], help="Agent type: agent (StandaloneAgent), sft (AL_LLM), game (AL_LLM)")
    parser.add_argument("--model", type=str, default="./models/Qwen2-0_5B-DPO-0727", help="Model path for DPO agent (qwen, deepseek, gpt-5, qwen3-blind etc, or local path)")
    parser.add_argument("--n_init_pseudorandom", type=int, default=10, help="Number of initial pseudorandom sequences to sample")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Arguments:", args)
    
    warnings.filterwarnings("ignore")
    
    # Load seeds from file (same as execute_simulation.py)
    seeds = []
    with open(SRC_DIR / 'rndseed.txt', 'r') as f:
        lines = f.readlines()
        for i in range(args.runs):
            seed_idx = args.seed_index + i
            if seed_idx < len(lines):
                line = lines[seed_idx].strip('\n')
                seeds.append(int(line))
                if args.verbose:
                    print(f'Run {i+1}, seed index: {seed_idx}, seed: {int(line)}')
            else:
                # Generate random seed if we run out of predefined seeds
                seed = random.randint(1, 1000000)
                seeds.append(seed)
                if args.verbose:
                    print(f'Run {i+1}, generated seed: {seed}')
    # Process each protein-encoding combination
    for protein in args.names:
        for encoding in args.encodings:
            if args.verbose:
                print(f"\nProcessing {protein}/{encoding} with {args.type} agent")
            
            # Create output directory with agent type
            output_dir = Path(args.output_path) / f"{protein}" / encoding
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy this script to the output directory for reproducibility
            shutil.copy(__file__, output_dir)
            if args.verbose:
                print(f'Script stored in {output_dir}')
            
            # Validate protein data exists
            data_file = ROOT_DIR / f"data/{protein}/fitness.csv"
            if not data_file.exists():
                print(f"Warning: Data file not found: {data_file}")
                continue
            
            # Prepare arguments for agent runs
            agent_args = []
            for i, seed in enumerate(seeds):
                agent_args.append((protein, args.batch_size, args.total_budget, seed, output_dir, args.type, args.model, args.n_init_pseudorandom))
            
            # Run agents
            if args.verbose:
                print(f"Running {len(agent_args)} {args.type} agent instances...")
            
            start_time = time.time()
            
            # Sequential execution
            for agent_arg in agent_args:
                run_single_agent(agent_arg)
            
            total_time = time.time() - start_time
            print(f'Total runtime for {protein}/{encoding} with {args.type}: {total_time:.2f} seconds')
            print(f'Results saved in {output_dir}')


if __name__ == "__main__":
    main() 