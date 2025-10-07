#!/usr/bin/env python3
"""
Standalone LLM Agent for Protein Active Learning Campaign

This script runs a complete active learning campaign using a simplified LLM agent
that directly controls sequence selection without complex multi-agent workflows.
"""

import argparse
import os
import json
import random
import time
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing utilities for data loading
import sys
sys.path.append('src')
import src.objectives as objectives

class sft_run(object):
    """Simplified standalone LLM agent for protein active learning."""
    
    def __init__(
        self, 
        protein: str, 
        batch_size: int, 
        total_budget: int, 
        random_seed: int, 
        model: str = "Qwen/Qwen2.5-0.5B-Instruct", #"./models/Qwen2-0_5B-DPO-0727", 
        output_dir: str = "results/", 
        init_data: torch.Tensor = None,
        game: bool = False
    ):
        if model[0] == "." or model[0] == "/":
            model = os.path.abspath(model)
        print(f"Loading model from {model}")
        self.protein = protein
        self.batch_size = batch_size
        self.total_budget = total_budget 
        self.random_seed = random_seed
        self.model_name = model
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, local_files_only=True, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = 64
        self.temperature = 1
        self.budget = self.batch_size #96 #self.batch_size
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.game = game
        # Load protein data
        self.all_sequences = pd.read_csv(f"data/{protein}/fitness.csv")
        self.all_sequences.columns = ["sequence", "fitness"]
        self.all_sequences["sequence"] = [" ".join(s) for s in self.all_sequences["sequence"]]
        self.all_sequences["fitness"] = (self.all_sequences["fitness"] - self.all_sequences["fitness"].min()) / (self.all_sequences["fitness"].max() - self.all_sequences["fitness"].min())
        self.cutoff = self.all_sequences["fitness"].quantile(0.995)
        print(f"Cutoff: {self.cutoff}")
        # Fix: Create proper index column and seq2ind mapping
        self.all_sequences.reset_index(inplace=True)
        self.seq2ind = self.all_sequences.set_index("sequence")["index"].to_dict()
        self.n_total = len(self.all_sequences)
        self.global_step = 0
        print(f"Loaded {protein} dataset: {self.n_total} sequences")
        print(f"Fitness range: [{self.all_sequences['fitness'].min():.4f}, {self.all_sequences['fitness'].max():.4f}]")

        self.defult_prompt_format = """You are an expert bioinformatician conducting active learning to maximize the fitness of a 4-residue long protein motif (20^4=160000 possible sequences) using a budget of {budget} more sequences.

TASK: Given the current data (tested motifs and their respective fitness values, higher is better), generate ONE novel 4-residue sequence that maximises information gain.

CURRENT DATA (shuffled):
{protein_sequences}

Available amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
Answer only with the next sequence:
"""
        self.long_prompt_format = """You are an expert bioinformatician conducting active learning to maximize the fitness of a 4-residue long protein motif (20^4=160000 possible sequences) using a budget of {budget} more sequences.

TASK: Given the current data, generate 10 novel 4-residue sequences that maximises information gain.

CURRENT DATA (shuffled):
{protein_sequences}

Available amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

Answer only with the next 10 sequences, comma separated:
"""

        self.cycle = 0
        if protein == 'GB1':
            wt = {"sequence": "VDGV", "fitness": 0.11413}
        elif protein == 'TrpB':
            wt = {"sequence": "VFVS", "fitness": 0.408074}
        
        if init_data is not None:
            self.validated_results = self.all_sequences.iloc[init_data].copy()
        else:
            self.validated_results = pd.DataFrame([wt])
        
        # Initialize last_validated_results for tracking cycle changes
        self.last_validated_results = self.validated_results.copy()
        
        self.results_dir = Path(output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.all_indices = init_data.tolist() if init_data is not None else []
        # Add campaign history tracking for JSON output
        self.campaign_history = []  # Store cycle-by-cycle results in expected format

    def _generate(self, prompt):
        prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(inputs['input_ids'], max_new_tokens=self.max_new_tokens, temperature=self.temperature, top_k=20, top_p=0.8, repetition_penalty=1.05, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        self.global_step += 1
        print(f"Generated {len(generated_ids[0])} tokens")
        input_length = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_length:]
        ans = self.tokenizer.decode(new_tokens[0].tolist(), skip_special_tokens=True).strip("\n")
        print(f"Generated: {ans}")
        return ans
    
    def _reshuffle_prompt(self):
        prompt = self.long_prompt_format.format(budget = self.total_budget-len(self.validated_results), protein_sequences=self.validated_results.sort_values(by="fitness", ascending=False)[["sequence", "fitness"]].head(400).round(3).sample(frac=1).to_string(index=False))
        self.prompt = [{"role": "user", "content": prompt}]
        return self.prompt
    
    def run_campaign(self):
        if self.game:
            self._run_game_campaign()
        else:
            self._run_campaign()


    def _run_game_campaign(self):
        messages = [{"role": "system", "content": f"You are an expert bioinformatician."}]
        initial_prompt = f"""You are conducting active learning to find the highest fitness 4-residue protein sequences from 160,000 possibilities (20^4). 

GOAL: Maximize information gain to discover sequences with the highest possible fitness values.

PROCESS: 
1. Generate {self.batch_size} sequences
2. Receive measured fitness results for each sequence
3. Use results to inform next batch generation
4. Repeat for {self.total_budget//self.batch_size - 1} remaining cycles

TASK: Generate {self.batch_size} novel 4-residue sequences that will provide maximum information about high-fitness regions.

CURRENT DATA:
{self.validated_results.round(3)[["sequence", "fitness"]].rename(columns={"sequence":"Combo"}).to_string(index=False)}

Amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

Output only the {self.batch_size} sequences, comma-separated, no other text:"""
        print(initial_prompt)
        messages.append({"role": "user", "content": initial_prompt})
        while len(self.validated_results) < self.total_budget and self.cycle < 7:
            self.global_step += 1
            reply = self._get_batch_selections_long(messages, shuffle=False)
            self.validate_sequences(reply)
            self.cycle += 1
            self.save_cycle_results()
            messages.append({"role": "assistant", "content": ','.join(reply)})
            feedback_lines = [f"{seq} {self.validated_results.loc[self.seq2ind[seq], 'fitness']:.3f}" for seq in reply]
            messages.append({"role": "user", "content": f"""CYCLE {self.cycle} RESULTS:
{chr(10).join(feedback_lines)}

Generate the next {self.batch_size} sequences for cycle {self.cycle + 1}:"""})
        
        
    def _run_campaign(self):
        """Run the campaign"""
        batch_selections = []
        print(f"Running campaign with total budget {self.total_budget}, current budget {len(self.validated_results)}")
        while len(self.validated_results) < self.total_budget and self.cycle < 50:
            if self.global_step > 1000:
                print("Global step limit reached")
                break
            if "long" in self.model_name.lower():
                self.prompt = self.long_prompt_format.format(budget = self.total_budget-len(self.validated_results), protein_sequences=self.validated_results.sort_values(by="fitness", ascending=False)[["sequence", "fitness"]].head(250).round(3).sample(frac=1).to_string(index=False))
                self.prompt = [{"role": "user", "content": self.prompt}]
            elif "shorter" in self.model_name.lower():
                self.prompt = self.shorter_prompt_format.format(budget = self.total_budget-len(self.validated_results), protein_sequences=self.validated_results.sort_values(by="fitness", ascending=False)[["sequence", "fitness"]].head(10).round(3).sample(frac=1).to_string(index=False))
            else:
                self.prompt = self.defult_prompt_format.format(budget = self.total_budget-len(self.validated_results), protein_sequences=self.validated_results.sort_values(by="fitness", ascending=False)[["sequence", "fitness"]].head(50).round(3).sample(frac=1).to_string(index=False))
            if self.global_step == 0:
                print(self.prompt)
            if "long" in self.model_name.lower():
                batch_selections = self._get_batch_selections_long(self.prompt)
            else:   
                batch_selections = self._get_batch_selections(self.prompt)
            self.validate_sequences(batch_selections)
            self.cycle += 1
            self.save_cycle_results()
    
    def _get_batch_selections_long(self, prompt, shuffle=True):
        response = self._generate(prompt)
        batch_selections = response.strip('\n').strip().split(",")
        n_attempts = 0
        temperature = self.temperature
        selections = []
        if sum([(not self._test_sequence(s)) or (s in selections) for s in batch_selections]) == 0 and len(batch_selections) == len(set(batch_selections)):
            selections.extend(batch_selections)
        else:
            print(f"Invalid batch selections: {batch_selections}")
        while len(selections) < self.budget:
            if shuffle:
                prompt = self._reshuffle_prompt()
            response = self._generate(prompt).strip()
            batch_selections = response.strip('\n').split(",")
            if sum([(not self._test_sequence(s)) or (s in selections) for s in batch_selections]) == 0:
                selections.extend(batch_selections)
            else:
                self.temperature = min(self.temperature + 0.1, 1.5)
                print(f"Invalid batch selections: {batch_selections}")
            n_attempts += 1
            if n_attempts > 50:
                print(f"Max attempts reached: {n_attempts}")
                return selections
        print(f"Temperature increased from {temperature} to {self.temperature}")
        self.temperature = temperature
        return selections[:self.budget]
            
    def _get_batch_selections(self, prompt_format):
        selections = []
        original_temperature = self.temperature
        n_attempts = 0
        prompt = prompt_format.format(previous_sequence="")
        while len(selections) < self.batch_size:
            response = self._generate([{"role": "user", "content": prompt}])
            while response in selections or not self._test_sequence(response):
                print(f"Invalid response: {response}")
                n_attempts += 1
                if n_attempts > 50:
                    print(f"Max attempts reached: {n_attempts}")
                    return selections
                response = self._generate([{"role": "user", "content": prompt}]).strip('\n')
                self.temperature = min(self.temperature + 0.05, 1.5)
            print(f"Valid response: {response}")
            selections.append(response)
            prev_df = pd.DataFrame({"sequence": selections, "fitness": ["0.000"]*len(selections)})
            prompt = prompt_format.format(previous_sequence=prev_df.to_string(index=False, header=False))
        if original_temperature != self.temperature:
            print(f"Temperature increased from {original_temperature} to {self.temperature}")
        self.temperature = original_temperature
        return selections
    
    def _test_sequence(self, sequence):
        if len(sequence) != 7:
            print(f"Invalid sequence length")
            return False
        if sequence in self.validated_results["sequence"].values:
            print(f"Sequence already validated")
            return False
        if not self.seq2ind.get(sequence):
            print(f"Sequence not found in database")
            self.validated_results = pd.concat([self.validated_results, pd.DataFrame({"sequence": [sequence], "fitness": [0.0]})])
            return False
        return True
    
    def validate_sequences(self, sequences: List[str]) -> bool:
        """Look up fitness values for selected sequences and update tracking."""
        if len(sequences) == 0 or len(sequences) > self.budget or len(sequences) != len(set(sequences)) or sum([not self._test_sequence(s) for s in sequences]) > 0:
            print(f"Invalid sequences: {sequences}")
            return False
        indicies = [self.seq2ind[s] for s in sequences]
        subdf = self.all_sequences.iloc[indicies].copy()  
        
        # Track last validated results before updating
        self.last_validated_results = subdf
        
        self.validated_results = pd.concat([self.validated_results, subdf])
        self.all_indices.extend(indicies)
    
        # Print progress
        mean_fitness = np.mean(subdf["fitness"])
        max_fitness = np.max(subdf["fitness"])
        overall_max = np.max(self.validated_results["fitness"])
        recall_05pc = sum(self.validated_results["fitness"] > self.cutoff)
        
        print(f"Cycle {self.cycle + 1} results from {len(subdf)} sequences - Mean: {mean_fitness:.4f}, Max: {max_fitness:.4f}, Overall Max: {overall_max:.4f}, Recall@99.5pc: {recall_05pc:.4f}")
        
        return True
    
    def save_cycle_results(self):
        """Save results in the same format as cursorrules example."""
        
        # Save indices as .pt file (matching optimize.py format)
        indices_tensor = torch.tensor(self.all_indices, dtype=torch.long)
        torch.save(indices_tensor, self.results_dir / f"cycle_{self.cycle}_indices.pt")
        
        # Also save in execute_simulation.py compatible format
        if hasattr(self, 'random_seed'):
            agent_tag = "LLM"
            output_file = self.results_dir / f"AGENT_{agent_tag}-DO-0-RBF-AGENT-[1, 1]_{self.random_seed}indices.pt"
            torch.save(indices_tensor, output_file)
        
        # Get current cycle data from last_validated_results
        overall_mean_fitness = self.validated_results["fitness"].mean()
        overall_max_fitness = self.validated_results["fitness"].max()
        cycle_mean_fitness = self.last_validated_results["fitness"].mean()
        cycle_max_fitness = self.last_validated_results["fitness"].max()
        # Create strategy description
        if self.cycle == 0:
            strategy_desc = "Initial training data"
        else:
            strategy_desc = f"LLM agent selection cycle {self.cycle}: strategic sequence selection based on fitness landscape analysis"
        
        # Build cycle entry in expected format
        cycle_entry = {
            "cycle": self.cycle,
            "summary": f"Cycle {self.cycle} completed with {len(self.last_validated_results)} sequences validated. Mean fitness: {cycle_mean_fitness:.4f}, Max fitness: {cycle_max_fitness:.4f}",
            "mean_fitness": overall_mean_fitness,
            "max_fitness": overall_max_fitness,
            "oracle_rmse": float('nan'),  # Set to NaN as requested
            "strategies": [
                {
                    "strategy": strategy_desc,
                    "selected": self.last_validated_results["sequence"].tolist(),
                    "validated_fitness": self.last_validated_results["fitness"].tolist(),
                    "predicted_fitness": [0.0] * len(self.last_validated_results),  # No predictions in standalone mode
                    "predicted_std": [0.0] * len(self.last_validated_results),  # No predictions in standalone mode
                    "mean_fitness": cycle_mean_fitness,
                    "max_fitness": cycle_max_fitness,
                    "oracle_rmse": float('nan')
                }
            ]
        }
        
        # Add to campaign history
        self.campaign_history.append(cycle_entry)
        
        # Save complete campaign history in expected format
        with open(self.results_dir / "campaign_history.json", 'w') as f:
            json.dump(self.campaign_history, f, indent=2)
        
        # Also save individual cycle for compatibility
        with open(self.results_dir / f"cycle_{self.cycle}_summary.json", 'w') as f:
            json.dump(cycle_entry, f, indent=2)
        
        print(f"Saved results to {self.results_dir}")


def main():
    """Main function to run the AL_LLM agent."""
    parser = argparse.ArgumentParser(description="AL_LLM Agent for Protein Active Learning")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of sequences per cycle")
    parser.add_argument("--total_budget", type=int, required=True, help="Total experimental budget")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--protein", type=str, required=True, choices=["GB1", "TrpB"], help="Target protein")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="results/standalone_DPO/", help="Output directory")
    parser.add_argument("--init_data", type=str, default=None, help="Initial data file")
    parser.add_argument("--game", action="store_true", help="Run game campaign")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.total_budget <= args.batch_size:
        raise ValueError("total_budget must be greater than batch_size")
    if args.init_data is not None:
        args.init_data = torch.load(args.init_data)
    print(f"Initializing AL_LLM agent...")
    print(f"Protein: {args.protein}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total budget: {args.total_budget}")
    print(f"Random seed: {args.random_seed}")
    print(f"Model: {args.model}")
    
    # Initialize and run agent
    agent = sft_run(
        protein=args.protein,
        batch_size=args.batch_size,
        total_budget=args.total_budget,
        random_seed=args.random_seed,
        model=args.model,
        output_dir=args.output_dir,
        init_data=args.init_data
    )
    
    start_time = time.time()
    if args.game:
        agent.run_game_campaign()
    else:
        agent.run_campaign()
    end_time = time.time()
    
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 