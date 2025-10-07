"""
Evaluates all active learning campaigns in a folder by recall 0.5 % metric, extracts the top 10 % of the best performing campaigns. 
From these, it samples N steps from each campaign and creates two kinds of prompts for each of these steps.
Fitness and sequence of each selection is extracted from the fitness.csv file provided as arg.
"""

import os
import glob
import json
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
import itertools

class LLM:
    """Simple wrapper for LLM inference."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split(text)[-1].strip()

class ParallelLLM:
    """Multi-GPU parallel LLM inference wrapper."""
    def __init__(self, model_name, num_gpus=4):
        self.model_name = model_name
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.models = []
        self.tokenizers = []
        
        print(f"Initializing {self.num_gpus} model instances across GPUs...")
        
        # Load models on different GPUs
        for gpu_id in range(self.num_gpus):
            print(f"Loading model on GPU {gpu_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map=f"cuda:{gpu_id}"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        
        print(f"✓ Loaded {len(self.models)} model instances")
        
        # Thread-local storage for GPU assignment
        self.local = threading.local()
        self.gpu_queue = Queue()
        for i in range(self.num_gpus):
            self.gpu_queue.put(i)
    
    def _get_gpu_id(self):
        """Get an available GPU ID for the current thread."""
        if not hasattr(self.local, 'gpu_id'):
            self.local.gpu_id = self.gpu_queue.get()
        return self.local.gpu_id
    
    def _release_gpu_id(self, gpu_id):
        """Release GPU ID back to the pool."""
        self.gpu_queue.put(gpu_id)
    
    def _generate_single(self, messages):
        """Generate response using an available GPU."""
        gpu_id = self._get_gpu_id()
        try:
            model = self.models[gpu_id]
            tokenizer = self.tokenizers[gpu_id]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu_id}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.split(text)[-1].strip()
        finally:
            # Reset thread-local GPU assignment for next request
            if hasattr(self.local, 'gpu_id'):
                delattr(self.local, 'gpu_id')
    
    def generate(self, messages):
        """Single generation (for backward compatibility)."""
        return self._generate_single(messages)
    
    def generate_parallel(self, messages_list):
        """Generate responses for multiple prompts in parallel."""
        results = [None] * len(messages_list)
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._generate_single, messages): i 
                for i, messages in enumerate(messages_list)
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_index), total=len(messages_list), desc="Parallel generation"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    print(result)
                    results[index] = result
                except Exception as exc:
                    print(f'Generation {index} generated an exception: {exc}')
                    results[index] = ""
        
        return results

def validate_args(args):
    """Validate command line arguments."""
    print("Validating arguments...")
    # Check output directory exists or create it
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Validate folder path pattern
    folder_exists = False
    for i in range(args.n_folders):
        folder = f"{args.folder_path}_{i}"
        if os.path.exists(folder):
            folder_exists = True
            break
    
    if not folder_exists:
        raise FileNotFoundError(f"No folders found matching pattern: {args.folder_path}_{{0..{args.n_folders-1}}}")
    
    # Validate numeric arguments
    if args.n_folders <= 0:
        raise ValueError("Number of folders must be positive")
    if args.n_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if args.batch_samples <= 0:
        raise ValueError("Number of batch samples must be positive")
    
    print("✓ All arguments validated successfully")

def get_recall05_cutoff(fitness_df):
    """Calculate the 99.5th percentile fitness cutoff."""
    return fitness_df['fitness'].quantile(0.995)

def get_top_10(filenames, fitness_df):
    """Get top 10% performing campaigns based on recall metric.
    Computes number of selected sequences above the 99.5th percentile of the fitness scores.
    Sorts campaigns by the number of selected sequences above the cutoff and returns the top 10%."""
    print(f"Evaluating {len(filenames)} campaigns...")
    cutoff = get_recall05_cutoff(fitness_df)
    print(f"Recall 0.5% cutoff: {cutoff:.6f}")

    recall05_dict = {}
    valid_filenames = []
    
    for filename in tqdm(filenames, desc="Calculating recall metrics"):
        try:
            if not os.path.exists(filename):
                print(f"Warning: File not found: {filename}")
                continue
                
            indices = torch.load(filename)
            if len(indices) == 0:
                print(f"Warning: Empty indices in {filename}")
                continue
            
            x = range(len(indices))
            y = (fitness_df.iloc[indices]['fitness'] > cutoff).cumsum().tolist()
            plt.plot(x, y, c='red', alpha=0.2)
            recall = (fitness_df.iloc[indices]['fitness'] > cutoff).sum()
            recall05_dict[filename] = recall
            valid_filenames.append(filename)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not recall05_dict:
        raise ValueError("No valid campaign files found")

    best_filenames = sorted(recall05_dict, key=recall05_dict.get, reverse=True)[:max(1, len(valid_filenames)//10)]
    print(f"Selected top {len(best_filenames)} campaigns (top 10%)")
    
    best_df = []
    for filename in best_filenames:
        try:
            indices = torch.load(filename)
            df = fitness_df.iloc[indices].copy()
            best_df.append(df)
            x = range(len(indices))
            y = (fitness_df.iloc[indices]['fitness'] > cutoff).cumsum().tolist()
            plt.plot(x, y, c='blue', alpha=0.4)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # Get remaining dataframes (not in best_filenames)
    remaining_filenames = [f for f in valid_filenames if f not in best_filenames]
    remaining_df = []
    for filename in remaining_filenames:
        try:
            indices = torch.load(filename)
            df = fitness_df.iloc[indices].copy()
            remaining_df.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    plt.savefig(Path(filenames[0]).parent / "recall_curve.png")
    plt.close()
    return best_df, remaining_df

prompt_format_single = """You are an expert bioinformatician conducting active learning to maximize the fitness of a 4-residue long protein motif (20^4=160000 possible sequences) using a budget of {budget} more sequences.

TASK: Given the current data, generate ONE novel 4-residue sequence that maximises information gain.

CURRENT DATA (shuffled):
{protein_sequences}

Available amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
Answer only with the next sequence:
"""

prompt_format_batch = """You are an expert bioinformatician conducting active learning to maximize the fitness of a 4-residue long protein motif (20^4=160000 possible sequences) using a budget of {budget} more sequences.

TASK: Given the current data, generate {batch_samples} novel 4-residue sequences that maximises information gain.

CURRENT DATA (shuffled):
{protein_sequences}

Available amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

Answer only with the next {batch_samples} sequences, comma separated, no other text:
"""

class PromptGenerator:
    """Handles generation of both preference and conversational format prompts."""
    
    def __init__(self, llm_model=None):
        self.llm = llm_model
    
    def _gen_game_prompt(self, df, step, batch_samples=10, perm = None):
        """Generate a game prompt at a specific step."""
        if perm is None:
            perm = random.sample(range(4), 4)
        def permute_combo(combo, perm):
            parts = combo.split()
            return ' '.join([parts[i] for i in perm])
        reordered_df = df.copy()
        reordered_df['Combo'] = reordered_df['Combo'].apply(lambda x: permute_combo(x, perm))
        reordered_df['fitness'] = reordered_df['fitness'].apply(lambda x: x+np.random.randn()*0.005)
        reordered_df.iloc[:(step-1)*batch_samples, :] = reordered_df.iloc[:(step-1)*batch_samples, :].sample(frac=1)
        chunks = [reordered_df.iloc[i*batch_samples:(i+1)*batch_samples] for i in range(0, step)]
        total_budget = batch_samples * (step + np.random.randint(1,10))
        messages = [{"role": "system", "content": f"You are an expert bioinformatician."}]
        initial_prompt = f"""You are conducting active learning to find the highest fitness 4-residue protein sequences from 160,000 possibilities (20^4). 

GOAL: Maximize information gain to discover sequences with the highest possible fitness values.

PROCESS: 
1. Generate {batch_samples} sequences
2. Receive measured fitness results for each sequence
3. Use results to inform next batch generation
4. Repeat for {total_budget//batch_samples - 1} remaining cycles

TASK: Generate {batch_samples} novel 4-residue sequences that will provide maximum information about high-fitness regions.

CURRENT DATA:
{chunks[0].round(3).to_string(index=False)}

Amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

Output only the {batch_samples} sequences, comma-separated, no other text:"""
        messages.append({"role": "user", "content": initial_prompt})
        for i, chunk in enumerate(chunks[1:-1]):
            reply = ','.join(chunk['Combo'].tolist())
            messages.append({"role": "assistant", "content": reply})
            feedback_lines = [f"{seq} {fitness:.3f}" for seq, fitness in zip(chunk['Combo'], chunk['fitness'])]
            total_budget = total_budget - batch_samples
            messages.append({"role": "user", "content": f"""CYCLE {i+1} RESULTS:
{chr(10).join(feedback_lines)}

Generate the next {batch_samples} sequences for cycle {i + 2}:"""})
        # Compute the next batch start based on the number of completed cycles
        reply = ','.join(chunks[-1]['Combo'].tolist())
        messages.append({"role": "assistant", "content": reply})
        return {"messages": messages}

    def _gen_prompt(self, df, prompt_format, step, top_25=False, batch_samples=10):
        """Generate a single prompt at a specific step."""
        perm = random.sample(range(4), 4)
        def permute_combo(combo, perm):
            parts = combo.split()
            return ' '.join([parts[i] for i in perm])
        reordered_df = df.copy()
        reordered_df['Combo'] = reordered_df['Combo'].apply(lambda x: permute_combo(x, perm))
        try:
            if top_25:
                data = reordered_df.iloc[:step].sort_values(by='fitness', ascending=False).head(20).round(3).sample(frac=1)
                truth = reordered_df.iloc[step]['Combo']
                rejected = reordered_df.iloc[:step].sort_values(by='fitness', ascending=False).head(5).sample(n=1)['Combo'].iloc[0]
            else:
                data = reordered_df.iloc[:step].round(3).sample(frac=1)
                truth = ','.join(reordered_df.iloc[step:step+batch_samples]['Combo'].tolist())
                rejected = ""
            if len(data) == 0:
                raise ValueError(f"No data available for step {step}")
            prompt = prompt_format.format(protein_sequences=data.to_string(index=False), budget=random.randint(1, 400-step), batch_samples=batch_samples)
            
            return prompt, truth, rejected
        except Exception as e:
            raise ValueError(f"Error generating prompt for step {step}: {e}")

    def _gen_thinking_section(self, prompt, truth):
        """Generate detailed reasoning for the selection."""
        if not self.llm:
            return "Thinking not available without LLM model."
            
        instruction = f"""/nothink Generate a detailed step-by-step reasoning process that leads to the correct answer for this protein fitness optimization task.

TASK:
{prompt}

CORRECT ANSWER:
{truth}

REASONING REQUIREMENTS:
1. **Amino Acid Substitution Analysis**: Evaluate the biochemical consequences of specific residue changes (charge alterations, hydrophobicity shifts, size differences, hydrogen bonding capacity)
2. **Pairwise Interaction Effects**: Assess how amino acid substitutions affect local interactions with neighboring residues (salt bridges, hydrophobic contacts, steric clashes, backbone flexibility)
3. **Epistatic Relationships**: Identify compensatory mutations, synergistic effects, and how multiple substitutions interact to influence overall fitness
4. **Active Learning Strategy**: Balance exploration of sequence space with exploitation of high-fitness regions using substitution effect priors
5. **Statistical Modeling**: Discuss uncertainty quantification and information gain from proposed mutations in the context of amino acid interaction networks

OUTPUT FORMAT:
Step 1: [Individual amino acid substitution impact analysis]
Step 2: [Pairwise residue interaction assessment]
Step 3: [Epistatic effect identification and modeling]
Step 4: [Active learning strategy with substitution-based priors]
Step 5: [Final selection rationale based on amino acid interaction principles]

Generate only the reasoning steps above, no additional text. Be brief and straight to the point.
"""
        thinking_result = self.llm.generate([{"role": "user", "content": instruction}])
        print(thinking_result)
        return thinking_result
    
    def _gen_thinking_sections_parallel(self, prompts_and_truths):
        """Generate thinking sections in parallel for multiple prompts."""
        if not self.llm or not hasattr(self.llm, 'generate_parallel'):
            # Fallback to sequential generation
            return [self._gen_thinking_section(prompt, truth) for prompt, truth in prompts_and_truths]
        
        # Prepare all instruction messages
        messages_list = []
        for prompt, truth in prompts_and_truths:
            instruction = f"""/nothink Generate a detailed step-by-step reasoning process that leads to the correct answer for this protein fitness optimization task.
TASK:
{prompt}

CORRECT ANSWER:
{truth}

REASONING REQUIREMENTS:
1. **Sequence Analysis**: Examine patterns in the provided sequences (length, composition, conserved regions, mutations)
2. **Fitness Correlation**: Identify relationships between sequence features and fitness scores
3. **Active Learning Strategy**: Explain the exploration vs exploitation trade-offs in your selection
4. **Biochemical Rationale**: Apply protein structure-function principles (hydrophobicity, charge, secondary structure, etc.)
5. **Statistical Considerations**: Discuss uncertainty, information gain, and sampling strategy
6. **Selection Logic**: Connect observations to why this specific sequence maximizes expected utility

OUTPUT FORMAT:
Step 1: [Sequence pattern analysis]
Step 2: [Fitness landscape assessment] 
Step 3: [Active learning strategy identification]
Step 4: [Biochemical evaluation]
Step 5: [Statistical justification]
Step 6: [Final selection rationale]

Generate only the reasoning steps above, no additional text. Be brief and straight to the point:
"""
            messages_list.append([{"role": "user", "content": instruction}])
        
        # Generate all thinking sections in parallel
        return self.llm.generate_parallel(messages_list)

    def _generate_steps(self, df_len, n_samples, batch_samples):
        """Generate sampling steps for prompt generation."""
        max_step = min(df_len - 1, 300)
        if max_step < 1:
            return [], []
            
        upsample_start = np.array([10,12,14,18,22,28,35])
        steps = np.random.choice(range(1, max_step), min(n_samples-len(upsample_start), max_step-1), replace=False)
        steps = np.concatenate([steps, upsample_start])
        steps_long = np.random.choice(range(batch_samples, min(max_step, 400-batch_samples), batch_samples), 
                                     min(n_samples-len(upsample_start), len(range(batch_samples, 400-batch_samples, batch_samples))), 
                                     replace=False)
        steps_long = np.concatenate([steps_long, upsample_start])
        return steps, steps_long

    def generate_prompts(self, df, fitness_df, n_samples=25, batch_samples=10, 
                        thinking=False, format_type="preference"):
        """Generate prompts in either preference or conversational format."""
        
        results = {
            'short': [],
            'long': [],
            'thinking': [],
        }
        
        if len(df) < 2:
            print(f"Warning: Dataframe too small ({len(df)} sequences), skipping...")
            return results
        
        steps, steps_long = self._generate_steps(len(df), n_samples, batch_samples)
        
        # Generate single sequence prompts (short prompts)
        for step in tqdm(steps, desc=f"Generating {format_type} single prompts", leave=False):
            try:
                prompt, chosen, rejected = self._gen_prompt(df, prompt_format_single, step, top_25=True)
                rejected_rand = fitness_df.sample(n=1)['Combo'].iloc[0]
                
                if format_type == "preference":
                    results['short'].extend([
                        self._format_preference(prompt, chosen, rejected)
                    ])
                    if step > 40:
                        results['short'].extend([
                            self._format_preference(prompt, chosen, rejected_rand)
                        ])
                else:  # conversational
                    results['short'].append(self._format_conversational(prompt, chosen))

            except Exception as e:
                print(f"Error generating prompt for step {step}: {e}")
                continue
        
        # Generate batch sequence prompts (long prompts) - always use batch format
        for step in tqdm(steps_long, desc=f"Generating {format_type} batch prompts", leave=False):
            try:
                long_prompt, chosen, rejected = self._gen_prompt(df, prompt_format_batch, step, top_25=False, batch_samples=batch_samples)
                rejected_rand = ','.join(fitness_df.sample(n=batch_samples)['Combo'].tolist())
                
                if format_type == "preference":
                    results['long'].extend([
                        self._format_preference(long_prompt, chosen, rejected_rand)
                    ])
                else:  # conversational
                    results['long'].append(self._format_conversational(long_prompt, chosen))

            except Exception as e:
                print(f"Error generating prompt for step {step}: {e}")
                continue
        
        # Generate game prompts
        if not format_type == "preference":
            results['game'] = []
            vals = np.arange(1,7)
            w = 1/vals
            w = w/sum(w)
            vals = vals + 1
            # get a list of lists with all permutations of 4 numbers from 0 to 3
            perms = list(itertools.permutations(range(4)))
            steps= np.random.choice(vals, p=w, size=len(perms))
            for i in range(len(perms)):
                messages= self._gen_game_prompt(df, int(steps[i]), batch_samples, perm = perms[i])
                results['game'].append(messages)
        
        # Generate thinking prompts if requested (using batch format)
        if thinking and self.llm:
            # First, collect all prompts that need thinking generation
            thinking_data = []
            for step in steps_long:
                try:
                    if step + 10 >= len(df):
                        continue
                        
                    prompt, _, _ = self._gen_prompt(df, prompt_format_batch, step, top_25=True)
                    chosen_seqs = df.iloc[step:step+10]['Combo'].tolist()
                    chosen_str = ",".join(chosen_seqs)
                    
                    thinking_data.append({
                        'step': step,
                        'prompt': prompt,
                        'chosen_str': chosen_str,
                        'chosen_seqs': chosen_seqs
                    })
                    
                except Exception as e:
                    print(f"Error preparing thinking prompt for step {step}: {e}")
                    continue
            
            if thinking_data:
                print(f"Generating thinking sections for {len(thinking_data)} prompts in parallel...")
                
                # Prepare prompts and truths for parallel generation
                prompts_and_truths = [(item['prompt'], item['chosen_str']) for item in thinking_data]
                
                # Generate all chosen thinking sections in parallel
                chosen_thinking_results = self._gen_thinking_sections_parallel(prompts_and_truths)
                
                # For preference format, also generate rejected thinking sections
                if format_type == "preference":
                    rejected_prompts_and_truths = []
                    for item in thinking_data:
                        rejected_seqs = fitness_df.sample(n=10)['Combo'].tolist()
                        rejected_str = ",".join(rejected_seqs)
                        rejected_prompts_and_truths.append((item['prompt'], rejected_str))
                        item['rejected_str'] = rejected_str
                    
                    rejected_thinking_results = self._gen_thinking_sections_parallel(rejected_prompts_and_truths)
                
                # Process results
                for i, item in enumerate(thinking_data):
                    try:
                        chosen_thinking = chosen_thinking_results[i]
                        chosen = f"<think>\n{chosen_thinking}\n</think>\n{item['chosen_str']}"
                        
                        if format_type == "preference":
                            rejected_thinking = rejected_thinking_results[i]
                            rejected = f"<think>\n{rejected_thinking}\n</think>\n{item['rejected_str']}"
                            results['thinking'].append(self._format_preference(item['prompt'], chosen, rejected))
                        else:  # conversational
                            results['thinking'].append(self._format_conversational(item['prompt'], chosen))
                            
                    except Exception as e:
                        print(f"Error processing thinking result for step {item['step']}: {e}")
                        continue
        
        return results

    def _format_preference(self, prompt, chosen, rejected):
        """Format conversation for preference training data."""
        return {
            "prompt": [{'role': 'user', 'content': prompt}], 
            'chosen': [{'role': 'assistant', 'content': chosen}], 
            'rejected': [{'role': 'assistant', 'content': rejected}]
        }

    def _format_conversational(self, prompt, completion):
        """Format conversation for standard training data."""
        return {
            "prompt": prompt,
            "completion": completion
        }

class DatasetManager:
    """Manages saving and organizing different dataset formats."""
    
    def __init__(self, output_path):
        self.output_path = output_path
        
    def save_datasets(self, preference_data, conversational_data, thinking=False, batch_samples=10):
        """Save both preference and conversational datasets."""
        print(f"\nSaving results to {self.output_path}...")
        
        # Save preference datasets
        pref_files = {
            f"ALDE_short_prompts_{batch_samples}.json": preference_data['short'],
            f"ALDE_long_prompts_{batch_samples}.json": preference_data['long'],
        }
        if thinking:
            pref_files[f"ALDE_thinking_prompts_{batch_samples}.json"] = preference_data['thinking']
        
        for filename, data in pref_files.items():
            output_path = os.path.join(self.output_path, filename)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(data)} preference prompts to {output_path}")
        
        # Save conversational datasets
        conv_files = {
            f"ALDE_short_conversational_{batch_samples}.json": self._format_conversational_output(conversational_data['short']),
            f"ALDE_long_conversational_{batch_samples}.json": self._format_conversational_output(conversational_data['long']),
            f"ALDE_game_conversational_{batch_samples}.json": conversational_data['game'],
        }
        if thinking:
            conv_files[f"ALDE_thinking_conversational_{batch_samples}.json"] = self._format_conversational_output(conversational_data['thinking'])
        
        for filename, data in conv_files.items():
            output_path = os.path.join(self.output_path, filename)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(data['prompt']) if 'prompt' in data else len(data)} conversational prompts to {output_path}")
        
        self._print_summary(preference_data, conversational_data)

    def _format_conversational_output(self, conv_data):
        """Convert list of conversational items to prompt/completion lists."""
        prompts = []
        completions = []
        for item in conv_data:
            prompts.append(item["prompt"])
            completions.append(item["completion"])
        return {"prompt": prompts, "completion": completions}

    def _print_summary(self, pref_data, conv_data):
        """Print generation summary."""
        print(f"\n{'='*50}")
        print("GENERATION SUMMARY")
        print(f"{'='*50}")
        print("PREFERENCE DATASETS:")
        print(f"  Short prompts: {len(pref_data['short'])}")
        print(f"  Long prompts: {len(pref_data['long'])}")
        print(f"  Thinking prompts: {len(pref_data['thinking'])}")
        print(f"  Total preference prompts: {len(pref_data['short']) + len(pref_data['long']) + len(pref_data['thinking'])}")
        print("\nCONVERSATIONAL DATASETS:")
        print(f"  Short prompts: {len(conv_data['short'])}")
        print(f"  Long prompts: {len(conv_data['long'])}")
        print(f"  Thinking prompts: {len(conv_data['thinking'])}")
        print(f"  Game prompts: {len(conv_data['game'])}")
        print(f"  Total conversational prompts: {len(conv_data['short']) + len(conv_data['long']) + len(conv_data['thinking']) + len(conv_data['game'])}")
        print(f"\nOutput directory: {self.output_path}")
        print("✓ Generation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Generate training prompts from ALDE campaigns")
    parser.add_argument("--fitness_folder", type=str, required=True, default="data/artificial_ESM2",
                        help="Path to fitness CSV file")
    parser.add_argument("--folder_path", type=str, required=True, default="results/artificial/artificial_ESM2",
                        help="Base path for campaign folders (will append _0, _1, etc.)")
    parser.add_argument("--n_folders", type=int, required=True,
                        help="Number of campaign folders to process")
    parser.add_argument("--output_path", type=str, required=True, default="database/prompts/",
                        help="Output directory for generated prompts")
    parser.add_argument("--n_samples", type=int, default=25,
                        help="Number of single sequence samples per campaign (default: 25)")
    parser.add_argument("--batch_samples", type=int, default=10,
                        help="Number of batch sequence samples per campaign (default: 10)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model to use for thinking generation (default: Qwen/Qwen3.5-7B-Instruct)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for the output directories")
    parser.add_argument("--thinking", action="store_true",
                        help="Generate thinking prompts")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use for parallel thinking generation (default: 4)")
    
    args = parser.parse_args()
    
    # Initialize model if needed
    llm = None
    if args.thinking:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        from queue import Queue
        print(f"Loading model for parallel inference: {args.model}")
        print(f"Using {args.num_gpus} GPUs for parallel generation")
        
        try:
            llm = ParallelLLM(args.model, num_gpus=args.num_gpus)
            print("✓ Parallel model loaded successfully")
        except Exception as e:
            print(f"Failed to load parallel model: {e}")
            print("Falling back to single GPU model...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16, local_files_only=True, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            llm = LLM(model=model, tokenizer=tokenizer)
            print("✓ Single GPU model loaded successfully")
    
    # Initialize generators and managers
    prompt_generator = PromptGenerator(llm)
    dataset_manager = DatasetManager(args.output_path)
    
    # Initialize data collections
    all_preference_data = {'short': [], 'long': [], 'thinking': []}
    all_conversational_data = {'short': [], 'long': [], 'thinking': [], 'game': []}
    
    # Process each folder
    for i in tqdm(range(args.start_index, args.start_index + args.n_folders), desc="Processing folders"):
        folder = f"{args.folder_path}_{i}"
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist, skipping...")
            continue
            
        print(f"\nProcessing folder: {folder}")
        print(f"Loading fitness data from {args.fitness_folder}...")
        fitness_df = pd.read_csv(os.path.join(f"{args.fitness_folder}_{i}/", "fitness.csv"))
        fitness_df["Combo"] = [" ".join(s) for s in fitness_df["Combo"]]
        # skew data
        def check_skew(fitness_df):
            return sum(fitness_df['fitness'] > 0.5) / len(fitness_df) > 0.02
        while check_skew(fitness_df):
            fitness_df['fitness'] = fitness_df['fitness'].apply(lambda x: x**1.5)
        print(f"Loaded {len(fitness_df)} sequences")
        plt.hist(fitness_df['fitness'], bins=100)
        plt.savefig(f"fitness_distribution.png")
        plt.close()

        # Get campaign files
        campaign_files = glob.glob(f"{folder}/onehot/*indices.pt")
        if not campaign_files:
            print(f"Warning: No campaign files found in {folder}")
            continue
        campaign_files = [c for c in campaign_files if not "Random" in c]
        print(f"Found {len(campaign_files)} campaign files")
        
        # Get top performing campaigns and others
        top_dfs, other_dfs = get_top_10(campaign_files, fitness_df)
        if not top_dfs:
            print(f"Warning: No valid campaigns found in {folder}")
            continue
        
        # Generate preference datasets from top campaigns
        for j, top_df in enumerate(top_dfs):
            pref_data = prompt_generator.generate_prompts(
                top_df, fitness_df, args.n_samples, args.batch_samples, 
                args.thinking, "preference"
            )
            
            for key in all_preference_data:
                all_preference_data[key].extend(pref_data[key])
        
        # Generate conversational datasets from remaining campaigns
        for j, other_df in enumerate(other_dfs + top_dfs):
            conv_data = prompt_generator.generate_prompts(
                other_df, fitness_df, 15, args.batch_samples, 
                args.thinking, "conversational"
            )
            
            for key in all_conversational_data:
                all_conversational_data[key].extend(conv_data[key])
    
    # Save all datasets
    dataset_manager.save_datasets(all_preference_data, all_conversational_data, args.thinking, args.batch_samples)

if __name__ == "__main__":
    main()