#!/usr/bin/env python3
"""
Compute convergence statistics for ALDE results by analyzing increasingly large subsets of data files.

This script processes tensor files in the same way as tabulate_results.py but computes
mean and standard deviation at the final timestep for incrementally increasing numbers
of data files to analyze convergence behavior.
"""

import argparse
import numpy as np
import torch
import glob
import os
import math
import pandas as pd
from pathlib import Path
import sys
import re

def index2regret(indices, y):
    """
    Converts list of queried indices to regret (difference between the max value 
    in the design space and the max queried value).
    
    Args:
        indices: Array of queried indices
        y: Array of fitness values
    
    Returns:
        regret: Tensor of regret values
    """
    indices = np.array(indices, dtype=int)
    regret = torch.zeros((indices.shape[0], indices.shape[1]))
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            regret[i, j] = 1 - y[indices[i, :j+1]].max()
    return regret


def compute_recall_accuracy(indices, y, threshold):
    """
    Compute recall and accuracy for given indices and threshold.
    
    Args:
        indices: Array of queried indices for a single run
        y: Array of fitness values (normalized)
        threshold: Fitness threshold for positive class
    
    Returns:
        recall_values: Array of recall values at each timestep
        accuracy_values: Array of accuracy values at each timestep
    """
    indices = np.array(indices, dtype=int)
    total_positives = np.sum(y >= threshold)
    
    if total_positives == 0:
        # If no positives exist, recall is undefined, set to 0
        recall_values = np.zeros(indices.shape[0])
        accuracy_values = np.zeros(indices.shape[0])
        return recall_values, accuracy_values
    
    recall_values = np.zeros(indices.shape[0])
    accuracy_values = np.zeros(indices.shape[0])
    
    for j in range(indices.shape[0]):
        # Get cumulative selected indices up to timestep j
        selected_indices = indices[:j+1]
        
        # True positives: selected indices that are above threshold
        true_positives = np.sum(y[selected_indices] >= threshold)
        
        # False positives: selected indices that are below threshold
        false_positives = np.sum(y[selected_indices] < threshold)
        
        # Recall = TP / (TP + FN) = TP / total_positives
        recall_values[j] = true_positives / total_positives
        
        # Accuracy = TP / (TP + FP) = TP / total_selected
        total_selected = j + 1
        accuracy_values[j] = true_positives / total_selected if total_selected > 0 else 0
    
    return recall_values, accuracy_values


def tabulate_convergence(df, tests, max_files, subdir):
    """
    Tabulates convergence statistics into an organized dataframe.
    Similar to tabulate_regret but for convergence analysis.
    
    Args:
        df: DataFrame to append results to
        tests: Dictionary of test results
        max_files: Maximum number of files processed
        subdir: Directory path for extracting metadata
    
    Returns:
        df: Updated DataFrame with new results
    """
    for name in sorted(tests.keys()):
        result_tuple = tests[name]
        mean, sem = result_tuple[0], result_tuple[1]
        mean = 1 - mean  # Convert regret back to fitness
        
        # Extract recall values if available
        if len(result_tuple) >= 6:
            recall_2pc_mean = result_tuple[2]
            recall_2pc_std = result_tuple[3]
            recall_05pc_mean = result_tuple[4]
            recall_05pc_std = result_tuple[5]
        else:
            recall_2pc_mean = 0.0
            recall_2pc_std = 0.0
            recall_05pc_mean = 0.0
            recall_05pc_std = 0.0
        
        protein = Path(subdir).parent.name
        encoding = Path(subdir).name

        if 'Random' in name:
            encoding_clean = 'Random'
            model = 'Random'
            acquisition = 'Random'
        else:
            names = name.split('-')
            model = names[0]
            acquisition = names[-2]

        df.loc[len(df.index)] = [protein, encoding, model, acquisition, max_files, mean.item(), sem.item(),
                                recall_2pc_mean, recall_2pc_std, recall_05pc_mean, recall_05pc_std]
    
    return df


def compute_convergence(subdir, data_dir, verbose=False, max_seed_index=np.inf):
    """
    Compute convergence statistics for increasingly large subsets of data files.
    
    Args:
        subdir: Directory containing tensor files
        data_dir: Base directory containing fitness data
        verbose: Whether to print detailed information
        max_seed_index: Maximum seed index to consider
    
    Returns:
        convergence_df: DataFrame with convergence statistics
    """
    # First, find and group tensor files by acquisition function
    pattern = re.compile(r'(\d+)indices\.pt$')
    all_tensors = []
    for file in glob.glob(os.path.join(subdir,f'*indices.pt')):
        match = pattern.search(os.path.basename(file))
        if match and int(match.group(1)) <= max_seed_index:
            all_tensors.append(file)
    
    all_tensors = sorted(all_tensors, key=lambda x: int(re.search(r'_(\d+)indices\.pt', x).group(1)))

    
    if len(all_tensors) == 0:
        if verbose:
            print(f"No tensor files found in {subdir}")
        return pd.DataFrame()
    
    # Group files by model and acquisition function
    file_groups = {}
    for tensor_file in all_tensors:
        filename = os.path.basename(tensor_file)
        
        # Extract model and acquisition info for grouping
        if "Random" not in filename:
            # For non-random files, extract model and acquisition
            # Example: DNN_ENSEMBLE-DO-0-RBF-AGENT-[30, 1]_1indices.pt
            # Split by '_' and take everything before the seed number
            parts = filename.split('_')
            if len(parts) >= 2 and parts[-1].startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                base_part = '_'.join(parts[:-1])  # Everything except the last part (seed_indices.pt)
            else:
                base_part = parts[0]
            
            # Extract model (first part before first dash)
            model_parts = base_part.split('-')
            model = model_parts[0].replace('_', ' ')  # Convert DNN_ENSEMBLE to DNN ENSEMBLE
            
            # Look for acquisition function patterns in the entire base_part
            # Order matters - match longest patterns first
            acquisition = 'UNKNOWN'
            acquisition_patterns = ['SIMPLEAGENT', 'GREEDY', 'AGENT', 'TS', 'UCB']
            for pattern in acquisition_patterns:
                if pattern in base_part:
                    acquisition = pattern
                    break
            
            group_key = f"{model}_{acquisition}"
        else:
            # For random files
            model = 'Random'
            acquisition = 'Random'
            group_key = 'Random_Random'
        
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(tensor_file)
    
    if verbose:
        print(f"Found {len(all_tensors)} tensor files grouped into {len(file_groups)} acquisition groups")
        for group_key, files in file_groups.items():
            print(f"  {group_key}: {len(files)} files")
    
    # Initialize results dataframe
    df = pd.DataFrame(columns=['Protein', 'Encoding', 'Model', 'Acquisition', 'n_samples', 'Mean', 'Std',
                              'recall2pc', 'recall2pc_std', 'recall05pc', 'recall05pc_std'])
    
    # Process each group separately
    for group_key, group_files in file_groups.items():
        if verbose:
            print(f"\nProcessing group {group_key} with {len(group_files)} files")
        
        # Process increasingly large subsets within this group
        for n_files in range(1, len(group_files) + 1):
            if verbose:
                print(f"  Processing subset with {n_files} files...")
            
            # Create temporary directory structure for this subset
            subset_files = group_files[:n_files]
            
            # Process this subset using the existing logic
            batch, budget, actual_files = load_tensors_subset_from_files(
                subset_files, subdir, data_dir, verbose=verbose
            )
            
            if budget == 0 or 'indices' not in batch:
                if verbose:
                    print(f"    No valid data found for {n_files} files")
                continue
            
            # Add results to dataframe
            df = tabulate_convergence(df, batch['indices'], n_files, str(subdir))
    
    return df


def load_tensors_subset_from_files(tensor_files, subdir, data_dir, verbose=False):
    """
    Loads AL indices from a specific list of tensor files and converts them to regret values.
    
    Args:
        tensor_files: List of tensor file paths to process
        subdir: Directory containing tensor files (for protein name extraction)
        data_dir: Base directory containing fitness data
        verbose: Whether to print detailed information
    
    Returns:
        batch: Dictionary containing processed results
        budget: Number of queries processed
        actual_files: Number of files actually processed
    """
    actual_files = len(tensor_files)

    tests = {}
    if verbose:
        print(f'    Processing {actual_files} files')

    # Load fitness data first
    protein_name = Path(subdir).parent.name
    fitness_file = Path(data_dir) / protein_name / 'fitness.csv'
    
    if not fitness_file.exists():
        if verbose:
            print(f"    Warning: fitness file not found at {fitness_file}")
        return {}, 0, actual_files
    
    fitness_df = pd.read_csv(fitness_file)
    y = fitness_df['fitness'].values
    y_normalized = y / y.max()
    
    # Compute thresholds for top 2% and 0.5%
    threshold_2pc = np.percentile(y_normalized, 98)  # Top 2%
    threshold_05pc = np.percentile(y_normalized, 99.5)  # Top 0.5%

    for tensor in tensor_files:
        if '.pt' in tensor and 'state_dict' not in tensor:
            if "Random" not in os.path.basename(tensor):
                nm = os.path.basename(tensor).split('_')[0] + '_' + os.path.basename(tensor).split('_')[1]
            else:     
                nm = os.path.basename(tensor).split('_')[0]

            try:
                t = torch.load(tensor, map_location='cpu').detach()
                original_indices = t.clone()  # Keep original indices for recall computation
                t = torch.reshape(t, (1, -1))
                original_indices = torch.reshape(original_indices, (1, -1))
            except Exception as e:
                if verbose:
                    print(f"    Error loading tensor file {tensor}: {e}")
                continue
                
            if nm in tests.keys():
                d = tests[nm]
            else:
                d = {}
                tests[nm] = d

            dtype = os.path.basename(tensor).split('_')[-1].split('.')[0]
            dtype = ''.join([i for i in dtype if not i.isdigit()])
            if dtype in d.keys():
                tensor_list = d[dtype]
                original_indices_list = d[dtype + '_original']
            else:
                tensor_list = []
                original_indices_list = []
                d[dtype] = tensor_list
                d[dtype + '_original'] = original_indices_list
            
            # Store regret tensors and original indices
            regret_tensor = index2regret(original_indices, y_normalized)
            tensor_list.append(regret_tensor)
            original_indices_list.append(original_indices)
            
            d[dtype] = tensor_list
            d[dtype + '_original'] = original_indices_list
            tests[nm] = d

    batch = {}
    budget, total = math.inf, math.inf
    
    for key in tests.keys():
        if verbose:
            print(f"    {key}")
        num_runs = -1
        for dtype in tests[key].keys():
            if '_original' in dtype:
                continue  # Skip original indices in main processing
                
            tensor_list = tests[key][dtype]
            original_indices_list = tests[key][dtype + '_original']
            
            # Skip if no tensors were loaded
            if not tensor_list:
                continue
            
            # Find maximum length across all tensors
            tensor_lengths = [t.size(-1) for t in tensor_list]
            if not tensor_lengths or all(length == 0 for length in tensor_lengths):
                continue
            max_length = max(tensor_lengths)
            
            if 'indices' == dtype and max_length != 0:
                if max_length < budget:
                    budget = max_length
                num_runs = sum(t.size(0) for t in tensor_list)
            elif 'y' in dtype and max_length < total and max_length != 0:
                total = max_length
            
            # Compute statistics only at the final timestep
            final_timestep = max_length - 1
            
            # Collect values from all tensors at the final timestep
            values_at_final_timestep = []
            recall_2pc_at_final_timestep = []
            recall_05pc_at_final_timestep = []
            
            for tensor_idx, tensor in enumerate(tensor_list):
                if final_timestep < tensor.size(-1):
                    values_at_final_timestep.append(tensor[:, final_timestep])
                    
                    # Compute recall for this run at final timestep
                    original_indices = original_indices_list[tensor_idx]
                    for run_idx in range(original_indices.size(0)):
                        indices_up_to_final = original_indices[run_idx, :final_timestep+1].numpy()
                        
                        recall_2pc, _ = compute_recall_accuracy(indices_up_to_final, y_normalized, threshold_2pc)
                        recall_05pc, _ = compute_recall_accuracy(indices_up_to_final, y_normalized, threshold_05pc)
                        
                        recall_2pc_at_final_timestep.append(recall_2pc[-1])  # Get value at final timestep
                        recall_05pc_at_final_timestep.append(recall_05pc[-1])
            
            if values_at_final_timestep:
                # Concatenate all available values at final timestep
                combined_values = torch.cat(values_at_final_timestep, dim=0)
                mean_val = torch.mean(combined_values)
                
                if len(combined_values) > 1:
                    std_val = torch.std(combined_values)
                    sem_val = std_val / (len(combined_values) ** 0.5)
                else:
                    # If only one value, std is 0
                    sem_val = torch.tensor(0.0)
                
                # Compute recall statistics
                recall_2pc_mean = np.mean(recall_2pc_at_final_timestep) if recall_2pc_at_final_timestep else 0.0
                recall_2pc_std = np.std(recall_2pc_at_final_timestep) if len(recall_2pc_at_final_timestep) > 1 else 0.0
                recall_05pc_mean = np.mean(recall_05pc_at_final_timestep) if recall_05pc_at_final_timestep else 0.0
                recall_05pc_std = np.std(recall_05pc_at_final_timestep) if len(recall_05pc_at_final_timestep) > 1 else 0.0
                
                if dtype in batch.keys():
                    d = batch[dtype]
                else:
                    d = {}
                    batch[dtype] = d

                d[key] = (mean_val, sem_val, recall_2pc_mean, recall_2pc_std, recall_05pc_mean, recall_05pc_std)
                batch[dtype] = d
            
        if verbose:
            print(f"    Runs: {num_runs}")
            
    if verbose:
        print(f'    Budget: {budget}')
        print(f'    Total queries (incl. init): {budget}')

    return batch, budget, actual_files


def clean_dataframe(df):
    """
    Clean up the results dataframe with proper naming conventions.
    
    Args:
        df: Raw results DataFrame
    
    Returns:
        df: Cleaned DataFrame
    """
    df = df.drop_duplicates(subset=['Protein', 'Encoding', 'Model', 'Acquisition', 'n_samples'], keep='first')
    df['Model'] = df['Model'].replace('BOOSTING_ENSEMBLE', 'Boosting Ensemble')
    df['Model'] = df['Model'].replace('GP_BOTORCH', 'GP')
    df['Model'] = df['Model'].replace('DNN_ENSEMBLE', 'DNN Ensemble')
    df['Model'] = df['Model'].replace('DKL_BOTORCH', 'DKL')
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute convergence statistics for ALDE results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--proteins', 
        nargs='+', 
        default=['GB1', 'TrpB'],
        help='List of proteins to process'
    )
    
    parser.add_argument(
        '--encodings', 
        nargs='+', 
        default=['AA', 'georgiev', 'onehot', 'ESM2'],
        help='List of encodings to process'
    )
    
    parser.add_argument(
        '--results-dir', 
        default='../results/5x96_simulations',
        help='Base directory containing results'
    )
    
    parser.add_argument(
        '--output-path', 
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Print detailed processing information'
    )
    
    parser.add_argument(
        '--data-dir',
        default='../data',
        help='Base directory containing fitness data'
    )
    
    parser.add_argument(
        '--max-seed-index',
        type=int,
        default=np.inf,
        help='Maximum seed index to consider'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Initialize results dataframe
    df = pd.DataFrame(columns=['Protein', 'Encoding', 'Model', 'Acquisition', 'n_samples', 'Mean', 'Std',
                              'recall2pc', 'recall2pc_std', 'recall05pc', 'recall05pc_std'])
    
    # Process each protein-encoding combination
    for protein in args.proteins:
        for encoding in args.encodings:
            subdir = results_dir / protein / encoding
            if not subdir.exists():
                if args.verbose:
                    print(f"Warning: Directory {subdir} does not exist, skipping")
                continue
            
            if args.verbose:
                print(f"\nProcessing {protein}/{encoding}")
            
            # Compute convergence statistics
            convergence_df = compute_convergence(
                str(subdir), 
                args.data_dir, 
                args.verbose, 
                args.max_seed_index
            )
            
            if convergence_df.empty:
                if args.verbose:
                    print(f"No convergence data found in {subdir}")
                continue
            
            # Append to main dataframe
            df = pd.concat([df, convergence_df], ignore_index=True)
    
    if df.empty:
        print("No convergence data was computed. Check your input directories and file patterns.")
        sys.exit(1)
    
    # Clean up results
    df = clean_dataframe(df)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(df)} convergence points across {len(df['Protein'].unique())} proteins and {len(df['Encoding'].unique())} encodings")
    
    if args.verbose:
        print(f"\nSummary:")
        print(f"Proteins: {sorted(df['Protein'].unique())}")
        print(f"Encodings: {sorted(df['Encoding'].unique())}")
        print(f"Models: {sorted(df['Model'].unique())}")
        print(f"Acquisitions: {sorted(df['Acquisition'].unique())}")
        print(f"Sample sizes: {sorted(df['n_samples'].unique())}")


if __name__ == "__main__":
    main() 