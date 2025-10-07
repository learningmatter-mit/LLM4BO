#!/usr/bin/env python3
"""
Tabulate ALDE results from tensor files into a consolidated CSV format.

This script processes tensor files containing active learning indices and converts
them to regret values for analysis and comparison across different models,
acquisition functions, and encodings.
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


def load_tensors(subdir, data_dir, verbose=False, max_seed_index=np.inf):
    """
    Loads AL indices from a directory and converts them to regret values.
    Also computes recall and accuracy metrics.
    
    Args:
        subdir: Directory containing tensor files
        data_dir: Base directory containing fitness data
        verbose: Whether to print detailed information
    
    Returns:
        batch: Dictionary containing processed results
        budget: Number of queries processed
    """
    pattern = re.compile(r'(\d+)indices\.pt$')
    tensors = []
    for file in glob.glob(os.path.join(subdir, '*indices.pt')):
        match = pattern.search(os.path.basename(file))
        if match and int(match.group(1)) <= int(max_seed_index):
            tensors.append(file)

    tensors = sorted(tensors)
    if not tensors:
        if verbose:
            print(f"No tensor files found in {subdir}")
        return {}, 0
        
    tests = {}
    if verbose:
        print('Models not included/not over budget yet:\n')

    # Load fitness data first
    protein_name = Path(subdir).parent.name
    fitness_file = Path(data_dir) / protein_name / 'fitness.csv'
    
    if not fitness_file.exists():
        if verbose:
            print(f"Warning: fitness file not found at {fitness_file}")
        return {}, 0
    
    fitness_df = pd.read_csv(fitness_file)
    y = fitness_df['fitness'].values
    y_normalized = y / y.max()
    
    # Compute thresholds for top 2% and 0.5%
    threshold_2pc = np.percentile(y_normalized, 98)  # Top 2%
    threshold_05pc = np.percentile(y_normalized, 99.5)  # Top 0.5%

    for tensor in tensors:
        if '.pt' in tensor and 'state_dict' not in tensor:
            first = False
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
                    print(f"Error loading tensor file {tensor}: {e}")
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

    if verbose:
        print('\nModels included:\n')
        
    batch = {}
    budget, total = math.inf, math.inf
    
    for key in tests.keys():
        if verbose:
            print(key)
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
            
            # Compute statistics with variable-length handling
            mean_values = []
            std_values = []
            
            # Initialize recall and accuracy tracking
            recall_2pc_values = []
            recall_2pc_std_values = []
            recall_05pc_values = []
            recall_05pc_std_values = []
            accuracy_2pc_values = []
            accuracy_2pc_std_values = []
            accuracy_05pc_values = []
            accuracy_05pc_std_values = []
            
            for timestep in range(max_length):
                # Collect regret values from all tensors that have data at this timestep
                values_at_timestep = []
                recall_2pc_at_timestep = []
                recall_05pc_at_timestep = []
                accuracy_2pc_at_timestep = []
                accuracy_05pc_at_timestep = []
                
                for tensor_idx, tensor in enumerate(tensor_list):
                    if timestep < tensor.size(-1):
                        values_at_timestep.append(tensor[:, timestep])
                        
                        # Compute recall and accuracy for this run
                        original_indices = original_indices_list[tensor_idx]
                        for run_idx in range(original_indices.size(0)):
                            indices_up_to_timestep = original_indices[run_idx, :timestep+1].numpy()
                            
                            recall_2pc, accuracy_2pc = compute_recall_accuracy(indices_up_to_timestep, y_normalized, threshold_2pc)
                            recall_05pc, accuracy_05pc = compute_recall_accuracy(indices_up_to_timestep, y_normalized, threshold_05pc)
                            
                            recall_2pc_at_timestep.append(recall_2pc[-1])  # Get value at current timestep
                            recall_05pc_at_timestep.append(recall_05pc[-1])
                            accuracy_2pc_at_timestep.append(accuracy_2pc[-1])
                            accuracy_05pc_at_timestep.append(accuracy_05pc[-1])
                
                if values_at_timestep:
                    # Regret statistics
                    combined_values = torch.cat(values_at_timestep, dim=0)
                    mean_val = torch.mean(combined_values)
                    
                    if len(combined_values) > 1:
                        std_val = torch.std(combined_values)
                        sem_val = std_val / (len(combined_values) ** 0.5)
                    else:
                        sem_val = torch.tensor(0.0)
                    
                    mean_values.append(mean_val)
                    std_values.append(sem_val)
                    
                    # Recall and accuracy statistics
                    recall_2pc_mean = np.mean(recall_2pc_at_timestep)
                    recall_2pc_std = np.std(recall_2pc_at_timestep) if len(recall_2pc_at_timestep) > 1 else 0.0
                    recall_05pc_mean = np.mean(recall_05pc_at_timestep)
                    recall_05pc_std = np.std(recall_05pc_at_timestep) if len(recall_05pc_at_timestep) > 1 else 0.0
                    
                    accuracy_2pc_mean = np.mean(accuracy_2pc_at_timestep)
                    accuracy_2pc_std = np.std(accuracy_2pc_at_timestep) if len(accuracy_2pc_at_timestep) > 1 else 0.0
                    accuracy_05pc_mean = np.mean(accuracy_05pc_at_timestep)
                    accuracy_05pc_std = np.std(accuracy_05pc_at_timestep) if len(accuracy_05pc_at_timestep) > 1 else 0.0
                    
                    recall_2pc_values.append(recall_2pc_mean)
                    recall_2pc_std_values.append(recall_2pc_std)
                    recall_05pc_values.append(recall_05pc_mean)
                    recall_05pc_std_values.append(recall_05pc_std)
                    accuracy_2pc_values.append(accuracy_2pc_mean)
                    accuracy_2pc_std_values.append(accuracy_2pc_std)
                    accuracy_05pc_values.append(accuracy_05pc_mean)
                    accuracy_05pc_std_values.append(accuracy_05pc_std)
                else:
                    break
            
            mean = torch.stack(mean_values)
            sem = torch.stack(std_values)
            
            if dtype in batch.keys():
                d = batch[dtype]
            else:
                d = {}
                batch[dtype] = d

            d[key] = (mean, sem, 
                     np.array(recall_2pc_values), np.array(recall_2pc_std_values),
                     np.array(recall_05pc_values), np.array(recall_05pc_std_values),
                     np.array(accuracy_2pc_values), np.array(accuracy_2pc_std_values),
                     np.array(accuracy_05pc_values), np.array(accuracy_05pc_std_values))
            batch[dtype] = d
            
        if verbose:
            print(f"Runs: {num_runs}")
            
    if verbose:
        print(f'Budget: {budget}')
        print(f'Total queries (incl. init): {budget}')

    return batch, budget


def tabulate_regret(df, tests, budget, subdir):
    """
    Tabulates loaded regret values into an organized dataframe.
    
    Args:
        df: DataFrame to append results to
        tests: Dictionary of test results
        budget: Number of queries to process
        subdir: Directory path for extracting metadata
    
    Returns:
        df: Updated DataFrame with new results
    """
    queries = np.arange(budget) + 1
    
    for name in sorted(tests.keys()):
        result_tuple = tests[name]
        mean, sem = result_tuple[0], result_tuple[1]
        recall_2pc, recall_2pc_std = result_tuple[2], result_tuple[3]
        recall_05pc, recall_05pc_std = result_tuple[4], result_tuple[5]
        accuracy_2pc, accuracy_2pc_std = result_tuple[6], result_tuple[7]
        accuracy_05pc, accuracy_05pc_std = result_tuple[8], result_tuple[9]
        
        mean = 1 - mean
        
        if mean.size(0) < budget:
            continue
        if mean.size(0) > budget:
            mean = mean[:budget]
            sem = sem[:budget]
            recall_2pc = recall_2pc[:budget]
            recall_2pc_std = recall_2pc_std[:budget]
            recall_05pc = recall_05pc[:budget]
            recall_05pc_std = recall_05pc_std[:budget]
            accuracy_2pc = accuracy_2pc[:budget]
            accuracy_2pc_std = accuracy_2pc_std[:budget]
            accuracy_05pc = accuracy_05pc[:budget]
            accuracy_05pc_std = accuracy_05pc_std[:budget]

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

        for timestep, single_mean, single_std, r2pc, r2pc_std, r05pc, r05pc_std, a2pc, a2pc_std, a05pc, a05pc_std in zip(
            queries, np.array(mean), np.array(sem), recall_2pc, recall_2pc_std, 
            recall_05pc, recall_05pc_std, accuracy_2pc, accuracy_2pc_std, accuracy_05pc, accuracy_05pc_std):
            
            df.loc[len(df.index)] = [protein, encoding, model, acquisition, timestep, single_mean, single_std, 
                                   r2pc, r2pc_std, r05pc, r05pc_std, a2pc, a2pc_std, a05pc, a05pc_std]
    
    return df


def clean_dataframe(df):
    """
    Clean up the results dataframe with proper naming conventions.
    
    Args:
        df: Raw results DataFrame
    
    Returns:
        df: Cleaned DataFrame
    """
    df = df.drop_duplicates(subset=['Protein', 'Encoding', 'Model', 'Acquisition', 'Timestep'], keep='first')
    df['Model'] = df['Model'].replace('BOOSTING_ENSEMBLE', 'Boosting Ensemble')
    df['Model'] = df['Model'].replace('GP_BOTORCH', 'GP')
    df['Model'] = df['Model'].replace('DNN_ENSEMBLE', 'DNN Ensemble')
    df['Model'] = df['Model'].replace('DKL_BOTORCH', 'DKL')
    df['Acquisition'] = df['Acquisition'].replace('Random', 'GREEDY')
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Tabulate ALDE results from tensor files',
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
        '--output', 
        default='all_results.csv',
        help='Output CSV file name'
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
        '--max_seed_index',
        default=999999999999999,
        help='Index of seed to start with'
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
    
    # Initialize results dataframe with new columns
    df = pd.DataFrame(columns=['Protein', 'Encoding', 'Model', 'Acquisition', 'Timestep', 'Mean', 'Std',
                              'recall2pc', 'recall2pc_std', 'recall05pc', 'recall05pc_std',
                              'accuracy2pc', 'accuracy2pc_std', 'accuracy05pc', 'accuracy05pc_std'])
    
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
            
            # Load and process tensor files
            batch, budget = load_tensors(str(subdir), args.data_dir, args.verbose, args.max_seed_index)
            
            if budget == 0:
                if args.verbose:
                    print(f"No valid data found in {subdir}")
                continue
            
            # Add results to dataframe
            if 'indices' in batch:
                df = tabulate_regret(df, batch['indices'], budget, str(subdir))
    
    if df.empty:
        print("No data was processed. Check your input directories and file patterns.")
        sys.exit(1)
    
    # Clean up results
    df = clean_dataframe(df)
    
    # Save results
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(df)} rows across {len(df['Protein'].unique())} proteins and {len(df['Encoding'].unique())} encodings")
    
    if args.verbose:
        print(f"\nSummary:")
        print(f"Proteins: {sorted(df['Protein'].unique())}")
        print(f"Encodings: {sorted(df['Encoding'].unique())}")
        print(f"Models: {sorted(df['Model'].unique())}")
        print(f"Acquisitions: {sorted(df['Acquisition'].unique())}")


if __name__ == "__main__":
    main()