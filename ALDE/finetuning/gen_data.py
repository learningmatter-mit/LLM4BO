#!/usr/bin/env python3
"""
# Takes four arguments:
1. A path to a file with ESM2 embeddings a dataset of 4 amino acid sequences (.pt format)
2. A path to a file with the sequences in str format (.csv file, column name "Combo"). The sequences come in the same order as the embeddings.
2. A path to where the output file should be saved (output_name)
3. The number of artifical data sets to generate

Then, for i in range(num_datasets):
    - Draw a random weight vector w from Norm(0,1) for each dimension d in the embedding space
    - For each sequence s in the dataset, compute the dot product of w and s
    - Create a new dataset with the sequences and the dot products with columns "Combo" and "fitness"
    - Normalise the fitness values to be between 0 and 1
    - Save the dataset to a new file: "output_name_i/fitness.csv"
    - Print out summary statistics of the dataset
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

def generate_artificial_datasets(embeddings_path, sequences_path, output_name, num_datasets, start_index):
    """Generate artificial fitness datasets from ESM2 embeddings"""
    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = torch.load(embeddings_path, map_location='cpu')
    
    # Convert to numpy if it's a torch tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    # Load sequences
    print(f"Loading sequences from {sequences_path}...")
    sequences_df = pd.read_csv(sequences_path)
    
    if 'Combo' not in sequences_df.columns:
        raise ValueError("CSV file must contain a 'Combo' column with sequences")
    
    sequences = sequences_df['Combo'].tolist()
    
    # Verify dimensions match
    if len(sequences) != embeddings.shape[0]:
        raise ValueError(f"Number of sequences ({len(sequences)}) doesn't match number of embeddings ({embeddings.shape[0]})")
    
    print(f"Loaded {len(sequences)} sequences with embeddings of dimension {embeddings.shape[1]}")
    
    # Generate datasets
    for i in range(start_index, start_index + num_datasets):
        print(f"\nGenerating dataset {i}...")
        
        # Draw random weight vector from Normal(0,1)
        embedding_dim = embeddings.shape[1]
        p_active = 0.3  # fraction of dimensions that are active
        mask = np.random.binomial(1, p_active, embedding_dim)
        log_weights = np.random.normal(-6.5, 2, embedding_dim)  # μ=-6.5, σ=1.5
        weight_vector = mask * np.exp(log_weights)

        # Compute dot products (fitness values)
        fitness_values = np.dot(embeddings, weight_vector)

        # Generate log-normal-like fitness by using normal in log space
        log_fitness = np.random.normal(np.log(np.maximum(fitness_values, 1e-10)), 0.01)
        fitness_values = np.exp(log_fitness)
        # Create output directory
        output_dir = f"{output_name}_{i}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset
        dataset = pd.DataFrame({
            'Combo': sequences,
            'fitness': fitness_values
        })
        dataset['fitness'] = (dataset['fitness'] - dataset['fitness'].min()) / (dataset['fitness'].max() - dataset['fitness'].min())
        # Randomly remove 10 % of data by setting fitness to 0
        dataset.loc[dataset.sample(frac=0.1).index, 'fitness'] = 0
        dataset.loc[dataset['fitness'] < 0.001, 'fitness'] = 0
        # Save dataset
        output_file = os.path.join(output_dir, 'fitness.csv')
        dataset.to_csv(output_file, index=False)
        print(f"Saved dataset to {output_file}")


        # Assuming sequences_path is a file path, get its directory
        source_dir = os.path.dirname(sequences_path)
        source_file = os.path.join(source_dir, "onehot_x.pt")

        # Copy to output_dir
        destination = os.path.join(output_dir, "onehot_x.pt")
        shutil.copy2(source_file, destination)
                
        # Print summary statistics
        print(dataset.describe())
        print(dataset.sort_values(by='fitness', ascending=False).head(25))
        plt.hist(dataset['fitness'].apply(lambda x : np.log(x+5e-5)), bins=100)
        plt.savefig(f"{output_dir}/fitness_histogram.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate artificial fitness datasets from ESM2 embeddings')
    parser.add_argument('embeddings_path', type=str, help='Path to ESM2 embeddings file (.pt format)')
    parser.add_argument('sequences_path', type=str, help='Path to sequences CSV file with "Combo" column')
    parser.add_argument('output_name', type=str, help='Base name for output directories')
    parser.add_argument('num_datasets', type=int, help='Number of artificial datasets to generate')
    parser.add_argument('start_index', type=int, help='Start index for the output directories')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings_path}")
    
    if not os.path.exists(args.sequences_path):
        raise FileNotFoundError(f"Sequences file not found: {args.sequences_path}")
    
    if args.num_datasets <= 0:
        raise ValueError("Number of datasets must be positive")
    
    # Set random seed for reproducibility (optional)
    np.random.seed(args.start_index)
    
    # Generate datasets
    generate_artificial_datasets(
        args.embeddings_path,
        args.sequences_path, 
        args.output_name,
        args.num_datasets,
        args.start_index
    )
    
    print(f"\nSuccessfully generated {args.num_datasets} artificial datasets!")


if __name__ == "__main__":
    main()
