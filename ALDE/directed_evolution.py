#!/usr/bin/env python3
"""
Greedy Directed Evolution Algorithm for Protein Optimization

Simulates directed evolution of a protein for N steps with 1000 different random seeds.
Tests all possible amino acid position processing orderings.
"""

import pandas as pd
import numpy as np
import itertools
import random
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import csv

# Standard amino acids (excluding B, J, O, U, X, Z as per the rules)
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def load_fitness_data(protein: str) -> Tuple[Dict[str, float], float, float]:
    """Load fitness data from CSV file and return normalized values"""
    fitness_file = f"data/{protein}/fitness.csv"
    if not os.path.exists(fitness_file):
        raise FileNotFoundError(f"Fitness file not found: {fitness_file}")
    
    # First pass: collect all fitness values to calculate min/max
    all_fitness_values = []
    raw_fitness_dict = {}
    
    with open(fitness_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fitness_val = float(row['fitness'])
            raw_fitness_dict[row['Combo']] = fitness_val
            all_fitness_values.append(fitness_val)
    
    # Calculate normalization parameters
    min_fitness = min(all_fitness_values)
    max_fitness = max(all_fitness_values)
    fitness_range = max_fitness - min_fitness
    
    # Second pass: normalize fitness values (min-max normalization to 0-1)
    normalized_fitness_dict = {}
    if fitness_range > 0:
        for sequence, fitness in raw_fitness_dict.items():
            normalized_fitness_dict[sequence] = (fitness - min_fitness) / fitness_range
    
    return normalized_fitness_dict, min_fitness, max_fitness


def get_sequence_fitness(sequence: str, fitness_dict: Dict[str, float]) -> float:
    """Get fitness value for a sequence"""
    return fitness_dict.get(sequence, 0.0)


def generate_mutations(sequence: str, position: int, num_mutations: int, 
                      fitness_dict: Dict[str, float], seed: int) -> Tuple[str, float]:
    """
    Generate mutations at a specific position and return the best one
    
    Args:
        sequence: Current sequence
        position: Position to mutate (0-indexed)
        num_mutations: Number of mutations to try
        fitness_dict: Dictionary mapping sequences to fitness values
        
    Returns:
        Tuple of (best_sequence, best_fitness)
    """
    current_aa = sequence[position]
    random.seed(seed)
    
    # Get available amino acids (excluding current one)
    available_aas = [aa for aa in AMINO_ACIDS if aa != current_aa and sequence[:position] + aa + sequence[position+1:] in fitness_dict]
    
    # Sample without replacement
    num_to_sample = min(num_mutations, len(available_aas))
    sampled_aas = random.sample(available_aas, num_to_sample)
    
    best_sequence = sequence
    best_fitness = get_sequence_fitness(sequence, fitness_dict)
    all_fitness = []
    # Test each mutation
    for aa in sampled_aas:
        mutated_sequence = sequence[:position] + aa + sequence[position+1:]
        fitness = get_sequence_fitness(mutated_sequence, fitness_dict)
        all_fitness.append(fitness)
        if fitness > best_fitness:
            best_sequence = mutated_sequence
            best_fitness = fitness
    
    return best_sequence, best_fitness, all_fitness

def directed_evolution_single_run(protein: str, budget: int, start_sequence: str, 
                                position_order: List[int], seed: int, 
                                fitness_dict: Dict[str, float]) -> Tuple[str, float]:
    """
    Run directed evolution for a single position ordering and seed
    
    Args:
        protein: Protein name
        budget: Total budget for mutations
        start_sequence: Starting sequence
        position_order: Order to process positions (0-indexed)
        seed: Random seed
        fitness_dict: Fitness lookup dictionary
        
    Returns:
        Tuple of (final_sequence, final_fitness)
    """
    current_sequence = start_sequence
    current_budget = budget
    
    Nmutations = 0
    np.random.seed(seed)
    selection_fitness = []
    while current_budget > 0:
        # Calculate mutations per residue for this round
        mutations_per_residue = min(current_budget // 4, len(AMINO_ACIDS) - 1)  # 23 max
        remaining_mutations = current_budget % 4
        mutations_for_position = [mutations_per_residue] * 4
        
        # Randomly distribute the remaining mutations (only if we're not at the max)
        if remaining_mutations > 0 and mutations_per_residue < len(AMINO_ACIDS) - 1:
            # Randomly select positions to get the extra mutations
            selected_positions = random.sample(range(4), remaining_mutations)
            for pos_idx in selected_positions:
                mutations_for_position[pos_idx] += 1
        
        # Process each position in the specified order
        best_sequence = current_sequence
        best_fitness = get_sequence_fitness(current_sequence, fitness_dict)
        for i, position in enumerate(position_order):
            if mutations_for_position[i] > 0:
                Nmutations += mutations_for_position[i]
                sequence, fitness, all_fitness = generate_mutations(
                    current_sequence, position, mutations_for_position[i], 
                    fitness_dict, seed)
                selection_fitness.extend(all_fitness)
                if fitness > best_fitness:
                    best_sequence = sequence
                    best_fitness = fitness
        current_sequence = best_sequence
        # Deduct budget used
        budget_used = sum(mutations_for_position)
        current_budget -= budget_used
    final_fitness = get_sequence_fitness(current_sequence, fitness_dict)
    return current_sequence, final_fitness, selection_fitness


def directed_evolution_single_run_iterate(protein: str, budget: int, start_sequence: str, 
                                position_order: List[int], seed: int, 
                                fitness_dict: Dict[str, float]) -> Tuple[str, float]:
    """
    Run directed evolution for a single position ordering and seed
    
    Args:
        protein: Protein name
        budget: Total budget for mutations
        start_sequence: Starting sequence
        position_order: Order to process positions (0-indexed)
        seed: Random seed
        fitness_dict: Fitness lookup dictionary
        
    Returns:
        Tuple of (final_sequence, final_fitness)
    """
    current_sequence = start_sequence
    current_budget = budget
    np.random.seed(seed)
    Nmutations = 0
    selections = []
    while current_budget > 0:
        # Calculate mutations per residue for this round
        mutations_per_residue = min(current_budget // 4, len(AMINO_ACIDS) - 1)  # 23 max
        remaining_mutations = current_budget % 4
        mutations_for_position = [mutations_per_residue] * 4

        # Randomly distribute the remaining mutations (only if we're not at the max)
        if remaining_mutations > 0 and mutations_per_residue < len(AMINO_ACIDS) - 1:
            # Randomly select positions to get the extra mutations
            selected_positions = random.sample(range(4), remaining_mutations)
            for pos_idx in selected_positions:
                mutations_for_position[pos_idx] += 1
        
        # Process each position in the specified order
        for i, position in enumerate(position_order):
            if mutations_for_position[i] > 0:
                Nmutations += mutations_for_position[i]
                current_sequence, _ = generate_mutations(
                    current_sequence, position, mutations_for_position[i], 
                    fitness_dict, seed)
        selections.append(current_sequence)
        # Deduct budget used
        budget_used = sum(mutations_for_position)
        current_budget -= budget_used
    final_fitness = get_sequence_fitness(current_sequence, fitness_dict)
    selected_fitness = [get_sequence_fitness(sq, fitness_dict) for sq in selections]
    return current_sequence, final_fitness, selected_fitness


def run_directed_evolution(protein: str, budget: int, start_sequence: str, greedy: bool) -> None:
    """
    Run directed evolution with all position orderings and random seeds
    
    Args:
        protein: Protein name
        budget: Total budget for mutations
        start_sequence: Starting sequence (4 amino acids)
    """

    main_start_sequence = start_sequence
    # Load fitness data
    print(f"Loading fitness data for {protein}...")
    fitness_dict, min_fitness, max_fitness = load_fitness_data(protein)
    print(f"Loaded {len(fitness_dict)} fitness values")
    print(f"Fitness range: {min_fitness:.6f} to {max_fitness:.6f} (normalized to 0-1)")
    all_fitness = list(fitness_dict.values())
    all_fitness.sort()
    cutoff05 = np.quantile(all_fitness, 0.995)
    cutoff2 = np.quantile(all_fitness, 0.98)
    print(f"Cutoff 0.5%: {cutoff05:.6f}, Cutoff 2%: {cutoff2:.6f}")

    # Generate all position orderings (4! = 24 permutations)
    positions = list(range(4))  # [0, 1, 2, 3]
    all_orderings = list(itertools.permutations(positions))
    
    print(f"Testing {len(all_orderings)} position orderings with 1000 seeds each...")
    
    results = []
    all_final_sequences = []
    all_final_fitnesses = []
    recall05 = []
    recall2 = []
    
    # Test each position ordering
    for ordering_idx, position_order in enumerate(all_orderings):
        print(f"Processing ordering {ordering_idx + 1}/{len(all_orderings)}: {position_order}")
        
        ordering_sequences = []
        ordering_fitnesses = []
        ordering_recall05 = []
        ordering_recall2 = []
        
        # Test 1000 random seeds for this ordering
        for seed in range(10000):
            if main_start_sequence == 'Random':
                start_sequence = list(fitness_dict.keys())[np.random.randint(len(fitness_dict))]
            if greedy:
                final_sequence, final_fitness, selection_fitness = directed_evolution_single_run(
                    protein, budget, start_sequence, list(position_order), seed, fitness_dict
                )
            else:
                final_sequence, final_fitness = directed_evolution_single_run_iterate(
                    protein, budget, start_sequence, list(position_order), seed, fitness_dict
                )
            
            ordering_sequences.append(final_sequence)
            ordering_fitnesses.append(final_fitness)
            if seed == 0:
                print(selection_fitness)
            ordering_recall05.append(sum(selection_fitness > cutoff05) / (len(all_fitness)*0.005))
            ordering_recall2.append(sum(selection_fitness > cutoff2) / (len(all_fitness)*0.02))
            all_final_sequences.append(final_sequence)
            all_final_fitnesses.append(final_fitness)
        # Calculate statistics for this ordering
        mean_fitness = np.mean(ordering_fitnesses)
        max_fitness = np.max(ordering_fitnesses)
        std_fitness = np.std(ordering_fitnesses)
        best_sequence_idx = np.argmax(ordering_fitnesses)
        best_sequence = ordering_sequences[best_sequence_idx]
        recall05.append(np.mean(ordering_recall05))
        recall2.append(np.mean(ordering_recall2))
        
        results.append({
            'ordering': '-'.join(map(str, position_order)),
            'mean_fitness': mean_fitness,
            'max_fitness': max_fitness,
            'std_fitness': std_fitness,
            'best_sequence': best_sequence,
            'best_fitness': max_fitness,
            'recall05': recall05[-1],
            'recall2': recall2[-1]
        })
        
        print(f"  Mean: {mean_fitness:.6f}, Max: {max_fitness:.6f}, Std: {std_fitness:.6f}, Best: {best_sequence}, Recall@99.5pc: {recall05[-1]:.6f}, Recall@98pc: {recall2[-1]:.6f}")
        if greedy:
            break
    
    # Calculate overall statistics
    overall_mean = np.mean(all_final_fitnesses)
    overall_max = np.max(all_final_fitnesses)
    overall_std = np.std(all_final_fitnesses)
    overall_best_idx = np.argmax(all_final_fitnesses)
    overall_best_sequence = all_final_sequences[overall_best_idx]
    overall_recall05 = np.mean(recall05)
    overall_recall2 = np.mean(recall2)
    # Save results
    results_file = f"results/DE/results.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    
    # Print overall results
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Overall Mean Fitness: {overall_mean:.6f}")
    print(f"Overall Max Fitness: {overall_max:.6f}")
    print(f"Overall Std Fitness: {overall_std:.6f}")
    print(f"Overall Best Sequence: {overall_best_sequence}")
    print(f"Recall@99.5pc: {overall_recall05:.6f}, Recall@98pc: {overall_recall2:.6f}")
    print(f"Results saved to: {results_file}")
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Greedy Directed Evolution for Protein Optimization")
    parser.add_argument("protein", help="Protein name (e.g., GB1, TrpB, ParPgb)")
    parser.add_argument("budget", type=int, help="Total mutation budget")
    parser.add_argument("start_sequence", help="Starting sequence (4 amino acids)")
    parser.add_argument("--greedy", default=False, help="Optional, Mode: single or iterate", action='store_true')
    
    args = parser.parse_args()
    
    run_directed_evolution(args.protein, args.budget, args.start_sequence, args.greedy)


if __name__ == "__main__":
    main() 