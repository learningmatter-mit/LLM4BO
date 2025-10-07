"""
Tool functions for LLM Agent-based Active Learning Selector Module

This module contains utility functions for protein sequence analysis, filtering,
and similarity calculations used in the active learning workflow.

Functions:
- normalize_range: Normalize range parameters with None handling
- apply_fitness_filters: Apply fitness and std filters to sequences
- filter_by_pattern: Filter sequences by regex patterns
- pairwise_similarity: Calculate similarity between sequences
- find_similarity_threshold_start: Binary search for similarity thresholds
- filter_max_sim: Filter sequences based on maximum similarity constraints
- filter_mean_sim: Filter sequences based on mean similarity constraints

Dependencies:
- pandas/numpy: For data manipulation and analysis
- Bio.Align: For BLOSUM62 substitution matrix
- re: For regex pattern matching
"""

import pandas as pd
import numpy as np
import re
import random
from typing import List, Optional
from Bio.Align import substitution_matrices

# Load BLOSUM62 matrix for sequence similarity calculations
blosum62_matrix = substitution_matrices.load("BLOSUM62")

# Precompute minimum possible BLOSUM62 scores for each amino acid
min_possible = {a: min([blosum62_matrix.get((a, aa), -4) for aa in 'ARNDCQEGHILKMFPSTWYV'] + 
                    [blosum62_matrix.get((aa, a), -4) for aa in 'ARNDCQEGHILKMFPSTWYV']) for a in 'ARNDCQEGHILKMFPSTWYV'}


def normalize_range(range_val, default_range):
    """Normalize range parameters with None handling and edge cases."""
    if range_val is None:
        return default_range
    
    # Handle edge cases for malformed lists
    if not isinstance(range_val, list) or len(range_val) == 0:
        return default_range
    elif len(range_val) == 1:
        # Single element list: treat as minimum value, use default for maximum
        min_val = range_val[0] if range_val[0] is not None else default_range[0]
        max_val = default_range[1]
    elif len(range_val) >= 2:
        # Standard two-element list (or longer, take first 2)
        min_val = range_val[0] if range_val[0] is not None else default_range[0]
        max_val = range_val[1] if range_val[1] is not None else default_range[1]
    
    return [min_val, max_val]


def apply_fitness_filters(sequences, range_pred_fitness=None, range_pred_std=None):
    """Apply fitness and std filters to sequences inplace."""
    
    # Apply fitness filter
    if range_pred_fitness is not None and range_pred_fitness != [None, None]:
        min_fitness, max_fitness = normalize_range(range_pred_fitness, [0, float('inf')])
        sequences = sequences[
            (sequences['predictions'] >= min_fitness) & 
            (sequences['predictions'] <= max_fitness)
        ]
    
    # Apply std filter
    if range_pred_std is not None and range_pred_std != [None, None]:
        min_conf, max_conf = normalize_range(range_pred_std, [0, 1])
        sequences = sequences[
            (sequences['std_predictions'] >= min_conf) & 
            (sequences['std_predictions'] <= max_conf)
        ]
    
    return sequences


def filter_by_pattern(sequences, pattern_query):
    """Filter sequences by pattern with wildcards."""
    if not pattern_query:
        return sequences
    regex_pattern = re.compile(pattern_query)
    mask = sequences['sequence'].str.match(regex_pattern)
    return sequences[mask]


def hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences.
    
    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence
        
    Returns:
        int: Number of positions where sequences differ
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    return sum(a != b for a, b in zip(seq1, seq2))


def pairwise_similarity(seq1, seq2, method="blosum"):
    """Calculate pairwise similarity between two sequences.
    
    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence  
        method: Similarity method ('identity' or 'blosum')
        
    Returns:
        float: Similarity score
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    if method == "identity":
        return sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)
    
    elif method == "blosum":
        score = sum([blosum62_matrix[(s1, s2)] for s1, s2 in zip(seq1, seq2)])/4
        return score


def find_similarity_threshold_start(available_df, train_df, max_pairwise_similarity: float, 
                                   method: str = 'blosum', n_trials: int = 50) -> int:
    """
    Binary search utility to find starting row index where similarity threshold is likely met.
    
    Args:
        available_df: DataFrame sorted by fitness/predictions 
        train_df: Training dataframe
        max_pairwise_similarity: Similarity threshold
        n_trials: Number of random starting points to try
    
    Returns:
        Starting index for filtering (min(lowest_found - 1000, 0))
    """
    
    # Combine all sequences to check against
    comparison_sequences = train_df['sequence'].tolist()
    
    def similarity_check(row_idx: int) -> bool:
        """Check if row at index meets similarity threshold"""
        if row_idx >= len(available_df):
            return False
            
        sequence = available_df.iloc[row_idx]['sequence']
        
        for comp_seq in comparison_sequences:
            sim = pairwise_similarity(comp_seq, sequence, method=method)
            if sim >= max_pairwise_similarity:
                return False  # Fails threshold
        return True  # Passes threshold
    
    def binary_search_threshold(start_range: int, end_range: int, mid: int) -> int:
        """Binary search for first index that meets similarity threshold"""
        left, right = start_range, end_range
        result = end_range  # Default to end if no valid found
        
        while left <= right:
            if similarity_check(mid):
                result = mid
                right = mid - 1  # Look for earlier valid index
            else:
                left = mid + 1   # Current fails, look later
            mid = (left + right) // 2
        
        return result
    
    # Perform binary search from multiple random starting points
    df_len = len(available_df)
    valid_indices = []
    
    for trial in range(n_trials):
        # Random starting range for this trial
        start_point = 0
        end_point = df_len - 1
        mid = random.randint(start_point, end_point)
        
        # Binary search within this range
        threshold_idx = binary_search_threshold(start_point, end_point, mid)
        
        # Verify the result is actually valid
        if threshold_idx < df_len and similarity_check(threshold_idx):
            valid_indices.append(threshold_idx)
    
    if not valid_indices:
        print("Warning: No valid starting indices found in binary search")
        return 0
    
    # Statistics
    mean_idx = np.mean(valid_indices)
    lowest_idx = min(valid_indices)
    
    print(f"Binary search results - Mean: {mean_idx:.1f}, Lowest: {lowest_idx}, saved {lowest_idx - 250 - 100*np.log2(len(available_df)):.1f} computations")
    
    # Return conservative starting point
    return max(0, lowest_idx - 250)


def filter_max_sim(available_df, train_df, n_select: int,
                    pairwise_method: str = 'blosum', 
                    max_pairwise_similarity: Optional[float] = None,
                    to_n_train_range : Optional[List[Optional[float]]] = [None, None],
                    n_train_range : Optional[float] = 3,
                    sorted_by: str = 'predictions', 
                    prev_selected: Optional[List[str]] = None):
    """
    Filter sequences based on similarity using greedy approach with priority queue
    
    Args:
        available_df: DataFrame with sequences and predictions
        n_select: Number of sequences to select initially
        pairwise_method: Method for similarity calculation ('identity' or 'blosum')
        max_pairwise_similarity: Maximum allowed mean of max pairwise similarities
        prediction_column: Column name containing prediction scores
        prev_selected: List of previously selected sequences to avoid similarity with
    
    Returns:
        DataFrame of selected sequences
    """
    
    if max_pairwise_similarity is None and to_n_train_range is None:
        # If no similarity filter, just return top n_select
        return available_df.head(n_select).copy()
    
    if prev_selected is None:
        prev_selected = []
    if n_train_range is None:
        n_train_range = 3
    if to_n_train_range is None:
        to_n_train_range = [None, None]
    to_n_train_range[0] = -100 if to_n_train_range[0] is None else to_n_train_range[0]
    to_n_train_range[1] = 100 if to_n_train_range[1] is None else to_n_train_range[1]
    max_pairwise_similarity = 100 if max_pairwise_similarity is None else max_pairwise_similarity
    # Sort by prediction score (assuming higher is better)    
    prev_seqs = prev_selected.copy()  # Don't modify original
    train_seqs = train_df.sort_values(by='fitness', ascending=False).loc[:n_train_range, 'sequence'].tolist()


    selected = []
    sims_prev = []
    sims_train = []
    inds = []

    def get_sims(sq):
        prev_sim = -100
        train_sim = 0
        for sq2 in prev_seqs + selected:
            sim = pairwise_similarity(sq2, sq, method=pairwise_method)
            prev_sim = max(prev_sim, sim)
        for seq2 in train_seqs:
            train_sim += pairwise_similarity(seq2, sq, method=pairwise_method)
        return prev_sim, train_sim/len(train_seqs)

    for index, row in available_df.iterrows():
        sq = row['sequence']
        prev_sim, train_sim = get_sims(sq)
        if prev_sim <= max_pairwise_similarity and train_sim >= to_n_train_range[0] and train_sim <= to_n_train_range[1]:
            selected.append(sq)
            sims_prev.append(prev_sim)
            sims_train.append(train_sim)
            inds.append(index)
            if len(selected) >= n_select:
                break
    for i, sq in enumerate(selected):
        for j, sq2 in enumerate(selected):
            if i == j:
                continue
            sim = pairwise_similarity(sq, sq2, method=pairwise_method)
            if sim > sims_prev[i]:
                sims_prev[i] = sim

    df_sel = available_df.loc[inds]
    df_sel['max_similarity_to_selected'] = sims_prev
    df_sel['mean_similarity_to_training'] = sims_train
    return df_sel


def filter_mean_sim(available_df, n_select: int, 
                    pairwise_method: str = 'identity', 
                    max_pairwise_similarity: Optional[float] = None,
                    prediction_column: str = 'predictions', 
                    prev_selected: List[str] = None):
    """
    Filter sequences based on similarity using greedy approach with priority queue
    
    Args:
        available_df: DataFrame with sequences and predictions
        n_select: Number of sequences to select initially
        pairwise_method: Method for similarity calculation ('identity' or 'blosum')
        max_pairwise_similarity: Maximum allowed mean of max pairwise similarities
        prediction_column: Column name containing prediction scores
    
    Returns:
        DataFrame of selected sequences
    """
    
    if max_pairwise_similarity is None:
        # If no similarity filter, just return top n_select
        return available_df.head(n_select).copy()
    # Initialize with top n_select sequences
    selected_sequences = available_df[:n_select]['sequence'].tolist()
    selected_indices = available_df[:n_select].index.tolist()
    
    # Track remaining candidates
    remaining_candidates = available_df[n_select:].index.tolist()
    candidate_idx = 0
    
    max_similarities = None  # Initialize to track final similarities
    
    while True:
        # Calculate all pairwise similarities
        similarity_matrix = np.zeros((len(selected_sequences), len(selected_sequences)))
        
        for i in range(len(selected_sequences)):
            for j in range(i + 1, len(selected_sequences)):
                sim = pairwise_similarity(
                    selected_sequences[i],
                    selected_sequences[j],
                    method=pairwise_method
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        # Calculate max similarity for each sequence
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Check if mean of max similarities exceeds threshold
        mean_max_sim = np.mean(max_similarities)
        
        if mean_max_sim <= max_pairwise_similarity:
            break
        
        # Find the sequence with highest max similarity
        # If multiple sequences have the same max similarity, take the first one
        remove_idx = np.argmax(max_similarities)
        
        # Remove the sequence
        selected_sequences.pop(remove_idx)
        selected_indices.pop(remove_idx)
        
        # Add next candidate if available
        if candidate_idx < len(remaining_candidates):
            next_candidate_idx = remaining_candidates[candidate_idx]
            # Get the sequence string, not the entire Series
            next_sequence = available_df.loc[next_candidate_idx, 'sequence']
            selected_sequences.append(next_sequence)
            selected_indices.append(next_candidate_idx)
            candidate_idx += 1
        else:
            # No more candidates available
            break
    
    # Convert back to DataFrame
    selected_df = available_df.loc[selected_indices].copy()
    
    # Add pairwise similarity information if we have it
    if max_similarities is not None and len(max_similarities) == len(selected_df):
        selected_df['max_pairwise_similarity'] = max_similarities
    
    return selected_df


def filter_hamming_distance(available_df, train_df, n_select: int,
                           min_hamming_distance: Optional[int] = None,
                           max_hamming_distance: Optional[int] = None,
                           n_best_training: Optional[int] = 5,
                           prev_selected: Optional[List[str]] = None,
                           max_position_frequency_batch: Optional[float] = None,
                           max_position_frequency_training: Optional[float] = None):
    """
    Filter sequences based on Hamming distance constraints and position frequency caps.
    
    Applies multiple filtering steps in order during greedy selection:
    1. Position frequency caps (both training and batch constraints checked together)
    2. Distance to top training sequences must be within [min_hamming_distance, max_hamming_distance]
    3. Pairwise distances between selected sequences must be >= min_hamming_distance (no maximum enforced)
    
    Args:
        available_df: DataFrame with sequences and predictions, sorted by preference
        train_df: Training dataframe with sequences and fitness
        n_select: Number of sequences to select
        min_hamming_distance: Minimum allowed Hamming distance (default: None = no minimum)
        max_hamming_distance: Maximum allowed Hamming distance (default: None = no maximum)
        n_best_training: Number of top training sequences to check distance against
        prev_selected: List of previously selected sequences to avoid
        max_position_frequency_batch: Maximum frequency (0.0-1.0) any amino acid can appear at any position in selected batch
        max_position_frequency_training: Maximum frequency (0.0-1.0) any amino acid can appear at any position in training data
    
    Returns:
        DataFrame of selected sequences with distance and frequency information
        
    Example:
        # Select 10 sequences with diverse amino acids at each position
        filter_hamming_distance(df, train_df, 10, 
                               min_hamming_distance=2, max_hamming_distance=3, 
                               max_position_frequency_batch=0.6, max_position_frequency_training=0.8)
    """
    
    # Handle default values
    if prev_selected is None:
        prev_selected = []
    if n_best_training is None:
        n_best_training = 5
    if min_hamming_distance is None:
        min_hamming_distance = 0
    if max_hamming_distance is None:
        max_hamming_distance = float('inf')
    
    # If no constraints at all, just return top n_select
    if (min_hamming_distance == 0 and max_hamming_distance == float('inf') and 
        max_position_frequency_batch is None and max_position_frequency_training is None):
        return available_df.head(n_select).copy()
    
    def calculate_position_frequencies(sequences):
        """Calculate amino acid frequencies at each position"""
        if not sequences:
            return {}
        
        seq_length = len(sequences[0])
        position_frequencies = {}
        
        for pos in range(seq_length):
            aa_counts = {}
            for seq in sequences:
                aa = seq[pos]
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Convert counts to frequencies
            total = len(sequences)
            position_frequencies[pos] = {aa: count/total for aa, count in aa_counts.items()}
        
        return position_frequencies
    
    def violates_position_frequency_cap(sequence, selected_sequences, max_batch_frequency, train_freqs, max_train_frequency):
        """Check if adding sequence would violate position frequency caps"""
        # Check training frequency constraint first (faster)
        if max_train_frequency is not None and train_freqs is not None:
            for pos in range(len(sequence)):
                aa = sequence[pos]
                if train_freqs[pos].get(aa, 0) > max_train_frequency:
                    return True
        
        # Check batch frequency constraint
        if max_batch_frequency is None or len(selected_sequences) == 0:
            return False  # No batch constraint or first sequence
        
        batch_size = len(selected_sequences) + 1  # Including the new sequence
        
        # For small batches, use simple counting
        if len(selected_sequences) < 20:
            for pos in range(len(sequence)):
                aa = sequence[pos]
                count = sum(1 for seq in selected_sequences if seq[pos] == aa) + 1
                if count / batch_size > max_batch_frequency:
                    return True
            return False
        
        # For larger batches, use vectorized approach
        else:
            # Convert to numpy array for faster computation
            selected_array = np.array([list(seq) for seq in selected_sequences])
            sequence_array = np.array(list(sequence))
            
            for pos in range(len(sequence)):
                aa = sequence_array[pos]
                # Count occurrences at this position
                count = np.sum(selected_array[:, pos] == aa) + 1
                if count / batch_size > max_batch_frequency:
                    return True
            return False
    
    # Precompute training position frequencies if needed
    train_position_freqs = None
    if max_position_frequency_training is not None:
        all_train_sequences = train_df['sequence'].tolist()
        if len(all_train_sequences) > 0:
            # Convert to numpy array for faster processing
            train_array = np.array([list(seq) for seq in all_train_sequences])
            seq_length = len(all_train_sequences[0])
            
            train_position_freqs = {}
            for pos in range(seq_length):
                # Get all amino acids at this position
                pos_aas = train_array[:, pos]
                # Count frequencies using numpy
                unique_aas, counts = np.unique(pos_aas, return_counts=True)
                frequencies = counts / len(pos_aas)
                train_position_freqs[pos] = dict(zip(unique_aas, frequencies))
    
    # Get top training sequences to compare against
    train_seqs = train_df.sort_values(by='fitness', ascending=False).head(n_best_training)['sequence'].tolist()
    prev_seqs = prev_selected.copy()

    selected = []
    distances_to_training = []
    distances_to_selected = []
    indices = []

    def check_training_distance(seq):
        """Check if sequence meets training distance constraints"""
        for train_seq in train_seqs:
            dist = hamming_distance(seq, train_seq)
            if not (min_hamming_distance <= dist <= max_hamming_distance):
                return False, -1
        # Return average distance to training sequences
        avg_dist = np.mean([hamming_distance(seq, train_seq) for train_seq in train_seqs])
        return True, avg_dist

    def check_selected_distance(seq):
        """Check if sequence meets distance constraints with already selected sequences"""
        min_dist_to_selected = float('inf')
        for sel_seq in prev_seqs + selected:
            dist = hamming_distance(seq, sel_seq)
            min_dist_to_selected = min(min_dist_to_selected, dist)
            # Only check maximum distance constraint for pairwise filtering
            if min_hamming_distance is not None and dist < min_hamming_distance:
                return False, min_dist_to_selected
        return True, min_dist_to_selected

    for index, row in available_df.iterrows():
        seq = row['sequence']
        
        # Step 1: Check position frequency caps (both training and batch, greedy)
        if violates_position_frequency_cap(seq, selected, max_position_frequency_batch, 
                                          train_position_freqs, max_position_frequency_training):
            continue
        
        # Step 2: Check training distance constraints
        training_ok, avg_training_dist = check_training_distance(seq)
        if not training_ok:
            continue
            
        # Step 3: Check selected sequence distance constraints
        selected_ok, min_selected_dist = check_selected_distance(seq)
        if not selected_ok:
            continue
        
        # Add to selection
        selected.append(seq)
        distances_to_training.append(avg_training_dist)
        distances_to_selected.append(min_selected_dist)
        indices.append(index)
        
        if len(selected) >= n_select:
            break

    # Create result DataFrame
    if indices:
        result_df = available_df.loc[indices].copy()
        result_df['avg_hamming_to_training'] = distances_to_training
        result_df['min_hamming_to_selected'] = distances_to_selected
        return result_df
    else:
        # Return empty DataFrame with same structure
        result_df = available_df.iloc[0:0].copy()
        result_df['avg_hamming_to_training'] = []
        result_df['min_hamming_to_selected'] = []
        return result_df
