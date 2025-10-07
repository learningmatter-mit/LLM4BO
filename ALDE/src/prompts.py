"""
Prompt Generation and Summary Management Module

This module handles all prompt generation for the LLM-based active learning selector,
as well as enhanced summary storage and performance tracking functionality.

Key Components:
- Prompt generation from YAML templates with AL state information
- Enhanced summary storage with strategy performance tracking
- Backward-compatible summary loading and management
- Performance metrics calculation and formatting

Dependencies:
- pandas/numpy: For data manipulation and analysis
- yaml: For prompt template loading
- json: For summary file storage
- pathlib: For file path handling
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def compact_df(df, index=True, sig_digits=3):
    """
    Ultra-compact DataFrame display for token efficiency.
    Maintains structure while minimizing tokens.
    
    Args:
        df: DataFrame to display
        index: Whether to include row indices (default True)
        sig_digits: Significant digits for numeric columns (default 3)
    
    Returns:
        str: Compact string representation
    """
    def format_num(x):
        if pd.isna(x):
            return 'nan'
        if isinstance(x, (int, np.integer)):
            return str(x)
        if isinstance(x, (float, np.floating)):
            # Format to sig_digits significant figures
            if x == 0:
                return '0'
            # Use scientific notation if very large/small, otherwise decimal
            if abs(x) >= 10**(sig_digits) or (abs(x) < 10**(-sig_digits+1) and x != 0):
                return f"{x:.{sig_digits-1}e}"
            else:
                # Round to sig_digits significant figures
                return f"{x:.{max(0, sig_digits - 1 - int(np.floor(np.log10(abs(x)))))}f}".rstrip('0').rstrip('.')
        return str(x)
    
    # Get column names - use shortest reasonable abbreviations
    cols = list(df.columns)
    
    # Format header
    if index:
        header = "index|" + "|".join(cols)
    else:
        header = "|".join(cols)
    
    # Format rows
    rows = []
    for idx, row in df.iterrows():
        formatted_row = []
        if index:
            formatted_row.append(str(idx))
        
        for col in cols:
            val = row[col]
            if pd.api.types.is_numeric_dtype(type(val)):
                formatted_row.append(format_num(val))
            else:
                # For strings, keep as-is (mainly sequences)
                formatted_row.append(str(val))
        
        rows.append("|".join(formatted_row))
    
    return header + "\n" + "\n".join(rows)

def get_prompts(oracle_model: str, batch_size: int, total_cycles: int, cycle: int, 
               train_df: pd.DataFrame, pred_df: pd.DataFrame, protein: str, 
               summaries_file: str = 'summaries.json', include_sequences: bool = True):
    """Generate prompts for strategy and selection phases.
    
    Loads prompt templates from YAML configuration and formats them with current AL state
    information including cycle details, training data statistics, and prediction summaries.
    
    Args:
        oracle_model: Name of the oracle model being used
        batch_size: Number of compounds to select in this cycle
        total_cycles: Total number of AL cycles planned
        cycle: Current cycle number
        train_df: Training data DataFrame with sequence and fitness columns
        pred_df: Prediction DataFrame with sequence, predictions, and std_predictions
        protein: Name of the target protein
        summaries_file: Path to JSON file containing cycle summaries
        include_sequences: Whether to include sequence data in summaries (default True)
        
    Returns:
        tuple: (strategy_prompt, selection_prompt, implementer_system_prompt) - Formatted prompt strings
        
    Raises:
        FileNotFoundError: If prompts_LLMChainSelector.yaml is not found
        KeyError: If required prompt templates are missing from YAML
    """
    with open('src/prompts_LLMChainSelector.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    strategy_prompt = config['strategy_prompt']['template']
    selection_prompt = config['selection_prompt']['template']
    cycle_info = config['cycle_info']['template']
    implementer_system_prompt = config['implementer_system_prompt']['template']
    description = config['backgrounds'][protein]
    
    # Ensure cycle 0 summary exists with initial training data
    save_cycle_0_data(train_df, summaries_file, include_sequences)
    
    # Load existing summaries from file (including the new cycle 0)
    agent_summaries = load_agent_summaries(summaries_file)
    #BLIND:
    description = "No background available, we enter the campaign blind."
    protein = "unknown"
    def safe_format(x):
        return f"{x:.3f}" if x is not None else "N/A"
    # Format all cycle summaries (including cycle 0 with initial training data)
    if agent_summaries:
        past_cycles_data = 'Previous Cycle Performance:\n'
        for cycle_data in agent_summaries:
            past_cycles_data += f"""<cycle {cycle_data['cycle']} summary>
Cycle : {cycle_data['cycle']}
{cycle_data['summary']}
</cycle {cycle_data['cycle']} summary>
"""
    else:
        past_cycles_data = 'No previous cycles available.'
    # Format cycle info
    cycle_info = cycle_info.format(
        cycles_completed=cycle-1, total_cycles=total_cycles, cycle=cycle,
        batch_size=batch_size, oracle_model=oracle_model, protein=protein
        )

    strategy_prompt = strategy_prompt.format(
        cycle_info=cycle_info, batch_size=batch_size,
        protein=protein, past_cycles_data=past_cycles_data, background=description,
        train_df_describe=train_df[['sequence', 'fitness']].describe().round(4), 
        train_df='')
    selection_prompt = selection_prompt.format(cycle_info=cycle_info, pred_df_describe=pred_df[['sequence', 'predictions', 'std_predictions']].describe().round(4))
    return strategy_prompt, selection_prompt, implementer_system_prompt


def save_agent_strategies(cycle: int, strategies: List[str], strategy_selections: List[List[int]], 
                         pred_df: pd.DataFrame, summary_text: str, summaries_file: str = 'summaries.json'):
    """Save agent strategies and predictions immediately after agent selection.
    
    Args:
        cycle: Current cycle number
        strategies: List of strategy descriptions  
        strategy_selections: List of lists of selected indices for each strategy
        pred_df: Prediction DataFrame to get sequences and predictions
        summary_text: Generated summary text from agent
        summaries_file: Path to JSON file for storing summaries
    """
    summaries = load_agent_summaries(summaries_file)
    
    # Ensure list is long enough for this cycle
    while len(summaries) <= cycle:
        summaries.append({})
    
    # Convert strategy selections to new format
    strategies_data = []
    for i, (strategy_desc, selected_indices) in enumerate(zip(strategies, strategy_selections)):
        if not selected_indices:
            strategies_data.append({
                "strategy": strategy_desc,
                "selected": [],
                "validated_fitness": [],
                "predicted_fitness": [],
                "predicted_std": [],
                "mean_fitness": None,
                "max_fitness": None,
                "oracle_rmse": None
            })
        else:
            # Get sequences and predictions for selected indices
            selected_sequences = pred_df.loc[selected_indices]['sequence'].tolist()
            predicted_fitness = pred_df.loc[selected_indices]['predictions'].tolist()
            predicted_std = pred_df.loc[selected_indices]['std_predictions'].tolist()
            
            strategies_data.append({
                "strategy": strategy_desc,
                "selected": selected_sequences,
                "validated_fitness": [],  # Will be filled later
                "predicted_fitness": predicted_fitness,
                "predicted_std": predicted_std,
                "mean_fitness": None,  # Will be computed later
                "max_fitness": None,   # Will be computed later
                "oracle_rmse": None    # Will be computed later
            })
    
    # Store cycle data
    cycle_data = {
        "cycle": cycle,
        "summary": summary_text,
        "mean_fitness": None,  # Will be computed later
        "max_fitness": None,   # Will be computed later
        "oracle_rmse": None,   # Will be computed later
        "strategies": strategies_data
    }
    
    summaries[cycle] = cycle_data
    
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved cycle {cycle} agent strategies to {summaries_file}")


def update_agent_performance(cycle: int, train_df: pd.DataFrame, summaries_file: str = 'summaries.json', include_sequences: bool = True):
    """Update cycle with validated fitness and performance metrics after experimental validation.
    
    Args:
        cycle: Current cycle number
        train_df: Training DataFrame with experimental results (sequence, fitness columns)
        summaries_file: Path to JSON file containing summaries
    """
    summaries = load_agent_summaries(summaries_file)
    if cycle >= len(summaries) or not summaries[cycle]:
        print(f"No strategy data found for cycle {cycle}")
        return
    
    cycle_data = summaries[cycle]
    
    # Update each strategy with validated fitness
    for strategy_data in cycle_data["strategies"]:
        selected_sequences = strategy_data["selected"]
        predicted_fitness = strategy_data["predicted_fitness"]
        
        if not selected_sequences:
            # No sequences selected by this strategy
            continue
            
        # Match sequences in training data
        sequence_mask = train_df['sequence'].isin(selected_sequences)
        matched_data = train_df[sequence_mask]
        
        if len(matched_data) == 0:
            print(f"Warning: No sequences from strategy '{strategy_data['strategy'][:50]}...' found in training data for cycle {cycle}")
            continue
        
        # Get validated fitness in same order as selected sequences
        validated_fitness = []
        matched_predicted = []
        
        for seq in selected_sequences:
            seq_matches = matched_data[matched_data['sequence'] == seq]
            if len(seq_matches) > 0:
                fitness_val = seq_matches['fitness'].iloc[0]
                validated_fitness.append(float(fitness_val))
                seq_idx = selected_sequences.index(seq)
                matched_predicted.append(predicted_fitness[seq_idx])
        
        # Update strategy data
        strategy_data["validated_fitness"] = validated_fitness
        
        # Only compute metrics if we have validated fitness values
        if validated_fitness:
            strategy_data["mean_fitness"] = float(np.mean(validated_fitness))
            strategy_data["max_fitness"] = float(np.max(validated_fitness))
        
            if matched_predicted and len(matched_predicted) == len(validated_fitness):
                rmse = float(np.sqrt(np.mean((np.array(validated_fitness) - np.array(matched_predicted))**2)))
                strategy_data["oracle_rmse"] = rmse
        else:
            # No validated fitness data - keep as None
            strategy_data["mean_fitness"] = None
            strategy_data["max_fitness"] = None
            strategy_data["oracle_rmse"] = None

    # Compute cycle-level metrics across all strategies with validated data
    all_validated_fitness = []
    all_predicted_fitness = []
    
    for strategy_data in cycle_data["strategies"]:
        if strategy_data["validated_fitness"]:
            all_validated_fitness.extend(strategy_data["validated_fitness"])
            if strategy_data["predicted_fitness"]:
                all_predicted_fitness.extend(strategy_data["predicted_fitness"])
    
    # Only compute cycle metrics if we have validated data
    if all_validated_fitness:
        cycle_data["mean_fitness"] = float(np.mean(all_validated_fitness))
        cycle_data["max_fitness"] = float(np.max(all_validated_fitness))
        
        if all_predicted_fitness and len(all_predicted_fitness) == len(all_validated_fitness):
            cycle_rmse = float(np.sqrt(np.mean((np.array(all_validated_fitness) - np.array(all_predicted_fitness))**2)))
            cycle_data["oracle_rmse"] = cycle_rmse
    else:
        # No validated data found - keep as None
        cycle_data["mean_fitness"] = None
        cycle_data["max_fitness"] = None
        cycle_data["oracle_rmse"] = None
    
    # Regenerate summary with performance metrics
    enhanced_summary = generate_agent_summary(cycle_data, train_df, include_sequences=include_sequences)
    cycle_data["summary"] = enhanced_summary
    
    summaries[cycle] = cycle_data
    
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Updated cycle {cycle} with performance metrics")


def save_cycle_0_data(train_df: pd.DataFrame, summaries_file: str = 'summaries.json', include_sequences: bool = True):
    """Create cycle 0 entry with initial training data.
    
    Args:
        train_df: Training DataFrame with initial sequences and fitness
        summaries_file: Path to JSON file for storing summaries
        include_sequences: Whether to include sequence data in cycle 0 summary (default True)
    """
    summaries = load_agent_summaries(summaries_file)
    
    # Check if cycle 0 already exists
    if len(summaries) > 0 and summaries[0] and summaries[0].get("cycle") == 0:
        return
    
    # Create cycle 0 data
    cycle_0_strategy = {
        "strategy": """Initial training data sampled randomly""",
        "selected": train_df['sequence'].tolist(),
        "validated_fitness": train_df['fitness'].tolist(),
        "predicted_fitness": [0.0] * len(train_df),  # No predictions for initial data
        "predicted_std": [0.0] * len(train_df),      # No std for initial data
        "mean_fitness": float(train_df['fitness'].mean()),
        "max_fitness": float(train_df['fitness'].max()),
        "oracle_rmse": None  # No predictions to compare
    }
    
    cycle_0_data = {
        "cycle": 0,
        "summary": "Initial validated sequences used to train initial oracle",
        "mean_fitness": float(train_df['fitness'].mean()),
        "max_fitness": float(train_df['fitness'].max()),
        "oracle_rmse": None,
        "strategies": [cycle_0_strategy]
    }
    
    # Generate formatted summary
    enhanced_summary = generate_agent_summary(cycle_0_data, train_df, include_sequences)
    cycle_0_data["summary"] = enhanced_summary
    
    # Ensure summaries list is long enough for cycle 0
    if len(summaries) == 0:
        summaries.append(cycle_0_data)
    else:
        summaries[0] = cycle_0_data
    
    with open(summaries_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Created cycle 0 summary with initial training data")


def load_agent_summaries(summaries_file: str) -> List[dict]:
    """Load agent summaries from JSON file with new structure.
    
    Args:
        summaries_file: Path to JSON file containing summaries
        
    Returns:
        List of cycle dictionaries, empty list if file doesn't exist
    """
    if not Path(summaries_file).exists():
        return []
    
    try:
        with open(summaries_file, 'r') as f:
            summaries = json.load(f)
        return summaries if isinstance(summaries, list) else []
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not load summaries from {summaries_file}: {e}")
        return []


def generate_agent_summary(cycle_data: dict, train_df: pd.DataFrame = None, include_sequences: bool = True) -> str:
    """Generate summary text with performance metrics in the new format.
    
    Args:
        cycle_data: Dictionary containing cycle information
        train_df: Training DataFrame for generating compact displays
        include_sequences: Whether to include sequence data in the summary (default True)
        
    Returns:
        Enhanced summary string with performance metrics
    """
    cycle = cycle_data["cycle"]
    base_summary = cycle_data.get("summary", f"Cycle {cycle} summary")
    strategies = cycle_data["strategies"]
    
    # Start with cycle summary
    enhanced_summary = f"**cycle {cycle} summary**\n\n{base_summary}\n\n"
    
    if not strategies:
        return enhanced_summary
    
    # Add selection strategies section
    enhanced_summary += "**selection strategies**\n"
    
    for i, strategy_data in enumerate(strategies):
        strategy_num = i + 1
        strategy_text = strategy_data["strategy"]
        
        enhanced_summary += f"Strategy {strategy_num}: {strategy_text}\n"
        
        # Handle different cases
        if not strategy_data["selected"]:
            enhanced_summary += "No sequences selected by this strategy\n"
        elif not strategy_data["validated_fitness"]:
            enhanced_summary += "Experimental validation pending\n"
        else:
            # Show performance metrics
            mean_fitness = strategy_data["mean_fitness"]
            max_fitness = strategy_data["max_fitness"]
            oracle_rmse = strategy_data["oracle_rmse"]
            
            # Handle None values gracefully
            if mean_fitness is not None and max_fitness is not None:
                enhanced_summary += f"Mean yielded fitness: {mean_fitness:.3f}, "
                enhanced_summary += f"max yielded fitness: {max_fitness:.3f}, "
                
                if oracle_rmse is not None:
                    enhanced_summary += f"Oracle RMSE on selection subset: {oracle_rmse:.3f}\n"
                else:
                    enhanced_summary += f"Oracle RMSE on selection subset: N/A (no predictions for initial training data)\n"
            else:
                enhanced_summary += "Performance metrics not available (no sequences found in training data)\n"
            
            # Add compact dataframe of selected sequences if available
            if train_df is not None and strategy_data["validated_fitness"] and include_sequences:
                selected_sequences = strategy_data["selected"][:len(strategy_data["validated_fitness"])]
                validated_fitness = strategy_data["validated_fitness"]
                
                # Create dataframe for display
                strategy_df = pd.DataFrame({
                    'sequence': selected_sequences,
                    'fitness': validated_fitness
                })
                
                if include_sequences and len(strategy_df) > 0:
                    enhanced_summary += f"Best 3 sequences: \n {compact_df(strategy_df.sort_values(by='fitness', ascending=False).head(3), index=False)}\n"
        
        enhanced_summary += "\n"
    
    return enhanced_summary


# Export functions for external use
__all__ = [
    # Core utility functions
    'compact_df',
    'get_prompts',
    
    # New agent functions (preferred)
    'save_agent_strategies',
    'update_agent_performance', 
    'save_cycle_0_data',
    'load_agent_summaries',
    'generate_agent_summary'
] 