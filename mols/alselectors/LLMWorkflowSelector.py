"""
LLMAgentSelector: LLM-driven Active Learning Selector

This module defines the LLMAgentSelector class, which leverages a multi-agent LLM-based
strategy for selecting compounds in active learning cycles for molecular property prediction.
It integrates with the agent-based workflow defined in agent.py and adheres to the
Selector interface.

Key Features:
- Modular, extensible selector for AL cycles
- Uses LLMs to analyze chemical space and select candidates
- Logs selection summaries and final indices for reproducibility and audit

Dependencies:
- pandas, numpy, logging
- agent.run_selector_agent (LLM agent workflow)
- utils.Selector.Selector (base class)
"""

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Any, List, Tuple, Dict
import logging
import random
import re
import time
import json
from anthropic._exceptions import OverloadedError

from alselectors.agent import run_selector_agent
from utils.Selector import Selector
from utils.APIUtils import exponential_backoff_retry

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Adjust based on your structure
    
class LLMWorkflowSelector(Selector):
    """LLM-driven Active Learning Selector for molecular property prediction.
    
    This selector uses a multi-agent LLM-based workflow to analyze chemical space and select
    the most informative compounds for labeling in each active learning cycle. It supports
    batch-based selection, robust logging, and is compatible with the broader AL framework.
    
    Attributes:
        batch_size (int): Number of samples to select per AL cycle
        random_seed (int): Random seed for reproducibility
        logger (logging.Logger): Logger for experiment tracking
    """
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize the LLMAgentSelector.
        
        Args:
            batch_size (int): Number of samples to select per AL cycle
            random_seed (int): Random seed for reproducibility
            **kwargs: Additional keyword arguments for Selector base class
        """
        super().__init__(batch_size, random_seed, **kwargs)
        self.simple_agent = 0
        
        # Store tool control flags from selector_kwargs
        self.disable_ucb = kwargs.get('disable_ucb', False)
        self.disable_smarts = kwargs.get('disable_smarts', False)
        self.disable_tanimoto = kwargs.get('disable_tanimoto', False)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with batch_size={batch_size}")
        if any([self.disable_ucb, self.disable_smarts, self.disable_tanimoto]):
            disabled_tools = []
            if self.disable_ucb: disabled_tools.append("UCB")
            if self.disable_smarts: disabled_tools.append("SMARTS")
            if self.disable_tanimoto: disabled_tools.append("Tanimoto")
            self.logger.info(f"Ablation study mode: Disabled tools: {disabled_tools}")

    
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select samples using LLM to analyze chemical space and process candidates in chunks.
        
        Args:
            predictions (np.ndarray): Model predictions
            confidence_scores (np.ndarray): Model confidence scores
            training_data (Tuple[Dict[str, np.ndarray], np.ndarray]):
                Current training set (features_dict, labels)
            unlabeled_data (Dict[str, np.ndarray]):
                Dictionary of unlabeled features, must contain 'SMILES' key
            random_seed (int): Random seed for reproducibility
            **kwargs: Additional parameters
                - cycle: Current AL cycle
                - total_cycles: Total number of AL cycles
                - oracle_name: Name of the oracle model
                - protein: Name of the protein
        
        Returns:
            List[int]: List of selected sample indices
        
        Raises:
            ValueError: If 'SMILES' key is missing or not enough samples for batch size
        """
        # Validate inputs
        if 'SMILES' not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")
        
        n_samples = len(unlabeled_data['SMILES'])
        if n_samples < self.batch_size:
            self.logger.error(f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}")
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
        # Get required parameters
        cycle = int(kwargs.get('cycle', 1))
        total_cycles = kwargs.get('total_cycles', 6)
        oracle_name = kwargs.get('oracle_name', 'Unknown')
        protein = kwargs.get('protein', 'Unknown')
        y_unlabeled = np.array(kwargs.get('y_unlabeled', None))
        if cycle == 1:
            #Empty "alselectors/past_cycle_report.txt"
            with open("alselectors/past_cycle_report.txt", "w") as f:
                f.write("")
        with open("alselectors/past_cycle_report.txt", "r") as f:
            past_cycle_report = f.read()

        self.logger.info(f"Starting selection for cycle {cycle}/{total_cycles}")
        self.logger.info(f"Protein: {protein}, Oracle: {oracle_name}")
        self.logger.info(f"Unlabeled pool size: {n_samples}")

        # Validate training data
        if len(training_data[0]['SMILES']) != len(training_data[1]) or len(training_data[0]['SMILES']) != len(training_data[0]['cycle_added']):
            raise ValueError(f"Training data arrays have different lengths: SMILES={len(training_data[0]['SMILES'])}, affinity={len(training_data[1])}, cycle_added={len(training_data[0]['cycle_added'])}")
        
        train_df = pd.DataFrame({
            'SMILES': training_data[0]['SMILES'],
            'affinity': training_data[1], 
            'cycle_added': training_data[0]['cycle_added']
        })

        # Validate prediction data
        if len(unlabeled_data['SMILES']) != len(predictions) or len(unlabeled_data['SMILES']) != len(confidence_scores):
            raise ValueError(f"Prediction data arrays have different lengths: SMILES={len(unlabeled_data['SMILES'])}, predictions={len(predictions)}, confidence_scores={len(confidence_scores)}")
        
        # Create DataFrame for candidates
        pred_df = pd.DataFrame({
            'SMILES': unlabeled_data['SMILES'],
            'predictions': predictions,
            'confidence_scores': confidence_scores
        })
        
        # Check for NaN values in predictions and confidence scores
        if np.any(np.isnan(predictions)):
            self.logger.warning("NaN values found in predictions, replacing with 0")
            predictions = np.nan_to_num(predictions, nan=0.0)
            pred_df['predictions'] = predictions
            
        if np.any(np.isnan(confidence_scores)):
            self.logger.warning("NaN values found in confidence scores, replacing with 0.5")
            confidence_scores = np.nan_to_num(confidence_scores, nan=0.5)
            pred_df['confidence_scores'] = confidence_scores
        @exponential_backoff_retry(max_attempts=3, base_delay=10.0, max_delay=300.0)
        def call_selector_agent():
            return run_selector_agent(
                oracle_model = oracle_name, batch_size = self.batch_size, 
                total_cycles = total_cycles, cycle = cycle, 
                train_df = train_df, pred_df = pred_df, protein = protein, past_cycles = past_cycle_report,
                training_results_path = str(kwargs.get('training_results_path', '')) if kwargs.get('training_results_path') else "",
                simple_agent = self.simple_agent,
                disable_ucb = self.disable_ucb,
                disable_smarts = self.disable_smarts,
                disable_tanimoto = self.disable_tanimoto
            )
        
        try:
            indicies_per_strategy, summary, strategies = call_selector_agent()
        except Exception as e:
            raise RuntimeError(f"Failed to get selection results from LLM agent after retries: {str(e)}")
        
        if indicies_per_strategy is None or summary is None:
            raise RuntimeError("Failed to get selection results from LLM agent")
        
        # Validate that we have some non-empty strategies
        if not indicies_per_strategy or all(len(indices) == 0 for indices in indicies_per_strategy):
            self.logger.error("All selection strategies returned empty results")
            raise RuntimeError("All selection strategies returned empty results - unable to make selections")
        
        # Handle empty strategy results robustly
        affinity_per_strategy = [y_unlabeled[indicies] if len(indicies) > 0 else np.array([]) for indicies in indicies_per_strategy]
        n_selections_per_strategy = [len(indicies) for indicies in indicies_per_strategy]
        mean_per_strategy = [np.mean(affinity) if len(affinity) > 0 else 0.0 for affinity in affinity_per_strategy]
        max_per_strategy = [np.max(affinity) if len(affinity) > 0 else 0.0 for affinity in affinity_per_strategy]
        predictions_per_strategy = [predictions[indicies] if len(indicies) > 0 else np.array([]) for indicies in indicies_per_strategy]
        
        # Calculate RMSE with proper handling of empty arrays
        rmse_per_strategy = []
        for pred, affinity in zip(predictions_per_strategy, affinity_per_strategy):
            if len(pred) > 0 and len(affinity) > 0 and len(pred) == len(affinity):
                rmse_per_strategy.append(np.sqrt(np.mean((pred - affinity) ** 2)))
            else:
                rmse_per_strategy.append(np.nan)
        
        # Format string with proper NaN handling
        formatted_metrics = []
        for i, (strategy, n_selections, rmse, mean, max_val) in enumerate(zip(strategies, n_selections_per_strategy, rmse_per_strategy, mean_per_strategy, max_per_strategy)):
            rmse_str = "N/A" if np.isnan(rmse) else f"{rmse:.2f}"
            formatted_metrics.append(
                f"Strategy {i+1}: {strategy}\n"
                f"Number final selections: {n_selections}\n"
                f"RMSE: {rmse_str}\n"
                f"Mean: {mean:.2f}\n"
                f"Max: {max_val:.2f}\n"
            )
        
        zipped_string = '\n'.join(formatted_metrics)
        summary = f"{summary}\n\n{zipped_string}"

        final_indicies = [item for sublist in indicies_per_strategy for item in sublist]
        # Save summary to temporary file for AL.py to pick up
        training_results_path = kwargs.get('training_results_path')
        if training_results_path:
            summary_file = Path(training_results_path).parent / f"cycle_{cycle}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
        
        # Also save to past cycle report file for backward compatibility
        with open("alselectors/past_cycle_report.txt", "w") as f:
            f.write(summary)
        
        self.logger.info(f"LLM Agent Selection Summary (cycle {cycle}):\n{summary}")

        if len(set(final_indicies)) != len(final_indicies):
            self.logger.warning(f"Duplicate indices in final selection: {final_indicies}")
            final_indicies = list(set(final_indicies))
            self.logger.warning(f"Removed duplicates, new final selection: {final_indicies}")
        if len(final_indicies) != self.batch_size:
            self.logger.warning(f"Final selection size ({len(final_indicies)}) does not match batch size ({self.batch_size})")
            if len(final_indicies) > self.batch_size:
                final_indicies = final_indicies[:self.batch_size]
                self.logger.warning(f"Truncated final selection to match batch size: {final_indicies}")
        self.logger.info(f"Final selected indices (cycle {cycle}): {final_indicies}")
        return final_indicies

class LLMWorkflowSimpleSelector(LLMWorkflowSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        super().__init__(batch_size, random_seed, **kwargs)
        self.simple_agent = 1