from typing import Any, List, Tuple, Dict
import numpy as np
from numpy.random import default_rng
import logging

from utils.Selector import Selector

class RandomSelector(Selector):
    """Random selection strategy for active learning.
    
    Selects samples uniformly at random from the unlabeled pool.
    Useful as a baseline for comparing other selection strategies.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize RandomSelector.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters (not used in random selection)
        """
        super().__init__(batch_size, random_seed, **kwargs)
        self.rng = default_rng(random_seed)
        
        # Log initialization
        self.logger.info(f"Initialized RandomSelector with batch_size={batch_size}")
    
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select samples randomly from unlabeled pool.
        
        Args:
            predictions: Model predictions (not used in random selection)
            confidence_scores: Model confidence scores (not used in random selection)
            training_data: Current training set (features_dict, labels) (not used in random selection)
            unlabeled_data: Dictionary of unlabeled features, where keys are feature names and values are feature arrays.
                           Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters (not used in random selection)
            
        Returns:
            List of randomly selected sample indices
        """
        if 'SMILES' not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")
        
        n_samples = len(unlabeled_data['SMILES'])
        if n_samples < self.batch_size:
            self.logger.error(f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}")
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
        
        self.logger.info(f"Starting random selection from {n_samples} candidates")
        
        # Select random indices
        selected_indices = self.rng.choice(
            n_samples,
            size=self.batch_size,
            replace=False
        ).tolist()
        
        self.logger.info(f"Selected {len(selected_indices)} candidates")
        self.logger.debug(f"Selected indices: {selected_indices}")
        
        return selected_indices 