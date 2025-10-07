from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
import logging
class Selector(ABC):
    """Base class for all active learning selectors.
    
    Defines interface for selecting most informative samples for labeling.
    Has access to model training data to make informed selection decisions.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize selector with batch size and parameters.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Selector-specific parameters
        """
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.hyperparameters = kwargs
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select most informative samples for labeling.
        
        Args:
            predictions: Model predictions on unlabeled data
            confidence_scores: Model confidence scores
            training_data: Current labeled training set (features_dict, labels) for context
            unlabeled_data: Dictionary of unlabeled features, where keys are feature names and values are feature arrays.
                           Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            **kwargs: Selection-specific parameters
            
        Returns:
            list: Indices of selected samples for labeling
        """
        pass 