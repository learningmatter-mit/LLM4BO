from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import logging
class Model(ABC):
    """Base class for all oracle models in active learning framework.
    
    Defines common interface for model training, fitting, and prediction.
    Handles input/output shape definitions and standardized model operations.
    """
    
    def __init__(self, random_seed: int, **kwargs: Any) -> None:
        """Initialize model with input/output dimensions.
        
        Args:
            random_seed: Random seed for reproducibility
            **kwargs: Model-specific hyperparameters
        """
        self.random_seed = random_seed
        self.hyperparameters = kwargs
        self.nbits =  2048
        self.logger = logging.getLogger(__name__)

    
    @abstractmethod
    def train(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray, random_seed: int, **kwargs: Any) -> Dict[str, Any]:
        """Train the model on labeled data.
        
        Args:
            X_train: Dictionary of training features, where keys are feature names and values are feature arrays.
                     Must contain 'SMILES' key with SMILES strings.
            y_train: Training labels
            random_seed: Random seed for reproducibility
            **kwargs: Training-specific parameters
            
        Returns:
            Training metrics and metadata
        """
        pass
    
    @abstractmethod
    def predict(self, X: Dict[str, np.ndarray], random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and confidence scores.
        
        Args:
            X: Dictionary of input features, where keys are feature names and values are feature arrays.
               Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
        Returns:
            tuple: (predictions, confidence_scores)
        """
        pass 