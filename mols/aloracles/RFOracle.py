from typing import Any, Dict, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils.Model import Model
from utils.SMILESUtils import smiles_to_fingerprint

class RFOracle(Model):
    """Random Forest based oracle model for molecular property prediction.
    
    Uses RDKit to convert SMILES to Morgan fingerprints and RandomForestRegressor
    for property prediction. Supports both regression and classification tasks.
    """
    
    def __init__(self, random_seed: int, **kwargs: Any) -> None:
        """Initialize RFOracle.
        
        Args:
            **kwargs: Additional RandomForest parameters
        """
        super().__init__(random_seed, **kwargs)
        
        # Initialize model components
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=random_seed,
            n_jobs=kwargs.get('n_jobs', -1)
        )
        self.nbits = 4096
        self.radius = 4
    
    def train(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray, random_seed: int = 42) -> Dict[str, Any]:
        """Train the RF model.
        
        Args:
            X_train: Dictionary of training features, where keys are feature names and values are feature arrays.
                     Must contain 'SMILES' key with SMILES strings.
            y_train: Array of target values for training
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics
        """
        if 'SMILES' not in X_train:
            raise ValueError("X_train must contain 'SMILES' key")
        
        # Convert SMILES to fingerprints
        X_fp = np.array([smiles_to_fingerprint(smiles, self.radius, self.nbits) for smiles in X_train['SMILES']])
        
        
        # Train model
        self.model.fit(X_fp, y_train)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_fp)
        mse = np.mean((y_train - y_pred) ** 2)
        r2 = self.model.score(X_fp, y_train)
        
        return {
            "mse": float(mse),
            "r2": float(r2),
            "n_trees": len(self.model.estimators_)
        }
    
    def predict(self, X: Dict[str, np.ndarray], random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and confidence scores.
        
        Args:
            X: Dictionary of input features, where keys are feature names and values are feature arrays.
               Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if 'SMILES' not in X:
            raise ValueError("X must contain 'SMILES' key")
        
        # Convert SMILES to fingerprints
        X_fp = np.array([smiles_to_fingerprint(smiles, self.radius, self.nbits) for smiles in X['SMILES']])
            
        # Get predictions from all trees
        predictions = np.array([tree.predict(X_fp) for tree in self.model.estimators_])
        
        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        # Confidence score based on variance across trees
        confidence = 1.0 / (1.0 + np.var(predictions, axis=0))
        
        return mean_pred, confidence 