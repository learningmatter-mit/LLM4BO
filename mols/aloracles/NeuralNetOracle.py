from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from utils.Model import Model
from utils.SMILESUtils import smiles_to_fingerprint

class MolecularNN(nn.Module):
    """Neural network for molecular property prediction."""
    
    def __init__(self, input_dim: int = 4096, hidden_dims: list = [512, 256, 128]) -> None:
        """Initialize neural network architecture.
        
        Args:
            input_dim: Input dimension (Morgan fingerprint size + additional features)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.nbits = 4096
        self.radius = 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

class NeuralNetOracle(Model):
    """Neural Network based oracle model for molecular property prediction.
    
    Uses RDKit to convert SMILES to Morgan fingerprints and a neural network
    for property prediction. Supports both regression and classification tasks.
    """
    
    def __init__(self, random_seed: int = 42, **kwargs: Any) -> None:
        """Initialize NeuralNetOracle.
        
        Args:
            **kwargs: Additional neural network parameters
        """
        super().__init__(random_seed=random_seed, **kwargs)
        
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_seed)
        
        # Set fingerprint parameters
        self.nbits = 4096
        self.radius = 4
        
        # Calculate input dimension (will be set during first training)
        self.input_dim = 4096  # Default Morgan fingerprint size
        self.model = None  # Will be initialized during first training
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def _initialize_model(self, input_dim: int):
        """Initialize the model with the correct input dimension."""
        if self.model is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            self.model = MolecularNN(
                input_dim=input_dim,
                hidden_dims=self.hyperparameters.get('hidden_dims', [512, 256, 128])
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hyperparameters.get('learning_rate', 0.001)
            )
        
    def train(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray, random_seed: int = 42, **kwargs: Any) -> Dict[str, Any]:
        """Train the neural network model.
        
        Args:
            X_train: Dictionary of training features, where keys are feature names and values are feature arrays.
                     Must contain 'SMILES' key with SMILES strings.
            y_train: Array of target values
            random_seed: Random seed for reproducibility
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics
        """
        if 'SMILES' not in X_train:
            raise ValueError("X_train must contain 'SMILES' key")
        
        # Convert SMILES to fingerprints
        X_fp = np.array([smiles_to_fingerprint(smiles, self.radius, self.nbits) for smiles in X_train['SMILES']])
        
        # Initialize model with correct input dimension 
        self._initialize_model(X_fp.shape[1])
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_fp).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device).view(-1, 1)
        
        # Training loop
        self.model.train()
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        
        train_losses = []
        for epoch in range(n_epochs):
            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i + batch_size]
                batch_y = y_tensor[i:i + batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
            mse = np.mean((y_train - y_pred) ** 2)
            r2 = 1 - mse / np.var(y_train)
        
        return {
            "mse": float(mse),
            "r2": float(r2),
            "final_loss": float(train_losses[-1])
        }
    
    def predict(self, X: Dict[str, np.ndarray], random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and confidence scores.
        
        Args:
            X: Dictionary of input features, where keys are feature names and values are feature arrays.
               Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (predictions, confidence_scores)
            Confidence scores are based on model uncertainty
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if 'SMILES' not in X:
            raise ValueError("X must contain 'SMILES' key")
        
        # Convert SMILES to fingerprints
        X_fp = np.array([smiles_to_fingerprint(smiles, self.radius, self.nbits) for smiles in X['SMILES']])
        
        # Add additional features if available
        additional_features = []
        for feature_name, feature_values in X.items():
            if feature_name != 'SMILES':
                additional_features.append(feature_values)
        
        if additional_features:
            X_fp = np.column_stack([X_fp] + additional_features)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_fp).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            # Get base predictions
            predictions = self.model(X_tensor).cpu().numpy().flatten()
            
            # Use dropout at inference for uncertainty estimation
            self.model.train()  # Enable dropout
            n_samples = 10
            pred_samples = []
            
            # Generate multiple predictions with dropout
            for _ in range(n_samples):
                pred = self.model(X_tensor).cpu().numpy().flatten()
                pred_samples.append(pred)
            
            # Calculate mean and variance
            pred_samples = np.array(pred_samples)  # Shape: (n_samples, n_predictions)
            mean_pred = np.mean(pred_samples, axis=0)  # Shape: (n_predictions,)
            variance = np.var(pred_samples, axis=0)  # Shape: (n_predictions,)
            
            # Convert variance to confidence scores (higher variance = lower confidence)
            confidence = 1.0 / (1.0 + variance)
            
            # Ensure predictions and confidence have same shape
            predictions = mean_pred
            confidence = confidence.reshape(-1)  # Ensure 1D array
        
        return predictions, confidence 