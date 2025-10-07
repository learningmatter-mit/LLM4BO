"""
This oracle uses a Gaussian Process Regressor to predict the target values of the unlabeled data.
Tanimoto similarity kernel
Based on GPyTorch
Molecules featurised using Morgan fingerprints with chirality with radius 4 and 4096 bits from RDKit
"""

from typing import Any, Dict, Tuple
import numpy as np
from numpy.linalg import svd
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.Model import Model
from utils.SMILESUtils import smiles_to_fingerprint

import os
import tempfile

class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto similarity kernel for molecular fingerprints."""
    
    def __init__(self, random_seed: int, **kwargs):
        super().__init__(**kwargs)
        self.random_seed = random_seed
    
    def forward(self, x1, x2, diag=False, **params):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if diag:
            return torch.ones(x1.size(0), dtype=x1.dtype, device=x1.device)
        
        # Convert to binary (assuming fingerprints are already 0/1)
        x1_bin = (x1 > 0.5).float()
        x2_bin = (x2 > 0.5).float()
        
        # Calculate intersection (bitwise AND)
        intersection = torch.mm(x1_bin, x2_bin.t())
        
        # Calculate union (bitwise OR)
        x1_sum = x1_bin.sum(dim=1, keepdim=True)
        x2_sum = x2_bin.sum(dim=1, keepdim=True)
        union = x1_sum + x2_sum.t() - intersection
        
        # Tanimoto similarity = intersection / union
        # Add small epsilon to avoid division by zero
        similarity = intersection / (union + 1e-8)
        
        return similarity

class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model for molecular property prediction."""
    
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, random_seed: int) -> None:
        """Initialize GP model.
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            likelihood: GP likelihood
            random_seed: Random seed for reproducibility
        """
        super().__init__(train_x, train_y, likelihood)
        
        # Initialize mean module
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Initialize Tanimoto kernel for molecular fingerprints
        self.covar_module = ScaleKernel(TanimotoKernel(random_seed))
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegOracle(Model):
    """Gaussian Process Regression based oracle model for molecular property prediction.
    
    Uses RDKit to convert SMILES to Morgan fingerprints and GPyTorch for property prediction.
    Provides uncertainty estimates through the GP posterior variance.
    """
    
    def __init__(self, random_seed: int, **kwargs: Any) -> None:
        """Initialize GPRegOracle.
        
        Args:
            **kwargs: Additional GP parameters including:
                - learning_rate: Learning rate for optimizer
                - n_epochs: Number of training epochs
                - batch_size: Batch size for training
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed, **kwargs)
        
        # Initialize model components
        self.device = torch.device("cpu")
        
        # Training parameters - more conservative defaults
        self.learning_rate = kwargs.get('learning_rate', 0.001) 
        self.n_epochs = kwargs.get('n_epochs', 500)     
        self.get_covariance = kwargs.get('get_covariance', False)

        # Set fingerprint parameters
        self.nbits = 4096
        self.radius = 4
        
        # Initialize likelihood and model (will be set in train)
        self.likelihood = None
        self.model = None
        
        # Store training data for proper GP evaluation
        self.X_train_tensor = None
        self.y_train_tensor = None
    
    def train(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray, random_seed: int, **kwargs: Any) -> Dict[str, Any]:
        """Train the Gaussian Process model.
        
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
        
        print(f"Training GP with {len(X_train['SMILES'])} samples")
        print(f"Target range: {np.min(y_train):.3f} to {np.max(y_train):.3f}")
        print(f"X_train keys: {X_train.keys()}")
        print(f"X_train['fingerprint'] shape: {X_train['fingerprint'].shape}")
        # Convert SMILES to fingerprints
        X_fp = X_train['fingerprint']
        
        # Convert to PyTorch tensors
        self.X_train_tensor = torch.FloatTensor(X_fp).to(self.device)
        self.y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Initialize likelihood and model
        self.likelihood = GaussianLikelihood().to(self.device)
        
        # Initialize likelihood noise to reasonable value
        self.likelihood.noise = 0.01
        
        self.model = ExactGPModel(self.X_train_tensor, self.y_train_tensor, self.likelihood, random_seed).to(self.device)
        
        # Set model to training mode
        self.model.train()
        self.likelihood.train()
        
        # Initialize optimizer with different parameter groups
        optimizer = torch.optim.Adam([
            {'params': self.model.mean_module.parameters(), 'lr': self.learning_rate},
            {'params': self.model.covar_module.parameters(), 'lr': self.learning_rate},
            {'params': self.likelihood.parameters(), 'lr': self.learning_rate * 0.1}
        ])
        
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop with early stopping
        train_losses = []
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(self.X_train_tensor)
            loss = -mll(output, self.y_train_tensor)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, stopping training")
                break
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
        
        # Calculate final metrics
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions on training data
            pred_dist = self.likelihood(self.model(self.X_train_tensor))
            y_pred = pred_dist.mean.cpu().numpy()
            
            mse = np.mean((y_train - y_pred) ** 2)
            mae = np.mean(np.abs(y_train - y_pred))
            
            # Calculate R² properly
            ss_res = np.sum((y_train - y_pred) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"Training MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            print(f"Prediction range: {np.min(y_pred):.3f} to {np.max(y_pred):.3f}")
            print(f"Prediction std: {np.std(y_pred):.3f}")
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "final_loss": float(train_losses[-1]) if train_losses else 0.0,
            "n_epochs_trained": len(train_losses)
        }
    
    def predict(self, X: Dict[str, np.ndarray], random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and confidence scores.
        
        Args:
            X: Dictionary of input features, where keys are feature names and values are feature arrays.
               Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
        Returns:
            Tuple of (predictions, confidence_scores)
            Confidence scores are based on GP posterior variance
        """

        if self.model is None or self.likelihood is None:
            raise ValueError("Model not trained yet!")

        if 'SMILES' not in X:
            raise ValueError("X must contain 'SMILES' key")

        # Convert SMILES to fingerprints
        X_fp = X['fingerprint']

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_fp).to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            pred_dist = self.likelihood(self.model(X_tensor))
            
            # Extract mean and covariance matrix
            mean = pred_dist.mean.cpu().numpy()
            print(f"mean shape: {mean.shape}")
            
            # Get full covariance matrix
            # Note: this can be memory intensive for large X
            if self.hyperparameters.get('selector_name') == 'ThompsonSelector':
                print("Computing covariance matrix...")
                try:
                    # Delete covariance matrix from tempfile if it exists
                    tmp_dir = tempfile.gettempdir()
                    U_path = os.path.join(tmp_dir, "GPR_U_components.npy")
                    s_path = os.path.join(tmp_dir, "GPR_singular_values.npy")
                    if os.path.exists(U_path):
                        os.remove(U_path)
                    if os.path.exists(s_path):
                        os.remove(s_path)

                    covariance_matrix = pred_dist.covariance_matrix.cpu().numpy()
                    # Perform SVD decomposition for low-rank approximation
                    rank = min(600, covariance_matrix.shape[0])  # Adaptive rank selection
                    print(f"Computing rank-{rank} approximation...")
                    
                    U, s, Vt = svd(covariance_matrix, full_matrices=False)
                    
                    # Keep top 'rank' components
                    U_truncated = U[:, :rank]  # Shape: (n_samples, rank)
                    s_truncated = s[:rank]     # Shape: (rank,)
                    
                    # Compute explained variance ratio
                    explained_variance = np.sum(s_truncated) / np.sum(s)
                    print(f"Rank-{rank} approximation captures {explained_variance:.2%} of variance")
                    print(f"U_truncated.shape: {U_truncated.shape}, s_truncated.shape: {s_truncated.shape}")
                    
                    # Create temp directory and save low-rank components
                    tmp_dir = tempfile.gettempdir()
                    np.save(U_path, U_truncated)
                    np.save(s_path, s_truncated)
                    
                except RuntimeError as e:
                    print(f"Warning: Could not compute full covariance matrix: {e}")
                    print("Falling back to diagonal variance only...")
                    
            variance = pred_dist.variance.cpu().numpy()

            # Convert variance to confidence scores (inverse uncertainty)
            # Add small epsilon to avoid division by zero
            confidence = 1.0 / (1.0 + variance + 1e-8)
            
            print(f"Prediction stats - Mean: {np.mean(mean):.3f}, Std: {np.std(mean):.3f}")
            print(f"Confidence stats - Mean: {np.mean(confidence):.3f}, Std: {np.std(confidence):.3f}")
            
        return mean, confidence