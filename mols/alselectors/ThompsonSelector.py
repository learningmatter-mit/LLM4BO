from typing import Any, List, Tuple, Dict
import numpy as np
import pandas as pd
import logging
import os
import tempfile
import torch
from utils.Selector import Selector

class ThompsonSelector(Selector):
    """
    Selects samples based on Thomsson sampling using low-rank covariance matrix approximation.
    
    Uses U and S components from SVD decomposition stored in tempfiles:
    - GPR_U_components.npy: Left singular vectors (n_samples x rank)
    - GPR_singular_values.npy: Singular values (rank,)
    
    Falls back to approximate uncertainty using variance of predictions if covariance matrix 
    is not available using var = 1/confidence_scores - 1
    
    Thomsson sampling works by:
    1. Sampling from the posterior distribution: z ~ N(μ, Σ)
    2. Using low-rank approximation: z = μ + U_k S_k^(1/2) ε, where ε ~ N(0, I)
    3. Selecting top-k samples based on sampled values
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize ThompsonSelector.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters including:
                - rank: Rank for low-rank approximation (default: 600)
        """
        super().__init__(batch_size, random_seed, **kwargs)
        
        # Thomsson sampling parameters
        self.rank = kwargs.get('rank', 600)
        
        # Set random seed for numpy
        np.random.seed(random_seed)
        
        # Log initialization
        self.logger.info(f"Initialized UncertaintySelector with batch_size={batch_size}")
    
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select samples using Thomsson sampling with low-rank covariance approximation.
        
        Args:
            predictions: Model predictions on unlabeled data
            confidence_scores: Model confidence scores
            training_data: Current labeled training set (features_dict, labels) (for context)
            unlabeled_data: Dictionary of unlabeled features, where keys are feature names and values are feature arrays.
                           Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            List of selected sample indices
        """
        if 'SMILES' not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")
        
        n_samples = len(unlabeled_data['SMILES'])
        if n_samples < self.batch_size:
            self.logger.error(f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}")
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
        
        self.logger.info(f"Starting uncertainty selection from {n_samples} candidates")
        # Validate input shapes
        if len(predictions) != n_samples:
            raise ValueError(f"Predictions shape {len(predictions)} doesn't match unlabeled data size {n_samples}")
        if len(confidence_scores) != n_samples:
            raise ValueError(f"Confidence scores shape {len(confidence_scores)} doesn't match unlabeled data size {n_samples}")
        
        try:
            selected_indices = self._torch_thompson_sampling(predictions)
        except Exception as e:
            self.logger.warning(f"Thomsson sampling failed: {e}")
            raise e
        
        self.logger.info(f"Selected {len(selected_indices)} candidates")
        self.logger.debug(f"Selected indices: {selected_indices}")
        
        return selected_indices
    
    def _torch_thompson_sampling(self, predictions,  n_function_samples=100):
        """
        GPU-accelerated Thompson sampling using PyTorch
        """
        tmp_dir = tempfile.gettempdir()
        U_path = os.path.join(tmp_dir, "GPR_U_components.npy")
        s_path = os.path.join(tmp_dir, "GPR_singular_values.npy")
        
        if not os.path.exists(U_path) or not os.path.exists(s_path):
            raise FileNotFoundError("Low-rank covariance components not found. Run GPRegOracle.predict() first.")
        
        try:
            U = np.load(U_path)  # Shape: (n_samples, rank)
            s = np.load(s_path)  # Shape: (rank,)
        except Exception as e:
            raise RuntimeError(f"Failed to load covariance components: {e}")
        
        # Validate shapes
        if U.shape[0] != len(predictions):
            raise ValueError(f"U matrix shape {U.shape[0]} doesn't match predictions length {len(predictions)}")
        # Convert to torch tensors and move to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        mean_tensor = torch.from_numpy(predictions).float().to(device)
        U_tensor = torch.from_numpy(U).float().to(device)
        s_tensor = torch.from_numpy(s).float().to(device)
        
        rank = U_tensor.shape[1]
        n_candidates = len(predictions)
        
        self.logger.info(f"Using GPU Thompson sampling on {device} with rank {rank}")
        
        # Sample all epsilon at once: Shape (n_function_samples, rank)
        epsilon = torch.randn(n_function_samples, rank, device=device)
        
        # Apply scaling: Shape (n_function_samples, rank)  
        s_sqrt_epsilon = epsilon * torch.sqrt(s_tensor).unsqueeze(0)
        
        # Transform to candidate space: Shape (n_function_samples, n_candidates)
        # function_samples = mean + (s_sqrt_epsilon @ U.T)
        function_samples = mean_tensor.unsqueeze(0) + torch.mm(s_sqrt_epsilon, U_tensor.T)
        
        # Find top batch_size indices for each function sample
        # topk returns (values, indices)
        _, top_indices = torch.topk(function_samples, self.batch_size, dim=1)  # Shape: (n_function_samples, batch_size)
        
        # Count selection frequencies using scatter_add
        selection_scores = torch.zeros(n_candidates, device=device)
        ones = torch.ones_like(top_indices, dtype=torch.float, device=device)
        
        # Flatten indices and ones for scatter_add
        flat_indices = top_indices.flatten()
        flat_ones = ones.flatten()
        
        selection_scores.scatter_add_(0, flat_indices, flat_ones)
        
        # Select final candidates
        _, selected_indices = torch.topk(selection_scores, self.batch_size)
        
        # Convert back to numpy
        selected_indices_np = selected_indices.cpu().numpy()
        selection_scores_np = selection_scores.cpu().numpy()
        
        self.logger.info(f"Thompson sampling completed. Selection frequency range: [{selection_scores_np.min():.1f}, {selection_scores_np.max():.1f}]")
        
        return selected_indices_np.tolist()
