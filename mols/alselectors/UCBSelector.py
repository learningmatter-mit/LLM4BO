from typing import Any, List, Tuple, Dict
import numpy as np
import logging

from utils.Selector import Selector

class UCBSelector(Selector):
    """Upper Confidence Bound (UCB) selector for active learning.
    
    Selects samples that maximize UCB(x) = μ(x) + β × σ(x), where:
    - μ(x) is the predicted mean (exploitation)
    - σ(x) is the predicted uncertainty (exploration)
    - β is the exploration-exploitation trade-off parameter
    
    UCB provides a principled way to balance exploitation of high-value predictions
    with exploration of uncertain regions. Higher β values favor exploration,
    while lower β values favor exploitation.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize UCBSelector.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters including:
                - beta: Exploration-exploitation trade-off parameter (default: 1.0)
                - confidence_threshold: Minimum confidence threshold (default: 0.0)
                - uncertainty_method: Method to compute uncertainty from confidence scores
                  Options: 'inverse', 'sqrt_inverse', 'log_inverse' (default: 'inverse')
        """
        super().__init__(batch_size, random_seed, **kwargs)
        
        # UCB parameters
        self.beta = kwargs.get('beta', 4.0)
        print(f"Beta: {self.beta}")
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.0)
        self.uncertainty_method = kwargs.get('uncertainty_method', 'inverse')
        
        # Validate uncertainty method
        valid_methods = ['inverse', 'sqrt_inverse', 'log_inverse']
        if self.uncertainty_method not in valid_methods:
            raise ValueError(f"uncertainty_method must be one of {valid_methods}")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Log initialization
        self.logger.info(f"Initialized UCBSelector with batch_size={batch_size}")
        self.logger.info(f"Beta (exploration parameter): {self.beta}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Uncertainty method: {self.uncertainty_method}")
    
    def _compute_uncertainty(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Convert confidence scores to uncertainty estimates.
        
        Args:
            confidence_scores: Model confidence scores (higher = more confident)
            
        Returns:
            Array of uncertainty estimates (higher = more uncertain)
        """
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-8
        confidence_scores = np.clip(confidence_scores, epsilon, 1.0 - epsilon)
        
        if self.uncertainty_method == 'inverse':
            # σ = 1/confidence - 1
            uncertainty = (1.0 / confidence_scores) - 1.0
        elif self.uncertainty_method == 'sqrt_inverse':
            # σ = sqrt(1/confidence - 1)
            uncertainty = np.sqrt((1.0 / confidence_scores) - 1.0)
        elif self.uncertainty_method == 'log_inverse':
            # σ = -log(confidence)
            uncertainty = -np.log(confidence_scores)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
        
        return uncertainty
    
    def _compute_ucb_scores(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray
    ) -> np.ndarray:
        """Compute UCB scores for each sample.
        
        Args:
            predictions: Model predictions (1D array)
            confidence_scores: Model confidence scores (1D array)
            
        Returns:
            Array of UCB scores: UCB(x) = μ(x) + β × σ(x)
        """
        # Ensure inputs are 1D arrays
        predictions = predictions.reshape(-1)
        confidence_scores = confidence_scores.reshape(-1)
        
        # Compute uncertainty from confidence scores
        uncertainty = self._compute_uncertainty(confidence_scores)
        
        # Compute UCB scores
        ucb_scores = predictions + self.beta**0.5 * uncertainty
        
        self.logger.debug(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        self.logger.debug(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
        self.logger.debug(f"UCB score range: [{ucb_scores.min():.3f}, {ucb_scores.max():.3f}]")
        
        return ucb_scores
    
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select samples with highest UCB scores.
        
        Args:
            predictions: Model predictions (1D array)
            confidence_scores: Model confidence scores (1D array)
            training_data: Current training set (features_dict, labels) (for context)
            unlabeled_data: Dictionary of unlabeled features, where keys are feature names and values are feature arrays.
                           Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            List of selected sample indices
        """
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Ensure inputs are 1D arrays
        predictions = predictions.reshape(-1)
        confidence_scores = confidence_scores.reshape(-1)
        
        if 'SMILES' not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")
        
        n_samples = len(unlabeled_data['SMILES'])
        if n_samples < self.batch_size:
            self.logger.error(f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}")
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
        
        # Validate input shapes
        if len(predictions) != n_samples or len(confidence_scores) != n_samples:
            self.logger.error(
                f"Shape mismatch: predictions ({len(predictions)}), "
                f"confidence_scores ({len(confidence_scores)}), "
                f"unlabeled_data SMILES ({n_samples})"
            )
            raise ValueError(
                f"Shape mismatch: predictions ({len(predictions)}), "
                f"confidence_scores ({len(confidence_scores)}), "
                f"unlabeled_data SMILES ({n_samples})"
            )
        
        self.logger.info(f"Starting UCB selection from {n_samples} candidates")
        
        # Compute UCB scores
        ucb_scores = self._compute_ucb_scores(predictions, confidence_scores)
        
        # Apply confidence threshold if specified
        if self.confidence_threshold > 0:
            mask = confidence_scores >= self.confidence_threshold
            n_valid = np.sum(mask)
            self.logger.info(f"Applied confidence threshold {self.confidence_threshold}: {n_valid} valid samples")
            
            if n_valid < self.batch_size:
                self.logger.error(
                    f"Not enough samples ({n_valid}) meet confidence threshold "
                    f"{self.confidence_threshold} for batch size {self.batch_size}"
                )
                raise ValueError(
                    f"Not enough samples ({n_valid}) meet confidence threshold "
                    f"{self.confidence_threshold} for batch size {self.batch_size}"
                )
            ucb_scores = ucb_scores[mask]
            valid_indices = np.where(mask)[0]
        else:
            valid_indices = np.arange(len(ucb_scores))
        
        # Select indices with highest UCB scores
        top_indices = np.argsort(ucb_scores)[-self.batch_size:]
        selected_indices = valid_indices[top_indices].tolist()
        
        self.logger.info(f"Selected {len(selected_indices)} candidates")
        self.logger.debug(f"Selected indices: {selected_indices}")
        self.logger.debug(f"Selected UCB scores range: {ucb_scores[top_indices].min():.3f} to {ucb_scores[top_indices].max():.3f}")
        
        # Log exploration vs exploitation contribution
        selected_predictions = predictions[selected_indices]
        selected_confidence = confidence_scores[selected_indices]
        selected_uncertainty = self._compute_uncertainty(selected_confidence)
        
        exploitation_contribution = np.mean(selected_predictions)
        exploration_contribution = np.mean(self.beta * selected_uncertainty)
        
        self.logger.info(f"Mean exploitation term: {exploitation_contribution:.3f}")
        self.logger.info(f"Mean exploration term: {exploration_contribution:.3f}")
        self.logger.info(f"Exploration/Exploitation ratio: {exploration_contribution/max(exploitation_contribution, 1e-8):.3f}")
        
        return selected_indices
