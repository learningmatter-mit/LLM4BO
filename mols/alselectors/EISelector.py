from typing import Any, List, Tuple, Dict
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm 

from utils.Selector import Selector


class EISelector(Selector):
    """Expected Improvement (EI) selection strategy for active learning.

    Assumes a Gaussian predictive distribution with mean given by `predictions`
    and standard deviation given by `confidence_scores`. Selects the
    `batch_size` candidates with highest EI with respect to the best observed
    training label.
    """

    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize EISelector.

        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
                - xi: Exploration parameter added to improvement threshold (default: 0.0)
                - min_std: Floor for standard deviation to avoid divide-by-zero (default: 1e-12)
        """
        super().__init__(batch_size, random_seed, **kwargs)
        self.xi = float(kwargs.get("xi", 0.0))
        self.min_std = float(kwargs.get("min_std", 1e-12))
        self.rng = default_rng(random_seed)

        self.logger.info(
            f"Initialized EISelector with batch_size={batch_size}, xi={self.xi}, min_std={self.min_std}"
        )

    def _normal_pdf(self, z: np.ndarray) -> np.ndarray:
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
        
    def _expected_improvement(
        self,
        means: np.ndarray,
        stddevs: np.ndarray,
        best_y: float,
    ) -> np.ndarray:
        """Compute the Expected Improvement for each candidate.

        EI(x) = (mu - best - xi) * Phi(z) + sigma * phi(z),
        where z = (mu - best - xi) / sigma and Phi/phi are standard normal CDF/PDF.
        If sigma == 0, EI = max(0, mu - best - xi).
        """
        improvement = means - best_y - self.xi
        safe_std = np.maximum(stddevs, self.min_std)
        z = improvement / safe_std

        phi = self._normal_pdf(z)
        Phi = norm.cdf(z)
        ei = improvement * Phi + safe_std * phi

        # Where the true stddev is zero, use the exact definition
        zero_std_mask = stddevs <= self.min_std
        if np.any(zero_std_mask):
            ei[zero_std_mask] = np.maximum(0.0, improvement[zero_std_mask])

        # Guard against NaNs/Infs
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any,
    ) -> List[int]:
        """Select samples with highest Expected Improvement.

        Args:
            predictions: Predictive means over unlabeled candidates (1D array)
            confidence_scores: Predictive standard deviations (1D array)
            training_data: Tuple of (features_dict, labels) for current labeled set
            unlabeled_data: Dictionary with at least 'SMILES' key
            random_seed: Random seed (unused; instance RNG is used)
            **kwargs: Unused

        Returns:
            List of selected sample indices of length `batch_size`.
        """
        if "SMILES" not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")

        n_samples = len(unlabeled_data["SMILES"])
        if n_samples < self.batch_size:
            self.logger.error(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )

        means = predictions.reshape(-1)
        uncertainty = (1.0 / confidence_scores) - 1.0
        stddevs = uncertainty.reshape(-1)
        
        if len(means) != n_samples or len(stddevs) != n_samples:
            self.logger.error(
                f"Shape mismatch: predictions ({len(means)}), stddevs ({len(stddevs)}), SMILES ({n_samples})"
            )
            raise ValueError(
                f"Shape mismatch: predictions ({len(means)}), stddevs ({len(stddevs)}), SMILES ({n_samples})"
            )

        # Determine best observed target from training labels
        _, labels = training_data
        if labels is None or len(labels) == 0:
            raise ValueError("Empty training labels")
        best_y = float(np.max(labels))

        self.logger.info(
            f"Starting EI selection from {n_samples} candidates with best_y={best_y:.6f}"
        )

        ei = self._expected_improvement(means, stddevs, best_y)

        # Select top-k by EI
        if self.batch_size >= ei.size:
            top_indices = np.arange(ei.size)
        else:
            top_indices = np.argsort(ei)[-self.batch_size:]
        selected_indices = top_indices.tolist()

        self.logger.info(f"Selected {len(selected_indices)} candidates")
        self.logger.debug(
            f"EI stats -> min: {float(np.min(ei)):.6e}, max: {float(np.max(ei)):.6e}, mean: {float(np.mean(ei)):.6e}"
        )
        self.logger.debug(f"Selected indices: {selected_indices}")

        return selected_indices


