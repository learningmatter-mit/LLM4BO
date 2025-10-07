import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

def evaluate_performance(predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    data: Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray],
    batch_size: int
) -> Dict[str, float]:
    """Evaluate model performance after an AL cycle.
    
    This function calculates various metrics to assess both the model's predictive
    performance and the effectiveness of the active learning selection process.
    
    Args:
        predictions: Tuple of (predictions, confidence_scores, train_predictions)
        data: Tuple of (X_train_features_dict, y_train, X_test_features_dict, y_test)
        batch_size: Number of samples acquired in last batch
        
    Returns:
        Dictionary containing:
            - recall_2pc: Recall for top 2% active compounds
            - recall_5pc: Recall for top 5% active compounds
            - f1_2pc: F1 score for top 2% active compounds
            - f1_5pc: F1 score for top 5% active compounds
            - rmse: Root mean squared error
            - r2: Coefficient of determination
            - spearman_rho: Spearman's rank correlation coefficient
    """
    predictions, confidence_scores, train_predictions = predictions
    X_train, y_train, X_test, y_test = data
    y_combined = np.concatenate([y_train, y_test])
    pred_combined = np.concatenate([train_predictions, predictions])
    # Calculate regression metrics
    rmse = np.sqrt(np.mean((y_combined - pred_combined) ** 2))
    r2 = stats.pearsonr(y_combined, pred_combined)[0] ** 2
    spearman_rho = stats.spearmanr(y_combined, pred_combined)[0]
    
    # Calculate top percentage
    n_total = len(y_combined)
    top_2p_threshold = np.percentile(y_combined, 98)  # Top 2%
    top_5p_threshold = np.percentile(y_combined, 95)  # Top 5%
    
    # Calculate true positives for each threshold
    tp_2p = np.sum((y_train >= top_2p_threshold))
    tp_5p = np.sum((y_train >= top_5p_threshold))
    
    # Calculate recall and F1 scores
    recall_2pc = tp_2p / (0.02 * n_total)
    recall_5pc = tp_5p / (0.05 * n_total)
    
    f1_2pc = 2 * tp_2p / (batch_size + 0.02 * n_total)
    f1_5pc = 2 * tp_5p / (batch_size + 0.05 * n_total)
    
    return {
        "recall_2pc": float(recall_2pc),
        "recall_5pc": float(recall_5pc),
        "f1_2pc": float(f1_2pc),
        "f1_5pc": float(f1_5pc),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman_rho": float(spearman_rho)
    }



