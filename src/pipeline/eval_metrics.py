import numpy as np
from sklearn.metrics import roc_auc_score

def calc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates AUC for a single impression."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5

def calc_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates Mean Reciprocal Rank (MRR) for a single impression.
    Finds the rank of the first clicked article."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    # Sort ground truths by predicted score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    
    # Find rank (1-indexed) of first clicked item
    first_click_idx = np.where(y_true_sorted == 1)[0]
    if len(first_click_idx) == 0:
        return 0.0
    return 1.0 / (first_click_idx[0] + 1)

def calc_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain (nDCG@K) for a single impression."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
        
    order = np.argsort(-y_score)[:k]
    y_true_top_k = y_true[order]
    
    # DCG calculation
    discounts = np.log2(np.arange(2, len(y_true_top_k) + 2))
    dcg = np.sum((2 ** y_true_top_k - 1) / discounts)
    
    # Ideal DCG calculation
    ideal_y_true = np.sort(y_true)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_y_true) + 2))
    idcg = np.sum((2 ** ideal_y_true - 1) / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)
