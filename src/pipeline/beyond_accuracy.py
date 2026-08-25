import numpy as np
import polars as pl
from typing import Dict, List, Set

def calc_intra_list_diversity(recommended_aids: List[str], article_embeddings: Dict[str, np.ndarray]) -> float:
    """
    Calculates Intra-List Diversity (ILD) for a list of recommended articles.
    ILD is the average pairwise cosine distance (1 - cosine_sim) between top recommendations.
    """
    vecs = [article_embeddings[aid] for aid in recommended_aids if aid in article_embeddings]
    if len(vecs) < 2:
        return 0.0
        
    matrix = np.vstack(vecs)
    # Ensure L2 normalized
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    
    # Cosine similarity matrix
    sim_matrix = matrix @ matrix.T
    
    # Pairwise distances (1 - similarity) for upper triangle excluding diagonal
    n = len(vecs)
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sim_matrix[iu]
    
    return float(np.mean(distances))

def calc_novelty(recommended_aids: List[str], article_popularities: Dict[str, float]) -> float:
    """
    Calculates Novelty as the average self-information (-log2(p(a))) of recommendations.
    p(a) is the relative popularity of article 'a' in the training set.
    Higher values indicate more novel (less mainstream) recommendations.
    """
    if not recommended_aids:
        return 0.0
        
    self_info_list = []
    for aid in recommended_aids:
        p = article_popularities.get(aid, 1e-6)
        p = max(p, 1e-6)
        self_info = -np.log2(p)
        self_info_list.append(self_info)
        
    return float(np.mean(self_info_list))

def calc_catalog_coverage(all_recommendations: List[List[str]], total_catalog_size: int) -> float:
    """
    Calculates Catalog Coverage: fraction of total catalog articles that were 
    recommended at least once across all users.
    """
    if total_catalog_size == 0:
        return 0.0
        
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
        
    return float(len(unique_recommended) / total_catalog_size)
