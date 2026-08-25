import numpy as np
import pytest
from src.pipeline.eval_metrics import calc_auc, calc_mrr, calc_ndcg
from src.pipeline.beyond_accuracy import calc_intra_list_diversity, calc_novelty, calc_catalog_coverage

def test_calc_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert calc_auc(y_true, y_score) == 1.0

def test_calc_mrr_first_place():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.1, 0.2, 0.3])
    # Clicked item has rank 1 -> MRR = 1/1 = 1.0
    assert calc_mrr(y_true, y_score) == 1.0

def test_calc_mrr_second_place():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.5, 0.9, 0.2, 0.3])
    # Clicked item (index 0) has score 0.5, second highest after index 1 -> rank 2 -> MRR = 1/2 = 0.5
    assert calc_mrr(y_true, y_score) == 0.5

def test_calc_ndcg_perfect():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.1, 0.2, 0.3])
    assert calc_ndcg(y_true, y_score, k=5) == 1.0

def test_calc_beyond_accuracy():
    embeddings = {
        "A1": np.array([1.0, 0.0]),
        "A2": np.array([0.0, 1.0])
    }
    ild = calc_intra_list_diversity(["A1", "A2"], embeddings)
    # Cosine sim is 0.0, distance is 1.0
    assert abs(ild - 1.0) < 1e-5

    popularities = {"A1": 0.5, "A2": 0.25}
    novelty = calc_novelty(["A1", "A2"], popularities)
    # -log2(0.5) = 1.0, -log2(0.25) = 2.0 -> mean = 1.5
    assert abs(novelty - 1.5) < 1e-5

    coverage = calc_catalog_coverage([["A1"], ["A2"]], total_catalog_size=10)
    assert coverage == 0.2
