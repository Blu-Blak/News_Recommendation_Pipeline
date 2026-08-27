import argparse
from pathlib import Path
import polars as pl
import numpy as np
import multiprocessing
import sys
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.bm25_retriever import BM25Retriever
from src.pipeline.semantic_retriever import SemanticRetriever
from src.pipeline.eval_metrics import calc_auc, calc_mrr, calc_ndcg
from src.pipeline.beyond_accuracy import calc_intra_list_diversity, calc_novelty, calc_catalog_coverage

def compute_bootstrap_ci(metric_values: np.ndarray, n_bootstraps: int = 1000, ci: float = 95.0) -> tuple[float, float, float]:
    """
    Computes mean and bootstrap 95% confidence interval [lower, upper].
    """
    if len(metric_values) == 0:
        return 0.0, 0.0, 0.0
        
    mean_val = float(np.mean(metric_values))
    boot_means = np.empty(n_bootstraps)
    n = len(metric_values)
    
    # Vectorized bootstrap sampling
    sample_indices = np.random.randint(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(metric_values[sample_indices], axis=1)
    
    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    
    return mean_val, lower, upper

def evaluate_harness(dataset: str, retriever_type: str, limit: int = 2000):
    print(f"\n=======================================================")
    print(f"  RUNNING FULL EVALUATION HARNESS: {dataset.upper()} ({retriever_type.upper()})")
    print(f"=======================================================")
    
    if dataset == "mind":
        proc_dir = Path("data/processed/mind")
    else:
        proc_dir = Path("data/processed/ebnerd_demo")
        
    val_path = proc_dir / "val.parquet"
    articles_path = proc_dir / "articles.parquet"
    train_path = proc_dir / "train.parquet"
    
    if not val_path.exists() or not articles_path.exists():
        print(f"Error: {val_path} or {articles_path} not found. Run `make data` first.")
        return None
        
    print(f"Loading articles and validation set...")
    df_articles = pl.read_parquet(articles_path)
    df_val = pl.read_parquet(val_path)
    
    if limit and limit < df_val.height:
        df_val = df_val.head(limit)
        
    print(f"Initializing {retriever_type} retriever...")
    if retriever_type == "bm25":
        retriever = BM25Retriever(df_articles)
    else:
        retriever = SemanticRetriever(df_articles)
        
    # Load article embeddings independently for ILD computation (works for all retrievers)
    article_embeddings = {}
    emb_col = "bert" if "bert" in df_articles.columns else None
    if emb_col:
        print("Loading article embeddings for ILD computation...")
        for row in df_articles.iter_rows(named=True):
            emb = row.get(emb_col)
            if emb is not None:
                article_embeddings[row["article_id"]] = np.array(emb, dtype=np.float32)
    else:
        print("Warning: No embedding column found in articles. ILD will be 0.0.")
        
    # Pre-calculate article popularities from train set for Novelty metric
    print("Calculating article popularities for Novelty metric...")
    article_pops = {}
    if train_path.exists():
        df_train = pl.read_parquet(train_path)
        if "article_ids_clicked" in df_train.columns:
            exploded = df_train.select("article_ids_clicked").explode("article_ids_clicked")
            counts = exploded.group_by("article_ids_clicked").len()
            total_clicks = df_train.height
            for row in counts.iter_rows():
                if row[0]:
                    article_pops[row[0]] = row[1] / max(1, total_clicks)

    # Process evaluation rows
    print(f"Evaluating {df_val.height} impressions...")
    
    auc_list = []
    mrr_list = []
    ndcg5_list = []
    ndcg10_list = []
    ild_list = []
    novelty_list = []
    all_recommendations = []
    
    # Slicing storage
    cold_aucs, warm_aucs = [], []
    
    hist_col = "article_id_fixed" if "article_id_fixed" in df_val.columns else "history_article_id"
    histories = df_val[hist_col].to_list() if hist_col in df_val.columns else [[]] * df_val.height
    inviews = df_val["article_ids_inview"].to_list()
    clickeds = df_val["article_ids_clicked"].to_list()
    
    for history, candidates, clicked in zip(histories, inviews, tqdm(clickeds, desc="Evaluating Impressions")):
        if history is None:
            history = []
        if candidates is None or not candidates or clicked is None or not clicked:
            continue
            
        # Ground truth labels
        y_true = np.array([1 if c in clicked else 0 for c in candidates])
        if np.sum(y_true) == 0:
            continue
            
        # Score candidates
        scores = retriever.score_candidates(history, candidates)
        scores_arr = np.array(scores)
        
        # Ranking metrics
        auc = calc_auc(y_true, scores_arr)
        mrr = calc_mrr(y_true, scores_arr)
        ndcg5 = calc_ndcg(y_true, scores_arr, k=5)
        ndcg10 = calc_ndcg(y_true, scores_arr, k=10)
        
        auc_list.append(auc)
        mrr_list.append(mrr)
        ndcg5_list.append(ndcg5)
        ndcg10_list.append(ndcg10)
        
        # Top-10 recommendations for beyond-accuracy metrics
        top10_idx = np.argsort(-scores_arr)[:10]
        top10_recs = [candidates[i] for i in top10_idx]
        all_recommendations.append(top10_recs)
        
        # Beyond-accuracy metrics (ILD uses standalone embeddings, works for all retrievers)
        if article_embeddings:
            ild = calc_intra_list_diversity(top10_recs, article_embeddings)
            ild_list.append(ild)
            
        novelty = calc_novelty(top10_recs, article_pops)
        novelty_list.append(novelty)
        
        # Slicing: Cold-start (<= 5 clicks) vs Warm (> 5 clicks)
        history_len = len([h for h in history if h and h != 0])
        if history_len <= 5:
            cold_aucs.append(auc)
        else:
            warm_aucs.append(auc)
            
    # Calculate bootstrap 95% CIs
    auc_mean, auc_l, auc_u = compute_bootstrap_ci(np.array(auc_list))
    mrr_mean, mrr_l, mrr_u = compute_bootstrap_ci(np.array(mrr_list))
    ndcg5_mean, ndcg5_l, ndcg5_u = compute_bootstrap_ci(np.array(ndcg5_list))
    ndcg10_mean, ndcg10_l, ndcg10_u = compute_bootstrap_ci(np.array(ndcg10_list))
    
    cold_mean, _, _ = compute_bootstrap_ci(np.array(cold_aucs)) if cold_aucs else (0.0, 0.0, 0.0)
    warm_mean, _, _ = compute_bootstrap_ci(np.array(warm_aucs)) if warm_aucs else (0.0, 0.0, 0.0)
    
    novelty_mean = float(np.mean(novelty_list)) if novelty_list else 0.0
    ild_mean = float(np.mean(ild_list)) if ild_list else 0.0
    catalog_cov = calc_catalog_coverage(all_recommendations, df_articles.height)
    
    results = {
        "dataset": dataset,
        "retriever": retriever_type,
        "eval_samples": len(auc_list),
        "AUC": f"{auc_mean:.4f} [{auc_l:.4f}, {auc_u:.4f}]",
        "MRR": f"{mrr_mean:.4f} [{mrr_l:.4f}, {mrr_u:.4f}]",
        "nDCG@5": f"{ndcg5_mean:.4f} [{ndcg5_l:.4f}, {ndcg5_u:.4f}]",
        "nDCG@10": f"{ndcg10_mean:.4f} [{ndcg10_l:.4f}, {ndcg10_u:.4f}]",
        "Novelty": f"{novelty_mean:.4f}",
        "ILD": f"{ild_mean:.4f}",
        "Catalog_Coverage": f"{catalog_cov:.4f}",
        "Slice_ColdStart_AUC": f"{cold_mean:.4f} (n={len(cold_aucs)})",
        "Slice_Warm_AUC": f"{warm_mean:.4f} (n={len(warm_aucs)})"
    }
    
    print(f"\n=======================================================")
    print(f"  HARNESS RESULTS: {dataset.upper()} - {retriever_type.upper()}")
    print(f"=======================================================")
    for k, v in results.items():
        print(f"  {k:25s}: {v}")
    print(f"=======================================================\n")
    
    return results

def evaluate_ablation(dataset: str, retriever_type: str, limit: int = 2000):
    """
    Q9 Anti-Gaming Ablation: Report metrics WITH and WITHOUT features 
    unavailable at serving time. We ablate by removing user click history,
    which simulates the cold-start / no-personalization scenario that would
    be the baseline at serving time without access to historical behavior data.
    """
    print(f"\n{'='*60}")
    print(f"  ANTI-GAMING ABLATION: {dataset.upper()} ({retriever_type.upper()})")
    print(f"  Comparing: Full History vs. No History (Random Scoring)")
    print(f"{'='*60}")
    
    if dataset == "mind":
        proc_dir = Path("data/processed/mind")
    else:
        proc_dir = Path("data/processed/ebnerd_demo")
        
    val_path = proc_dir / "val.parquet"
    articles_path = proc_dir / "articles.parquet"
    
    if not val_path.exists() or not articles_path.exists():
        print(f"Error: {val_path} or {articles_path} not found. Run `make data` first.")
        return None
        
    df_articles = pl.read_parquet(articles_path)
    df_val = pl.read_parquet(val_path)
    
    if limit and limit < df_val.height:
        df_val = df_val.head(limit)
        
    if retriever_type == "bm25":
        retriever = BM25Retriever(df_articles)
    else:
        retriever = SemanticRetriever(df_articles)
    
    hist_col = "article_id_fixed" if "article_id_fixed" in df_val.columns else "history_article_id"
    histories = df_val[hist_col].to_list() if hist_col in df_val.columns else [[]] * df_val.height
    inviews = df_val["article_ids_inview"].to_list()
    clickeds = df_val["article_ids_clicked"].to_list()
    
    # Run WITH full history (normal) and WITHOUT history (ablated)
    metrics_with = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}
    metrics_without = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}
    
    for history, candidates, clicked in zip(histories, inviews, tqdm(clickeds, desc="Ablation")):
        if history is None:
            history = []
        if candidates is None or not candidates or clicked is None or not clicked:
            continue
            
        y_true = np.array([1 if c in clicked else 0 for c in candidates])
        if np.sum(y_true) == 0:
            continue
            
        # WITH full history (normal mode)
        scores_with = retriever.score_candidates(history, candidates)
        scores_with_arr = np.array(scores_with)
        
        metrics_with["auc"].append(calc_auc(y_true, scores_with_arr))
        metrics_with["mrr"].append(calc_mrr(y_true, scores_with_arr))
        metrics_with["ndcg5"].append(calc_ndcg(y_true, scores_with_arr, k=5))
        metrics_with["ndcg10"].append(calc_ndcg(y_true, scores_with_arr, k=10))
        
        # WITHOUT history (empty history = no user behavior features)
        scores_without = retriever.score_candidates([], candidates)
        scores_without_arr = np.array(scores_without)
        
        metrics_without["auc"].append(calc_auc(y_true, scores_without_arr))
        metrics_without["mrr"].append(calc_mrr(y_true, scores_without_arr))
        metrics_without["ndcg5"].append(calc_ndcg(y_true, scores_without_arr, k=5))
        metrics_without["ndcg10"].append(calc_ndcg(y_true, scores_without_arr, k=10))
    
    result = {
        "dataset": dataset,
        "retriever": retriever_type,
        "ablation": "history_removal",
        "n_samples": len(metrics_with["auc"]),
        "with_history": {
            "AUC": f"{np.mean(metrics_with['auc']):.4f}",
            "MRR": f"{np.mean(metrics_with['mrr']):.4f}",
            "nDCG@5": f"{np.mean(metrics_with['ndcg5']):.4f}",
            "nDCG@10": f"{np.mean(metrics_with['ndcg10']):.4f}",
        },
        "without_history": {
            "AUC": f"{np.mean(metrics_without['auc']):.4f}",
            "MRR": f"{np.mean(metrics_without['mrr']):.4f}",
            "nDCG@5": f"{np.mean(metrics_without['ndcg5']):.4f}",
            "nDCG@10": f"{np.mean(metrics_without['ndcg10']):.4f}",
        }
    }
    
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS: {dataset.upper()} - {retriever_type.upper()}")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'With History':>14} {'Without History':>16} {'Δ':>8}")
    print(f"  {'-'*52}")
    for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        w = float(result["with_history"][metric])
        wo = float(result["without_history"][metric])
        delta = w - wo
        print(f"  {metric:<12} {w:>14.4f} {wo:>16.4f} {delta:>+8.4f}")
    print(f"{'='*60}\n")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["mind", "ebnerd", "all"])
    parser.add_argument("--retriever", type=str, default="all", choices=["bm25", "semantic", "all"])
    parser.add_argument("--limit", type=int, default=2000, help="Number of validation impressions to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Run Q9 anti-gaming ablation (with/without history)")
    args = parser.parse_args()
    
    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    retrievers = ["bm25", "semantic"] if args.retriever == "all" else [args.retriever]
    
    all_results = []
    ablation_results = []
    for d in datasets:
        for r in retrievers:
            res = evaluate_harness(d, r, limit=args.limit)
            if res:
                all_results.append(res)
            if args.ablation:
                abl = evaluate_ablation(d, r, limit=args.limit)
                if abl:
                    ablation_results.append(abl)
                
    out_file = Path("outputs/eval_harness_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {"harness": all_results}
    if ablation_results:
        output_data["ablation"] = ablation_results
        
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Full evaluation results saved to {out_file}")

if __name__ == "__main__":
    main()
