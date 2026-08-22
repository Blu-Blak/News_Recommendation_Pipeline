import argparse
from pathlib import Path
import polars as pl
from tqdm import tqdm
import sys
import numpy as np
import multiprocessing

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.bm25_retriever import BM25Retriever
from src.pipeline.semantic_retriever import SemanticRetriever

def calculate_recall_at_k(retrieved: list, ground_truth: list, k: int) -> float:
    if not ground_truth:
        return 0.0
    retrieved_k = set(retrieved[:k])
    gt_set = set(ground_truth)
    hits = len(retrieved_k.intersection(gt_set))
    return hits / len(gt_set)

# Global variables for multiprocessing worker
_global_retriever = None

def init_worker(retriever_inst):
    global _global_retriever
    _global_retriever = retriever_inst

def process_row(row_tuple):
    history, clicked = row_tuple
    if not clicked:
        return 0.0, 0.0, 0.0
    retrieved = _global_retriever.retrieve(history, top_k=200)
    return (
        calculate_recall_at_k(retrieved, clicked, 50),
        calculate_recall_at_k(retrieved, clicked, 100),
        calculate_recall_at_k(retrieved, clicked, 200)
    )

def evaluate_dataset(dataset: str, retriever_type: str, scale: str = "demo", limit: int = None):
    if dataset == "mind":
        proc_dir = Path("data/processed/mind")
    else:
        proc_dir = Path(f"data/processed/ebnerd_{scale}")
        
    print(f"Loading {dataset} from {proc_dir}...")
    df_articles = pl.read_parquet(proc_dir / "articles.parquet")
    df_val = pl.read_parquet(proc_dir / "val.parquet")
    
    if retriever_type == "semantic":
        retriever = SemanticRetriever(df_articles)
    else:
        retriever = BM25Retriever(df_articles)
    
    # Extract data for processing
    rows_to_process = []
    for row in df_val.iter_rows(named=True):
        history = row.get("article_id_fixed") or []
        clicked = row.get("article_ids_clicked") or []
        if clicked:
            rows_to_process.append((history, list(clicked)))
            
    if limit is not None:
        rows_to_process = rows_to_process[:limit]
            
    print(f"Evaluating {len(rows_to_process)} impressions with multiprocessing...")
    
    recall_50, recall_100, recall_200 = [], [], []
    
    if retriever_type == "semantic":
        # FAISS is natively multi-threaded in C++ and cannot be pickled for multiprocessing safely.
        # We run it sequentially (it will still be extremely fast).
        init_worker(retriever)
        results = []
        for row in tqdm(rows_to_process, desc="Evaluating Semantic"):
            results.append(process_row(row))
    else:
        # BM25 is pure Python and single-threaded, so we use multiprocessing
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(retriever,)) as pool:
            results = list(tqdm(pool.imap(process_row, rows_to_process, chunksize=100), total=len(rows_to_process)))
            
    for r50, r100, r200 in results:
        recall_50.append(r50)
        recall_100.append(r100)
        recall_200.append(r200)
        
    print("\n" + "="*40)
    print(f"{retriever_type.upper()} Retrieval Results - {dataset.upper()}")
    print("="*40)
    print(f"Recall@50:  {np.mean(recall_50):.4f}")
    print(f"Recall@100: {np.mean(recall_100):.4f}")
    print(f"Recall@200: {np.mean(recall_200):.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mind", "ebnerd", "all"])
    parser.add_argument("--scale", type=str, default="demo", choices=["demo", "small"])
    parser.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "semantic"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of impressions to evaluate")
    args = parser.parse_args()
    
    if args.dataset == "all":
        evaluate_dataset("ebnerd", args.retriever, args.scale, args.limit)
        evaluate_dataset("mind", args.retriever, args.limit)
    else:
        evaluate_dataset(args.dataset, args.retriever, args.scale, args.limit)
