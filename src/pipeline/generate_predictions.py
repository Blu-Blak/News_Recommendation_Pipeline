import argparse
from pathlib import Path
import polars as pl
import numpy as np
import sys
import zipfile
import pyarrow.parquet as pq
from tqdm import tqdm

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.bm25_retriever import BM25Retriever
from src.pipeline.semantic_retriever import SemanticRetriever
from ebrec.utils._behaviors import truncate_history
from ebrec.utils._constants import DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL

def rank_candidates(scores: list[float]) -> list[int]:
    """Converts a list of scores into a list of 1-indexed ranks. 
    Higher score gets better (lower number) rank."""
    n = len(scores)
    if n == 0:
        return []
    # Timsort is significantly faster than numpy argsort for small lists (len < 100)
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranks = [0] * n
    for rank0, (original_idx, _) in enumerate(indexed_scores):
        ranks[original_idx] = rank0 + 1
    return ranks

def generate_predictions(dataset: str, retriever_type: str, test_data_path: Path, articles_path: Path, out_dir: Path):
    print(f"Loading {dataset} test articles...")
    df_articles = pl.read_parquet(articles_path)
    
    print(f"Initializing {retriever_type} retriever...")
    if retriever_type == "bm25":
        retriever = BM25Retriever(df_articles)
    else:
        retriever = SemanticRetriever(df_articles)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    txt_filename = "prediction.txt" if dataset == "mind" else "predictions.txt"
    txt_path = out_dir / txt_filename
    
    print(f"Generating predictions to {txt_path}...")
    
    with open(txt_path, "w") as f_out:
        if dataset == "mind":
            # For MIND, we already built test_submission.parquet
            parquet_file = pq.ParquetFile(test_data_path)
            for i in range(parquet_file.num_row_groups):
                print(f"Processing row group {i+1}/{parquet_file.num_row_groups}...")
                df_batch = pl.from_arrow(parquet_file.read_row_group(i))
                total_rows = df_batch.height
                
                imp_ids = df_batch["impression_id"].to_list()
                histories = df_batch["article_id_fixed"].to_list() if "article_id_fixed" in df_batch.columns else [[]] * total_rows
                candidates_list = df_batch["article_ids_inview"].to_list()
                
                for imp_id, history, candidates in zip(imp_ids, histories, tqdm(candidates_list, desc=f"Batch {i+1}/{parquet_file.num_row_groups}")):
                    if history is None:
                        history = []
                    if candidates is None or not candidates or candidates == [""]:
                        f_out.write(f"{imp_id} []\n")
                        continue
                    
                    scores = retriever.score_candidates(history, candidates)
                    ranks = rank_candidates(scores)
                    
                    ranks_str = ",".join(map(str, ranks))
                    f_out.write(f"{imp_id} [{ranks_str}]\n")
        else:
            # For EB-NeRD, we do the join chunk-by-chunk in memory to avoid segfaults
            test_dir = Path("data/raw/ebnerd_testset")
            if (test_dir / "ebnerd_testset").exists():
                test_dir = test_dir / "ebnerd_testset"
            history_path = test_dir / "test/history.parquet"
            behaviors_path = test_dir / "test/behaviors.parquet"
            
            print("Loading history into memory...")
            df_history = pl.read_parquet(history_path).select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(
                truncate_history, column=DEFAULT_HISTORY_ARTICLE_ID_COL, history_size=30, padding_value=0, enable_warning=False
            )
            
            parquet_file = pq.ParquetFile(behaviors_path)
            for i in range(parquet_file.num_row_groups):
                print(f"Processing row group {i+1}/{parquet_file.num_row_groups}...")
                df_behaviors = pl.from_arrow(parquet_file.read_row_group(i))
                df_batch = df_behaviors.with_row_index("row_idx").join(
                    df_history, on=DEFAULT_USER_COL, how="left"
                ).sort("row_idx").drop("row_idx")
                total_rows = df_batch.height
                
                imp_ids = df_batch["impression_id"].to_list()
                hist_col = "article_id_fixed" if "article_id_fixed" in df_batch.columns else "history_article_id"
                histories = df_batch[hist_col].to_list() if hist_col in df_batch.columns else [[]] * total_rows
                candidates_list = df_batch["article_ids_inview"].to_list()
                
                for imp_id, history, candidates in zip(imp_ids, histories, tqdm(candidates_list, desc=f"Batch {i+1}/{parquet_file.num_row_groups}")):
                    if history is None:
                        history = []
                    if candidates is None or not candidates or candidates == [""]:
                        f_out.write(f"{imp_id} []\n")
                        continue
                    
                    scores = retriever.score_candidates(history, candidates)
                    ranks = rank_candidates(scores)
                    
                    ranks_str = ",".join(map(str, ranks))
                    f_out.write(f"{imp_id} [{ranks_str}]\n")

    zip_path = out_dir / f"{dataset}_{retriever_type}_submission.zip"
    print(f"Creating submission zip: {zip_path}")
    
    # Try standard DEFLATE at level 9 first
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        zipf.write(txt_path, arcname=txt_filename)
        
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Initial Zip Size (DEFLATE): {size_mb:.2f} MB")
    
    # If the zip file exceeds 48MB, re-compress using ZIP_LZMA to guarantee <50MB size for submission portals
    if size_mb > 48.0:
        print("Zip exceeds 48MB limit. Re-compressing using LZMA algorithm for ultra-high compression (<50MB)...")
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_LZMA) as zipf:
            zipf.write(txt_path, arcname=txt_filename)
        final_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Final Squeezed Zip Size (LZMA): {final_size_mb:.2f} MB")
    else:
        print(f"Final Zip Size: {size_mb:.2f} MB")
        
    txt_path.unlink()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mind", "ebnerd"])
    parser.add_argument("--retriever", type=str, required=True, choices=["bm25", "semantic"])
    args = parser.parse_args()
    
    if args.dataset == "mind":
        processed_dir = Path("data/processed/mind")
    else:
        processed_dir = Path("data/processed/ebnerd_demo")
        
    test_data_path = processed_dir / "test_submission.parquet"
    articles_path = processed_dir / "test_articles.parquet"
    
    if args.dataset == "mind" and not test_data_path.exists():
        print(f"Error: Test data not found in {processed_dir}. Run `make data-large` first.")
        sys.exit(1)
        
    if not articles_path.exists():
        print(f"Error: Articles not found in {processed_dir}. Run `make data-large` first.")
        sys.exit(1)
        
    out_dir = Path("outputs/submissions")
    generate_predictions(args.dataset, args.retriever, test_data_path, articles_path, out_dir)
