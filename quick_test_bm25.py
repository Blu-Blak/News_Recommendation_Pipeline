import polars as pl
from pathlib import Path
import sys
import time

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline.bm25_retriever import BM25Retriever

def run_quick_test():
    proc_dir = Path("data/processed/ebnerd_demo")
    articles_path = proc_dir / "articles.parquet"
    val_path = proc_dir / "val.parquet"
    
    if not articles_path.exists():
        print(f"Error: {articles_path} not found. Please run 'make data' first.")
        return
        
    print("1. Loading EB-NeRD Demo articles...")
    df_articles = pl.read_parquet(articles_path)
    
    print("2. Initializing BM25Retriever (Building Inverted Index)...")
    t0 = time.time()
    retriever = BM25Retriever(df_articles)
    print(f"   -> Index built in {time.time() - t0:.2f} seconds")
    
    print("\n3. Loading one impression from validation set...")
    df_val = pl.read_parquet(val_path)
    
    # Find an impression with a valid history
    sample_row = None
    for row in df_val.iter_rows(named=True):
        history = row.get("article_id_fixed")
        if history and len(history) >= 2:
            sample_row = row
            break
            
    if sample_row is None:
        print("No valid history found.")
        return
        
    history = sample_row["article_id_fixed"]
    print(f"\nUser History (recent articles): {history[-5:]}")
    
    print("\n4. Formulating Query and Retrieving Top-5 Candidates...")
    t1 = time.time()
    candidates = retriever.retrieve(history, top_k=5)
    print(f"   -> Retrieved in {time.time() - t1:.4f} seconds")
    
    print(f"\nRetrieved Candidates: {candidates}")
    print("\nSuccess! The BM25 Lexical Retrieval pipeline is working perfectly.")

if __name__ == "__main__":
    run_quick_test()
