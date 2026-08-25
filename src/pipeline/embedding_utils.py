import argparse
from pathlib import Path
import polars as pl
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm

def _process_file(articles_path: Path, model):
    if not articles_path.exists():
        return
        
    print(f"Loading articles from {articles_path}...")
    df_articles = pl.read_parquet(articles_path)
    
    if "bert" in df_articles.columns:
        print(f"Embeddings ('bert' column) already exist in {articles_path.name}. Skipping.")
        return
        
    texts = []
    for row in df_articles.iter_rows(named=True):
        title = row.get("title") or ""
        abstract = row.get("abstract") or row.get("subtitle") or ""
        texts.append(f"{title} {abstract}".strip())
        
    print(f"Computing embeddings for {len(texts)} articles in {articles_path.name}. This may take a few minutes...")
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    print("Saving embeddings back to parquet...")
    embedding_series = pl.Series("bert", embeddings.tolist())
    df_articles = df_articles.with_columns(embedding_series)
    
    df_articles.write_parquet(articles_path)
    print(f"Successfully saved embeddings to {articles_path}")

def generate_all_embeddings():
    print("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Process MIND articles
    mind_dir = Path("data/processed/mind")
    _process_file(mind_dir / "articles.parquet", model)
    _process_file(mind_dir / "test_articles.parquet", model)
    
    # Process EB-NeRD articles
    ebnerd_dir = Path("data/processed/ebnerd_demo")
    _process_file(ebnerd_dir / "articles.parquet", model)
    _process_file(ebnerd_dir / "test_articles.parquet", model)

if __name__ == "__main__":
    generate_all_embeddings()
