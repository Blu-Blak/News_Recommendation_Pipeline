import argparse
from pathlib import Path
import polars as pl
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm

def generate_mind_embeddings():
    proc_dir = Path("data/processed/mind")
    articles_path = proc_dir / "articles.parquet"
    
    if not articles_path.exists():
        print(f"File not found: {articles_path}")
        return
        
    print(f"Loading MIND articles from {articles_path}...")
    df_articles = pl.read_parquet(articles_path)
    
    if "bert" in df_articles.columns:
        print("Embeddings ('bert' column) already exist in MIND articles.parquet. Skipping.")
        return
        
    print("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = []
    for row in df_articles.iter_rows(named=True):
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        texts.append(f"{title} {abstract}".strip())
        
    print(f"Computing embeddings for {len(texts)} articles. This may take a few minutes...")
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    print("Saving embeddings back to articles.parquet...")
    embedding_series = pl.Series("bert", embeddings.tolist())
    df_articles = df_articles.with_columns(embedding_series)
    
    df_articles.write_parquet(articles_path)
    print(f"Successfully saved embeddings to {articles_path}")

if __name__ == "__main__":
    generate_mind_embeddings()
