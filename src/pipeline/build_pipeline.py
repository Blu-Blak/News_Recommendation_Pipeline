import argparse
from pathlib import Path
import polars as pl
import sys

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.download_data import download_mind, download_ebnerd
from src.pipeline.mind_parser import parse_mind_news, parse_mind_behaviors
from src.pipeline.ebnerd_parser import parse_ebnerd_news, parse_ebnerd_behaviors
from src.pipeline.temporal_split import temporal_split
from src.pipeline.feature_store import save_features

def build_mind(raw_dir: Path, out_dir: Path):
    print("Building MIND pipeline...")
    download_mind(raw_dir)
    
    train_dir = raw_dir / "MINDsmall_train"
    if (train_dir / "MINDsmall_train").exists():
        train_dir = train_dir / "MINDsmall_train"
        
    dev_dir = raw_dir / "MINDsmall_dev"
    if (dev_dir / "MINDsmall_dev").exists():
        dev_dir = dev_dir / "MINDsmall_dev"
    
    df_articles_train = parse_mind_news(train_dir / "news.tsv")
    df_articles_dev = parse_mind_news(dev_dir / "news.tsv")
    # Combine and deduplicate articles
    df_articles = pl.concat([df_articles_train, df_articles_dev]).unique(subset=["article_id"])
    
    df_train = parse_mind_behaviors(train_dir / "behaviors.tsv")
    df_dev = parse_mind_behaviors(dev_dir / "behaviors.tsv")
    
    # Temporal split of dev into val and test
    df_val, df_test = temporal_split(df_dev, time_col="impression_time", test_fraction=0.5)
    
    save_features(df_articles, df_train, df_val, df_test, out_dir)
    print(f"Saved MIND to {out_dir}")

def build_ebnerd(raw_dir: Path, out_dir: Path, scale: str):
    print(f"Building EB-NeRD ({scale}) pipeline...")
    download_ebnerd(raw_dir, scale)
    
    bundle_dir = raw_dir / f"ebnerd_{scale}"
    df_articles = parse_ebnerd_news(bundle_dir / "articles.parquet")
    
    # Load and join EB-NeRD embeddings if available
    w2v_path = raw_dir / "Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec/document_vector.parquet"
    if w2v_path.exists():
        print("Joining Word2Vec embeddings...")
        df_w2v = pl.read_parquet(w2v_path).rename({"document_vector": "word2vec"})
        df_articles = df_articles.join(df_w2v, on="article_id", how="left")
        
    bert_path = raw_dir / "google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
    if bert_path.exists():
        print("Joining BERT embeddings...")
        df_bert = pl.read_parquet(bert_path).rename({"google-bert/bert-base-multilingual-cased": "bert"})
        df_articles = df_articles.join(df_bert, on="article_id", how="left")
    
    df_train = parse_ebnerd_behaviors(bundle_dir / "train")
    df_validation = parse_ebnerd_behaviors(bundle_dir / "validation")
    
    df_val, df_test = temporal_split(df_validation, time_col="impression_time", test_fraction=0.5)
    
    save_features(df_articles, df_train, df_val, df_test, out_dir)
    print(f"Saved EB-NeRD to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mind", "ebnerd"])
    parser.add_argument("--scale", type=str, default="demo", choices=["demo", "small"])
    args = parser.parse_args()
    
    raw_dir = Path("data/raw")
    if args.dataset == "mind":
        out_dir = Path("data/processed/mind")
        build_mind(raw_dir, out_dir)
    else:
        out_dir = Path(f"data/processed/ebnerd_{args.scale}")
        build_ebnerd(raw_dir, out_dir, args.scale)
