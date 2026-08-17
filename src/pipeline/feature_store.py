import polars as pl
from pathlib import Path

def add_recency_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a history_recency column with linearly decaying weights for history articles."""
    if "article_id_fixed" in df.columns:
        df = df.with_columns(
            pl.int_ranges(1, pl.col("article_id_fixed").list.len() + 1)
            .list.eval(pl.element() / pl.element().max())
            .alias("history_recency")
        )
    return df

def save_features(df_articles: pl.DataFrame, df_train: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_articles.write_parquet(out_dir / "articles.parquet")
    add_recency_features(df_train).write_parquet(out_dir / "train.parquet")
    add_recency_features(df_val).write_parquet(out_dir / "val.parquet")
    add_recency_features(df_test).write_parquet(out_dir / "test.parquet")
