import polars as pl
from pathlib import Path

def parse_mind_news(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        quote_char=None
    )
    return df

def parse_mind_behaviors(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["impression_id", "user_id", "time", "history", "impressions"],
        quote_char=None
    )
    
    # Fill nulls in history (some users have no history)
    df = df.with_columns(pl.col("history").fill_null(""))
    
    # Process history: space separated string of article IDs -> list of strings
    df = df.with_columns(
        pl.col("history").str.split(" ").alias("article_id_fixed")
    )
    
    # Process impressions: space separated string of articleID-label -> extract lists
    # E.g. "N123-1 N456-0" -> article_ids_inview=["N123", "N456"], labels=[1, 0]
    df = df.with_columns(
        pl.col("impressions").str.split(" ").list.eval(pl.element().str.split("-").list.get(0)).alias("article_ids_inview"),
        pl.col("impressions").str.split(" ").list.eval(pl.element().str.split("-").list.get(1).cast(pl.Int8)).alias("labels")
    )
    
    # Explode to filter out clicked articles
    exploded = df.select(["impression_id", "article_ids_inview", "labels"]).explode(["article_ids_inview", "labels"])
    clicked = exploded.filter(pl.col("labels") == 1).group_by("impression_id").agg(pl.col("article_ids_inview").alias("article_ids_clicked"))
    
    df = df.join(clicked, on="impression_id", how="left")
    
    # Parse time to datetime
    df = df.with_columns(
        pl.col("time").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p").alias("impression_time")
    )
    
    return df
