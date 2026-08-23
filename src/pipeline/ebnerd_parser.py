import polars as pl
from pathlib import Path

# We need to import from ebrec
from ebrec.utils._behaviors import create_binary_labels_column, truncate_history
from ebrec.utils._constants import DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .join(df_history.collect(), on=DEFAULT_USER_COL, how="left")
    )
    return df_behaviors

def parse_ebnerd_news(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)

def parse_ebnerd_behaviors(path: Path) -> pl.DataFrame:
    df = ebnerd_from_path(path)
    df = create_binary_labels_column(df)
    return df

def parse_ebnerd_test_behaviors(path: Path, history_size: int = 30) -> pl.LazyFrame:
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .join(df_history, on=DEFAULT_USER_COL, how="left")
    )
    # The test set has article_ids_inview but NO article_ids_clicked and NO labels.
    # Return a LazyFrame to allow batched processing later.
    return df_behaviors
