import polars as pl

def temporal_split(df: pl.DataFrame, time_col: str = "impression_time", test_fraction: float = 0.5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sorts by time_col and splits into two dataframes (e.g. val and test).
    """
    df = df.sort(time_col)
    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    return df[:split_idx], df[split_idx:]
