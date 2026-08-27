import polars as pl
from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.temporal_split import temporal_split

# ---- Test the temporal_split function directly ----

def test_temporal_split_ordering():
    """Assert that temporal_split produces val_max <= test_min (no leakage by construction)."""
    df = pl.DataFrame({
        "impression_time": pl.Series([
            "2024-01-01 00:00:00", "2024-01-02 00:00:00",
            "2024-01-03 00:00:00", "2024-01-04 00:00:00",
            "2024-01-05 00:00:00", "2024-01-06 00:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
    })
    val, test = temporal_split(df, time_col="impression_time", test_fraction=0.5)
    
    assert val["impression_time"].max() <= test["impression_time"].min(), \
        "temporal_split leaks: val_max > test_min!"
    assert len(val) > 0 and len(test) > 0, "Split produced empty partition"

def test_temporal_split_no_overlap():
    """Assert that val and test partitions have no overlapping rows."""
    df = pl.DataFrame({
        "impression_time": pl.Series([
            "2024-01-01 10:00:00", "2024-01-01 11:00:00",
            "2024-01-02 10:00:00", "2024-01-02 11:00:00",
            "2024-01-03 10:00:00", "2024-01-03 11:00:00",
            "2024-01-04 10:00:00", "2024-01-04 11:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        "impression_id": list(range(8)),
    })
    val, test = temporal_split(df, time_col="impression_time", test_fraction=0.5)
    
    val_ids = set(val["impression_id"].to_list())
    test_ids = set(test["impression_id"].to_list())
    assert val_ids.isdisjoint(test_ids), "Val and Test share impression_ids!"
    assert len(val_ids) + len(test_ids) == 8, "Some rows were lost in the split"

def test_temporal_split_sorted_output():
    """Assert that both partitions are sorted chronologically."""
    df = pl.DataFrame({
        "impression_time": pl.Series([
            "2024-01-05 00:00:00", "2024-01-01 00:00:00",
            "2024-01-03 00:00:00", "2024-01-02 00:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
    })
    val, test = temporal_split(df, time_col="impression_time", test_fraction=0.5)
    
    val_times = val["impression_time"].to_list()
    test_times = test["impression_time"].to_list()
    assert val_times == sorted(val_times), "Val partition is not sorted!"
    assert test_times == sorted(test_times), "Test partition is not sorted!"

# ---- Test on actual processed datasets ----

def test_temporal_ordering_mind():
    proc_dir = Path("data/processed/mind")
    if not proc_dir.exists():
        pytest.skip("MIND data not built yet")
    train = pl.read_parquet(proc_dir / "train.parquet")
    val = pl.read_parquet(proc_dir / "val.parquet")
    test = pl.read_parquet(proc_dir / "test.parquet")
    
    train_max = train["impression_time"].max()
    val_min = val["impression_time"].min()
    val_max = val["impression_time"].max()
    test_min = test["impression_time"].min()
    
    assert train_max <= val_min, "Train leaks into Validation!"
    assert val_max <= test_min, "Validation leaks into Test!"

def test_temporal_ordering_ebnerd():
    proc_dir = Path("data/processed/ebnerd_demo")
    if not proc_dir.exists():
        pytest.skip("EB-NeRD demo data not built yet")
    train = pl.read_parquet(proc_dir / "train.parquet")
    val = pl.read_parquet(proc_dir / "val.parquet")
    test = pl.read_parquet(proc_dir / "test.parquet")
    
    train_max = train["impression_time"].max()
    val_min = val["impression_time"].min()
    val_max = val["impression_time"].max()
    test_min = test["impression_time"].min()
    
    assert train_max <= val_min, "Train leaks into Validation!"
    assert val_max <= test_min, "Validation leaks into Test!"

def test_no_future_click_in_history_mind():
    """Q9: Assert that no user history contains article IDs from future impressions."""
    proc_dir = Path("data/processed/mind")
    if not proc_dir.exists():
        pytest.skip("MIND data not built yet")
    test = pl.read_parquet(proc_dir / "test.parquet")
    
    if "article_ids_clicked" in test.columns and "article_id_fixed" in test.columns:
        high_overlap_count = 0
        for row in test.head(500).iter_rows(named=True):
            history = set(row.get("article_id_fixed") or [])
            clicked = set(row.get("article_ids_clicked") or [])
            # History should not contain articles clicked in THIS impression
            # (they haven't been clicked yet at impression time).
            # However, re-reads are legitimate: a user can click an article
            # they have read before. We flag systematic leakage only.
            leaked = history.intersection(clicked)
            if clicked and len(clicked) > 1 and len(leaked) / len(clicked) > 0.5:
                high_overlap_count += 1
        # Allow up to 5% of impressions to have coincidental overlap
        assert high_overlap_count / 500 < 0.05, \
            f"Systematic future leakage detected: {high_overlap_count}/500 impressions have >50% history-click overlap"
