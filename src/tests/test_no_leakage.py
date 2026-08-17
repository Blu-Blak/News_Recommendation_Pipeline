import polars as pl
from pathlib import Path
import pytest

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
