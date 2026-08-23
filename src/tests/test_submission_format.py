import zipfile
import re
from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.generate_predictions import rank_candidates

def test_rank_candidates_correctness():
    # Test descending score ranking
    scores = [0.1, 0.9, 0.4, 0.7]
    # Expected: 0.9 -> rank 1, 0.7 -> rank 2, 0.4 -> rank 3, 0.1 -> rank 4
    # Result ranks array for input indices [0, 1, 2, 3]:
    # index 0 (0.1) -> 4
    # index 1 (0.9) -> 1
    # index 2 (0.4) -> 3
    # index 3 (0.7) -> 2
    ranks = rank_candidates(scores)
    assert ranks == [4, 1, 3, 2]
    assert min(ranks) == 1
    assert max(ranks) == 4
    assert sorted(ranks) == [1, 2, 3, 4]

def test_rank_candidates_empty():
    assert rank_candidates([]) == []

def test_submission_zip_structure():
    submissions_dir = Path("outputs/submissions")
    if not submissions_dir.exists():
        pytest.skip("Submissions directory not found")
        
    zip_files = list(submissions_dir.glob("*.zip"))
    if not zip_files:
        pytest.skip("No submission zip files generated yet")
        
    for zpath in zip_files:
        with zipfile.ZipFile(zpath, 'r') as zf:
            namelist = zf.namelist()
            # 1. Exactly 1 file in zip
            assert len(namelist) == 1, f"{zpath.name} contains multiple files/folders: {namelist}"
            
            fname = namelist[0]
            # 2. Correct file name depending on dataset
            if zpath.name.startswith("mind"):
                assert fname == "prediction.txt", f"MIND zip must contain 'prediction.txt', found '{fname}'"
            elif zpath.name.startswith("ebnerd"):
                assert fname == "predictions.txt", f"EB-NeRD zip must contain 'predictions.txt', found '{fname}'"
                
            # 3. Read sample lines and verify format
            with zf.open(fname) as f:
                first_lines = [f.readline().decode('utf-8').strip() for _ in range(10)]
                for line in first_lines:
                    if not line:
                        continue
                    # Match pattern: ImpressionID [1,2,3,...]
                    assert re.match(r"^\w+\s+\[\d*(,\d+)*\]$", line), f"Line format invalid: {line}"
