# News Recommendation Pipeline

Implementation of **Assignment 1** for **CS4.406: Information Retrieval & Extraction** (Lexical & Semantic Retrieval on EB-NeRD and MIND).

---

## 🚀 Quick Start & Reproducibility

### 1. Installation & Environment Setup
```bash
make install
```

### 2. One-Command Data Pipeline
Downloads raw datasets, cleans articles/behaviors, applies temporal splitting, and constructs feature stores:
```bash
# Small/demo sets for quick iteration
make data

# Full large test datasets for Codabench submissions
make data-large
```

### 3. Compute Embeddings
Generates BERT article embeddings for semantic candidate retrieval:
```bash
make embed
```

### 4. Offline Evaluation Harness (Q4)
Runs official ranking metrics (AUC, MRR, nDCG@5, nDCG@10), beyond-accuracy metrics (ILD, Novelty, Coverage), user/item slicing, and 95% bootstrap confidence intervals:
```bash
make evaluate-harness
```

### 5. Codabench Prediction Generation (Q5)
Generates compliant, compressed submission `.zip` files for both MIND and EB-NeRD:
```bash
# Generate all 4 submissions
make submission

# Or individual submissions:
make submission-mind-bm25
make submission-mind-semantic
make submission-ebnerd-bm25
make submission-ebnerd-semantic
```

### 6. Automated Unit Tests
```bash
make test
```

### 7. HPC SLURM Batch Job Submission
```bash
sbatch run_submission.sbatch
```

---

## 📑 Deliverables & Documentation
- **Design Note Report ($\le 4$ pages)**: [`design_note.md`](file:///home/dhruvmalik/Desktop/IRE/News_Recommendation_Pipeline/design_note.md)
- **AI Usage Log**: [`AI_USAGE_LOG.md`](file:///home/dhruvmalik/Desktop/IRE/News_Recommendation_Pipeline/AI_USAGE_LOG.md)
- **Harness Results JSON**: `outputs/eval_harness_results.json`
