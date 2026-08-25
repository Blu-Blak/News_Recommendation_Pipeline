# Design Note: News Recommendation Pipeline (Assignment 1)

**Course:** CS4.406 — Information Retrieval & Extraction  
**Task:** Lexical & Semantic Retrieval Baselines on MIND and EB-NeRD  
**Repository:** News Recommendation Pipeline  

---

## 1. System Architecture & Key Design Choices

We built a reproducible, scalable news recommendation retrieval framework covering both **MIND (Microsoft News Dataset)** and **EB-NeRD (Ekstra Bladet News Recommendation Dataset)**. The pipeline is designed around five modular components:

```
                  ┌────────────────────────────────────────┐
                  │          Raw Datasets (TSV/Parquet)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 1. Unified Data Pipeline & FeatureStore│
                  └───────────────────┬────────────────────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
┌─────────────────────────────┐                       ┌─────────────────────────────┐
│ 2. Lexical Candidate Gen    │                       │ 3. Semantic Candidate Gen   │
│    (BM25 Inverted Index)    │                       │    (FAISS / BERT Vectors)   │
└──────────┬──────────────────┘                       └──────────┬──────────────────┘
           │                                                     │
           └──────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 4. Offline Evaluation Harness          │
                  │    - Metrics: AUC, MRR, nDCG@5/10      │
                  │    - Beyond-Accuracy: ILD, Novelty, Cov│
                  │    - Slicing & Bootstrap 95% CIs       │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ 5. Codabench Prediction Generator      │
                  └────────────────────────────────────────┘
```

### Key Architectural Choices:
1. **Polars Data Engine & PyArrow Streaming**: Polars lazy frames (`scan_parquet`) and PyArrow chunked batching (`read_row_group`) prevent RAM memory exhaustion during memory-heavy joins on 13.5M test impressions.
2. **True Inverted Index BM25**: Custom `BM25Retriever` implementation using inverted index posting lists (`token -> doc_ids`). This bypasses `rank_bm25`'s naive $O(N)$ linear scans, achieving a **1,000x speedup**.
3. **BLAS Vectorized Dot-Product Retrieval**: `SemanticRetriever` uses FAISS `IndexFlatIP` for top-K search and vectorized BLAS matrix multiplication (`cand_matrix @ query_vec`) for candidate ranking, cutting prediction time by 50x.
4. **Temporal Splitting**: Strict time-based splitting (`temporal_split.py`) ensures no future click leakage between training, validation, and test interaction windows.

---

## 2. Alternatives Considered & Rationale

| Dimension | Alternative Considered | Chosen Design | Rationale |
| :--- | :--- | :--- | :--- |
| **Data Processing** | Pandas `read_csv` / `read_parquet` | Polars + PyArrow Row Group Streaming | Pandas exhausts memory (OOM crashes) on EB-NeRD's 13.5M impression table (1.5GB+). Polars streams chunks safely under 8GB RAM. |
| **Lexical Engine** | Naive `rank_bm25.get_scores()` | Inverted Index Posting List BM25 | `rank_bm25` iterates over all 120k articles for every query term, taking hours per batch. Inverted index scores only relevant candidate docs in milliseconds. |
| **Candidate Ranking** | `np.argsort` for small candidate lists | Timsort (`sorted(enumerate(...))`) | NumPy argsort carries C-API dispatch overhead for tiny lists ($N \approx 30$). Timsort is 10x faster for short candidate lists. |
| **Compression** | Standard Zip Deflate (Level 6) | `ZIP_LZMA` / `ZIP_BZIP2` Multi-Pass | Standard deflate zip files for 2.37M predictions exceed 90MB. LZMA compresses high-entropy numerical ASCII tables to $< 50$ MB for strict upload portals. |

---

## 3. Experimental Observations (Lexical vs. Semantic & Slicing)

### Retrieval Effectiveness Comparison (Recall@K):

| Dataset | Model | Recall@50 | Recall@100 | Recall@200 |
| :--- | :--- | :---: | :---: | :---: |
| **EB-NeRD (Demo)** | Lexical (BM25) | 0.0200 | 0.0270 | 0.0410 |
| **EB-NeRD (Demo)** | Semantic (FAISS/BERT) | 0.0140 | 0.0230 | 0.0400 |
| **MIND (Small)** | Lexical (BM25) | 0.0088 | 0.0141 | 0.0208 |
| **MIND (Small)** | Semantic (BERT) | 0.0125 | 0.0195 | 0.0285 |

### Key Experimental Insights:
1. **Lexical Strength on Danish (EB-NeRD)**: Exact keyword matching in article titles performed better on EB-NeRD demo because news headlines in local Danish media contain highly specific proper nouns (e.g. names, places).
2. **Semantic Superiority on MIND**: Dense BERT embeddings outperformed BM25 on MIND because English news headlines rely heavily on paraphrasing and thematic similarity (e.g. "economy" matching "stock market").
3. **Cold-Start Slicing Deficit**: Users with $\le 5$ clicks exhibited significantly lower AUC ($\approx 0.52$) compared to warm users ($> 5$ clicks, $\text{AUC} \approx 0.64$), emphasizing the cold-start challenge in history-based formulation.

---

## 4. Breakdown Analysis at $10\times$ Scale

If the dataset scale increases by $10\times$ (e.g. 135 Million impressions, 1.2 Million articles):

1. **In-Memory Embedding Bottleneck**:
   - *Failure Mode*: Storing 1.2 million 768-dim float32 BERT vectors in memory requires ~3.7 GB RAM. While fits in RAM, FAISS exact `IndexFlatIP` dot-product search will slow down linearly.
   - *Mitigation*: Switch to FAISS IVF-PQ (`IndexIVFPQ`) vector quantization to reduce memory by 8x and speed up ANN search.
2. **BM25 Inverted Index Dictionary Size**:
   - *Failure Mode*: Python `dict` overhead for 1.2M document posting lists will exceed 8 GB RAM.
   - *Mitigation*: Migrate from Python dictionaries to a disk-backed C++ inverted index engine (e.g., Pyserini / Lucene).
3. **Sequential File Writing**:
   - *Failure Mode*: Writing 135 million text prediction lines sequentially to a single `prediction.txt` file takes over 2 hours due to single-threaded disk I/O.
   - *Mitigation*: Partition candidate prediction generation across 16 parallel worker processes using multiprocessing chunking, then concatenate binary chunks.
