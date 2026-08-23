import polars as pl
import numpy as np
import faiss

class SemanticRetriever:
    def __init__(self, df_articles: pl.DataFrame):
        self.df_articles = df_articles
        
        if "bert" not in df_articles.columns:
            raise ValueError("Embeddings ('bert' column) missing from articles.parquet. Run `make embed` first.")
            
        print("Extracting and normalizing embeddings...")
        
        self.article_embeddings = {}
        for row in df_articles.iter_rows(named=True):
            emb = row.get("bert")
            if emb is not None:
                self.article_embeddings[row["article_id"]] = np.array(emb, dtype=np.float32)
                
        self.article_ids = list(self.article_embeddings.keys())
        
        embeddings_matrix = np.vstack(list(self.article_embeddings.values()))
        
        # L2 normalize embeddings for cosine similarity search using Inner Product
        faiss.normalize_L2(embeddings_matrix)
        
        # Update dictionary with normalized vectors so we don't have to normalize later
        for i, aid in enumerate(self.article_ids):
            self.article_embeddings[aid] = embeddings_matrix[i]

        
        d = embeddings_matrix.shape[1]
        print(f"Building FAISS IndexFlatIP (dim={d}) over {len(self.article_ids)} articles...")
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings_matrix)
        print("FAISS index built successfully.")
        
    def formulate_query(self, history_article_ids: list, max_history: int = 5) -> np.ndarray:
        if not history_article_ids:
            return None
            
        # History is ordered chronologically, so we take the last `max_history` items
        recent_history = history_article_ids[-max_history:]
        
        history_embs = []
        for aid in recent_history:
            if aid in self.article_embeddings:
                history_embs.append(self.article_embeddings[aid])
                
        if not history_embs:
            return None
            
        # Mean pooling
        mean_emb = np.mean(history_embs, axis=0)
        # Reshape to (1, D) for FAISS and normalize
        mean_emb = mean_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(mean_emb)
        return mean_emb
        
    def retrieve(self, history_article_ids: list, top_k: int = 200) -> list:
        query_emb = self.formulate_query(history_article_ids)
        if query_emb is None:
            return []
            
        scores, indices = self.index.search(query_emb, top_k)
        
        return [self.article_ids[idx] for idx in indices[0]]

    def score_candidates(self, history_article_ids: list, candidate_article_ids: list) -> list[float]:
        """
        Scores specific candidate articles for a given history query.
        Returns a list of cosine similarity scores in the same order as candidate_article_ids.
        """
        query_emb = self.formulate_query(history_article_ids)
        if query_emb is None:
            return [0.0] * len(candidate_article_ids)
            
        # query_emb is (1, D), L2-normalized.
        query_vec = query_emb[0]
        
        # Build matrix of candidate embeddings
        valid_indices = []
        valid_vecs = []
        for i, cid in enumerate(candidate_article_ids):
            cand_vec = self.article_embeddings.get(cid)
            if cand_vec is not None:
                valid_indices.append(i)
                valid_vecs.append(cand_vec)
                
        scores = [0.0] * len(candidate_article_ids)
        if valid_vecs:
            cand_matrix = np.vstack(valid_vecs)
            dot_products = cand_matrix @ query_vec
            for idx, score in zip(valid_indices, dot_products):
                scores[idx] = float(score)
                
        return scores
