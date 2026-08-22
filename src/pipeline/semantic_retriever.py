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
