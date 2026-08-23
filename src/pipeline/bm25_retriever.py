import polars as pl
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, df_articles: pl.DataFrame):
        self.df_articles = df_articles
        # Create an article_id to title mapping for fast query formulation
        self.article_titles = dict(zip(
            df_articles["article_id"].to_list(),
            df_articles["title"].fill_null("").to_list()
        ))
        
        # Build index from title + abstract/subtitle
        # EB-NeRD uses 'subtitle', MIND uses 'abstract'
        if "subtitle" in df_articles.columns:
            abstract_col = "subtitle"
        else:
            abstract_col = "abstract"
            
        print("Building BM25 Index (this may take a minute)...")
        corpus = []
        self.article_ids = []
        
        # Using polars to concatenate and lower strings efficiently
        df_docs = df_articles.select([
            "article_id",
            pl.concat_str([
                pl.col("title").fill_null(""), 
                pl.lit(" "), 
                pl.col(abstract_col).fill_null("")
            ]).str.to_lowercase().alias("text")
        ])
        
        for row in df_docs.iter_rows(named=True):
            text = row["text"]
            tokens = text.split()
            corpus.append(tokens)
            self.article_ids.append(row["article_id"])
            
        self.bm25 = BM25Okapi(corpus)
        # Create map from article_id to its index in corpus
        self.article_id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        
        # Precompute the BM25 document length denominator term to save math operations in the inner loop
        doc_len = np.array(self.bm25.doc_len)
        self.doc_term = self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * doc_len / self.bm25.avgdl)
        self.k1_plus_1 = self.bm25.k1 + 1
        
        print("Building inverted index for ultra-fast search...")
        self.inverted_index = {}
        for idx, doc_freq in enumerate(self.bm25.doc_freqs):
            for word, count in doc_freq.items():
                if word not in self.inverted_index:
                    self.inverted_index[word] = {}
                self.inverted_index[word][idx] = count
                
        print(f"Index built successfully over {len(corpus)} articles.")
        
    def formulate_query(self, history_article_ids: list, max_history: int = 20) -> list:
        if not history_article_ids:
            return []
            
        query_text = ""
        # Use only the most recent articles
        for aid in history_article_ids[-max_history:]:
            # history items might be None or invalid IDs sometimes
            if aid is None:
                continue
            title = self.article_titles.get(aid, "")
            query_text += f"{title} "
            
        # Deduplicating tokens speeds up rank_bm25 massively since it avoids 
        # repeating the math over the whole 120k document index for repeated words
        return list(set(query_text.lower().split()))
        
    def retrieve(self, history_article_ids: list, top_k: int = 200) -> list:
        query = self.formulate_query(history_article_ids)
        if not query:
            return []
            
        # Ultra-fast inverted index scoring instead of rank_bm25's slow O(N) scoring
        scores = np.zeros(self.bm25.corpus_size)
        doc_len = np.array(self.bm25.doc_len)
        
        for q in query:
            if q not in self.bm25.idf:
                continue
            idf = self.bm25.idf[q]
            posting_list = self.inverted_index.get(q, {})
            
            for idx, q_freq in posting_list.items():
                scores[idx] += idf * (q_freq * self.k1_plus_1 / (q_freq + self.doc_term[idx]))
                                 
        top_n_idx = np.argsort(scores)[::-1][:top_k]
        return [self.article_ids[i] for i in top_n_idx]

    def retrieve_batch(self, histories: list[list], top_k: int = 200) -> list[list]:
        results = []
        for history in histories:
            results.append(self.retrieve(history, top_k))
        return results

    def score_candidates(self, history_article_ids: list, candidate_article_ids: list) -> list[float]:
        """
        Scores specific candidate articles for a given history query.
        Returns a list of BM25 scores in the same order as candidate_article_ids.
        """
        query = self.formulate_query(history_article_ids)
        if not query:
            return [0.0] * len(candidate_article_ids)
            
        candidate_indices = []
        for cid in candidate_article_ids:
            idx = self.article_id_to_idx.get(cid)
            candidate_indices.append(idx)
            
        scores = np.zeros(len(candidate_indices))
        
        # Valid candidates that are in the index
        valid_mask = [i is not None for i in candidate_indices]
        valid_indices = [idx for idx in candidate_indices if idx is not None]
        
        if not valid_indices:
            return scores.tolist()
            
        doc_term_valid = self.doc_term[valid_indices]
        
        valid_scores = np.zeros(len(valid_indices))
        for q in query:
            if q not in self.bm25.idf:
                continue
            idf = self.bm25.idf[q]
            q_freq = np.array([self.bm25.doc_freqs[i].get(q, 0) for i in valid_indices])
            
            valid_scores += idf * (q_freq * self.k1_plus_1 / (q_freq + doc_term_valid))
                             
        # Map valid scores back to their original positions
        valid_idx_ptr = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                scores[i] = valid_scores[valid_idx_ptr]
                valid_idx_ptr += 1
                
        return scores.tolist()
