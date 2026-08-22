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
        print(f"Index built successfully over {len(corpus)} articles.")
        
    def formulate_query(self, history_article_ids: list) -> list:
        if not history_article_ids:
            return []
            
        query_text = ""
        for aid in history_article_ids:
            # history items might be None or invalid IDs sometimes
            if aid is None:
                continue
            title = self.article_titles.get(aid, "")
            query_text += f"{title} "
            
        return query_text.lower().split()
        
    def retrieve(self, history_article_ids: list, top_k: int = 200) -> list:
        query = self.formulate_query(history_article_ids)
        if not query:
            return []
            
        scores = self.bm25.get_scores(query)
        top_n_idx = np.argsort(scores)[::-1][:top_k]
        return [self.article_ids[i] for i in top_n_idx]

    def retrieve_batch(self, histories: list[list], top_k: int = 200) -> list[list]:
        results = []
        for history in histories:
            results.append(self.retrieve(history, top_k))
        return results
