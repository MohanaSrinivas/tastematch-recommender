import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    """Recommend movies with similar genre text using TF-IDF cosine similarity."""
    def fit(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform(self.movies["genres"].fillna("").str.replace("|", " ", regex=False))
        self.ids = self.movies["movieId"].tolist()
        self.id_to_idx = {movie_id: i for i, movie_id in enumerate(self.ids)}
        self.similarity = cosine_similarity(self.matrix, dense_output=False)
        return self

    def recommend(self, movie_id: int, k: int = 10) -> list[int]:
        if movie_id not in self.id_to_idx: raise KeyError(f"Unknown movieId: {movie_id}")
        idx = self.id_to_idx[movie_id]
        scores = self.similarity[idx].toarray().ravel()
        order = np.argsort(-scores)
        return [self.ids[i] for i in order if i != idx][:k]
