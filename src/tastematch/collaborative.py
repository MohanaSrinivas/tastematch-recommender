import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemKNNRecommender:
    """Item-item collaborative filtering over a user-item rating matrix."""
    def fit(self, ratings: pd.DataFrame):
        self.ratings = ratings.copy()
        self.user_item = self.ratings.pivot_table(index="userId", columns="movieId", values="rating")
        filled = self.user_item.fillna(0)
        self.item_similarity = cosine_similarity(filled.T)
        self.movie_ids = self.user_item.columns.to_list()
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}
        self.user_index = {u: i for i, u in enumerate(self.user_item.index)}
        self.values = filled.to_numpy(dtype=float)
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if user_id not in self.user_index: raise KeyError(f"Unknown userId: {user_id}")
        row = self.values[self.user_index[user_id]]
        seen = row > 0
        scores = row @ self.item_similarity
        scores[seen] = -np.inf
        order = np.argsort(-scores)
        return [self.movie_ids[i] for i in order if np.isfinite(scores[i])][:k]
