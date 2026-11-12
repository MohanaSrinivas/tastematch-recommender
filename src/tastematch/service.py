from functools import lru_cache
from .data import load_movielens
from .content import ContentRecommender
from .collaborative import ItemKNNRecommender

@lru_cache(maxsize=1)
def models():
    ratings, movies = load_movielens()
    content = ContentRecommender().fit(movies)
    collaborative = ItemKNNRecommender().fit(ratings)
    return ratings, movies, content, collaborative
