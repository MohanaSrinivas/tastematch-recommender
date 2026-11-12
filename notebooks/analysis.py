"""Notebook companion analysis, executable as a script or imported in Jupyter."""
from tastematch.data import load_movielens
from tastematch.split import temporal_user_split
from tastematch.content import ContentRecommender
from tastematch.collaborative import ItemKNNRecommender
from tastematch.evaluation import evaluate_content_users, evaluate_users

def run():
    ratings, movies = load_movielens()
    train, test = temporal_user_split(ratings)
    content = ContentRecommender().fit(movies)
    collaborative = ItemKNNRecommender().fit(train)
    print("ratings:", len(ratings), "train:", len(train), "test:", len(test), "movies:", len(movies))
    print("content-based:", evaluate_content_users(train, test, content.recommend))
    print("collaborative:", evaluate_users(test, collaborative.recommend))
if __name__ == "__main__":
    run()
