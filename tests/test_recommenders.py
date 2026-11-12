import pandas as pd
import pytest
from tastematch.content import ContentRecommender
from tastematch.collaborative import ItemKNNRecommender
from tastematch.evaluation import evaluate_content_users, precision_recall_ndcg

def test_content_recommender_returns_other_similar_items():
    movies = pd.DataFrame({"movieId":[1,2,3],"genres":["Action|Adventure","Action|Adventure","Romance"]})
    model = ContentRecommender().fit(movies)
    result = model.recommend(1, 2)
    assert 1 not in result
    assert result[0] == 2

def test_item_knn_excludes_seen_items():
    ratings = pd.DataFrame({"userId":[1,1,2,2,3,3],"movieId":[10,20,10,30,20,30],"rating":[5,4,4,5,4,5]})
    model = ItemKNNRecommender().fit(ratings)
    assert 10 not in model.recommend(1, 2)

def test_ranking_metrics_hand_calculated():
    metrics = precision_recall_ndcg([10,20,30], {20,30}, 3)
    assert metrics["precision_at_k"] == pytest.approx(2/3)
    assert metrics["recall_at_k"] == 1.0
    assert 0 < metrics["ndcg_at_k"] <= 1

def test_content_evaluation_uses_latest_liked_training_movie_as_query():
    train = pd.DataFrame({"userId":[1,1,1,2],"movieId":[10,11,20,30],"rating":[5,4,2,2],"timestamp":[1,2,3,1]})
    test = pd.DataFrame({"userId":[1,2],"movieId":[40,50],"rating":[5,5],"timestamp":[4,2]})
    queries = []

    def recommend(movie_id, k):
        queries.append(movie_id)
        return [40]

    metrics = evaluate_content_users(train, test, recommend, k=1)
    assert queries == [11]
    assert metrics["users_evaluated"] == 1
    assert metrics["precision_at_k"] == 1.0
