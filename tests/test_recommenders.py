import pandas as pd
import pytest
from tastematch.content import ContentRecommender
from tastematch.collaborative import ItemKNNRecommender
from tastematch.evaluation import precision_recall_ndcg

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
