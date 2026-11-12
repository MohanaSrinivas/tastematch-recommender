import math
from collections.abc import Callable, Iterable

def precision_recall_ndcg(recommended: Iterable[int], relevant: set[int], k: int) -> dict[str, float]:
    recs = list(recommended)[:k]
    hits = [1 if item in relevant else 0 for item in recs]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return {"precision_at_k": precision, "recall_at_k": recall, "ndcg_at_k": dcg / ideal if ideal else 0.0}

def evaluate_users(test_ratings, recommend: Callable[[int, int], list[int]], k: int = 10, threshold: float = 4.0):
    """Macro-average ranking metrics over test users with at least one relevant item."""
    results = []
    for user_id, group in test_ratings.groupby("userId"):
        relevant = set(group.loc[group.rating >= threshold, "movieId"])
        if not relevant: continue
        results.append(precision_recall_ndcg(recommend(int(user_id), k), relevant, k))
    if not results: return {"users_evaluated": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}
    return {"users_evaluated": len(results), **{key: sum(row[key] for row in results)/len(results) for key in results[0]}}
