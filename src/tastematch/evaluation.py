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

def evaluate_content_users(train_ratings, test_ratings, recommend: Callable[[int, int], list[int]], k: int = 10, threshold: float = 4.0):
    """Evaluate item-to-item content recommendations from a known liked train item.

    The most recently rated training item at or above ``threshold`` is the query.
    Users without a positive training seed are omitted, as are users without a
    relevant held-out item (the latter is handled by ``evaluate_users``).
    """
    seeds = {}
    liked = train_ratings.loc[train_ratings.rating >= threshold]
    for user_id, group in liked.groupby("userId"):
        latest = group.sort_values("timestamp").iloc[-1]
        seeds[int(user_id)] = int(latest.movieId)
    eligible_test = test_ratings[test_ratings.userId.isin(seeds)]
    return evaluate_users(
        eligible_test,
        lambda user_id, limit: recommend(seeds[int(user_id)], limit),
        k=k,
        threshold=threshold,
    )
