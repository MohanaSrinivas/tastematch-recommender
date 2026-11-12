import pandas as pd

def temporal_user_split(ratings: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42):
    """Hold out each user's latest fraction by timestamp; users with <2 ratings stay train-only."""
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("userId"):
        ordered = group.sort_values("timestamp")
        if len(ordered) < 2:
            train_parts.append(ordered); continue
        n_test = max(1, int(round(len(ordered) * test_fraction)))
        n_test = min(n_test, len(ordered) - 1)
        train_parts.append(ordered.iloc[:-n_test]); test_parts.append(ordered.iloc[-n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test
