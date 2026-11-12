import pandas as pd
from tastematch.split import temporal_user_split

def test_temporal_split_keeps_latest_event_in_test_and_train_history():
    data = pd.DataFrame({"userId":[1,1,1,2],"movieId":[1,2,3,4],"rating":[3,4,5,4],"timestamp":[10,20,30,15]})
    train,test = temporal_user_split(data, test_fraction=.34)
    assert set(test[test.userId == 1].movieId) == {3}
    assert set(train[train.userId == 1].movieId) == {1,2}
    assert 4 in set(train.movieId)  # single-interaction users remain train-only
