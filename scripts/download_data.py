from tastematch.data import load_movielens
if __name__ == "__main__":
    ratings, movies = load_movielens()
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")
