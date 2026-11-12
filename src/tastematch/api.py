from fastapi import FastAPI, HTTPException, Query
from .service import models

app = FastAPI(title="TasteMatch API", version="1.0.0", description="Movie recommendation demo using MovieLens latest-small.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/content")
def content(movie_id: int, k: int = Query(10, ge=1, le=50)):
    try:
        _, movies, model, _ = models()
        ids = model.recommend(movie_id, k)
        names = movies.set_index("movieId").reindex(ids)["title"].fillna("Unknown").tolist()
        return {"movie_id": movie_id, "recommendations": [{"movie_id": i, "title": n} for i, n in zip(ids, names)]}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Dataset unavailable; see README to download MovieLens") from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

@app.get("/recommend/collaborative")
def collaborative(user_id: int, k: int = Query(10, ge=1, le=50)):
    try:
        _, movies, _, model = models()
        ids = model.recommend(user_id, k)
        names = movies.set_index("movieId").reindex(ids)["title"].fillna("Unknown").tolist()
        return {"user_id": user_id, "recommendations": [{"movie_id": i, "title": n} for i, n in zip(ids, names)]}
    except FileNotFoundError as exc:
        raise HTTPException(503, "Dataset unavailable; see README to download MovieLens") from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
