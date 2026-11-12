# TasteMatch Recommender

[![CI](https://github.com/MohanaSrinivas/tastematch-recommender/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MohanaSrinivas/tastematch-recommender/actions/workflows/ci.yml)

A reproducible movie-recommendation baseline built on GroupLens MovieLens latest-small. The project compares genre-based content recommendations with item-item collaborative filtering, evaluates ranked results on a per-user temporal holdout, and exposes both recommenders through a small FastAPI service.

**Portfolio scope:** educational baseline for reproducible experiments; no hosted API or production model service is provided.

## What it demonstrates

- Download the public MovieLens latest-small data on demand instead of committing the dataset.
- Build genre TF-IDF/cosine content recommendations and item-item cosine recommendations from ratings.
- Hold out each eligible user’s latest timestamped interactions and report macro Precision@K, Recall@K, and NDCG@K.
- Run the recommendation API and unit tests without downloading the dataset; the API downloads and fits data-backed models when a recommendation endpoint is first called.

## Architecture

```mermaid
flowchart LR
  Dataset[MovieLens latest-small<br/>movies.csv and ratings.csv] --> Content[Genre TF-IDF<br/>content recommender]
  Dataset --> Split[Per-user temporal split]
  Split -->|train ratings| EvalItem[Item-item model<br/>for evaluation]
  Split -->|held-out ratings| Eval[Precision@K<br/>Recall@K<br/>NDCG@K]
  Content --> Eval
  EvalItem --> Eval
  Dataset --> Runtime[First recommendation request<br/>load data and fit runtime models]
  Content --> Runtime
  Runtime --> API[FastAPI endpoints]
  API --> Client[HTTP client]
```

The content model uses movie genres. The collaborative model uses explicit user ratings. In evaluation, each user’s latest 20% of ratings (at least one, while retaining a training rating) are held out. Ratings of 4.0 or more count as relevant. Users with fewer than two ratings remain train-only. These are project choices, not universal evaluation standards.

## Data

- Source: [GroupLens MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/)
- Download: [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- Files used: `movies.csv` and `ratings.csv`
- The data is downloaded on demand to `data/raw/` and is not committed. Review GroupLens’ terms and credit GroupLens when presenting results.

## Setup

Requirements: Python 3.11 or newer. From the repository root, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Run the API

```bash
PYTHONPATH=src uvicorn tastematch.api:app --reload --port 8001
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn tastematch.api:app --reload --port 8001
```

Open [API docs](http://localhost:8001/docs) or [health](http://localhost:8001/health). Example requests:

- `GET /recommend/content?movie_id=1&k=10`
- `GET /recommend/collaborative?user_id=1&k=10`

The health route does not load MovieLens. The first recommendation request downloads and fits the models, so it needs network access and can take longer than later requests. The content endpoint expects a movie ID in the dataset; the collaborative endpoint expects a user ID in the ratings.

### Run the analysis

From the repository root, with the virtual environment active:

```bash
PYTHONPATH=src python notebooks/analysis.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python notebooks/analysis.py
```

This downloads the dataset if it is absent, then prints dataset/split sizes and evaluation metrics. No benchmark results are hard-coded in the README.

### Run with Docker

Requirements: Docker Desktop or Docker Engine with Compose v2.

```bash
docker compose up --build
```

The API is available at [http://localhost:8001](http://localhost:8001). The first recommendation request downloads the dataset into the mounted `data/` directory. Stop the container with `docker compose down`; the downloaded local data remains in that directory.

## Tests and CI

```bash
ruff check src tests
PYTHONPATH=src pytest -q
```

PowerShell:

```powershell
ruff check src tests
$env:PYTHONPATH = "src"
python -m pytest -q
```

Unit tests use small synthetic examples and do not download MovieLens. GitHub Actions installs the pinned requirements, runs Ruff, and runs the same tests. The pytest configuration adds `src/` to the import path, matching the documented command and CI behavior.

## Evaluation and limitations

The tests check split behavior, recommendation behavior on small examples, API health, and metric arithmetic. They do not establish recommendation quality on the full dataset. Run the analysis before quoting results and report the data version, split policy, relevance threshold, and K value alongside any metrics.

- Content similarity uses genres only; it does not use plot, cast, or semantic features.
- Collaborative filtering builds a dense item-similarity matrix and does not scale to a large catalog as written.
- New users without rating history have no collaborative recommendations.
- The temporal split can contain cold-start items; recommendations are limited to items represented in the training data.
- There is no authentication, request limiting, online feedback, or ranking experiment infrastructure.

## Project layout

- `src/tastematch/` — data loading, recommenders, temporal split, evaluation, and API.
- `tests/` — synthetic unit tests and API health test.
- `notebooks/analysis.py` — repeatable analysis entry point.
- `docs/evaluation.md` — evaluation definitions and caveats.
- `data/` — downloaded data location; dataset files are not tracked.
