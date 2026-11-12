from pathlib import Path
import zipfile
import urllib.request
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def load_movielens(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load MovieLens latest-small; data files are not committed."""
    root = Path(data_dir); root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-latest-small.zip"
    if not archive.exists():
        request = urllib.request.Request(MOVIELENS_URL, headers={"User-Agent": "TasteMatch-portfolio/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
            out.write(response.read())
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    folder = root / "ml-latest-small"
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv")
    return ratings, movies
