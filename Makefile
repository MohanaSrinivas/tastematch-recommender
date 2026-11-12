.PHONY: install test lint api analysis
install:
	python -m pip install -r requirements.txt
test:
	PYTHONPATH=src pytest -q
lint:
	ruff check src tests
api:
	PYTHONPATH=src uvicorn tastematch.api:app --reload --port 8001
analysis:
	PYTHONPATH=src python notebooks/analysis.py
