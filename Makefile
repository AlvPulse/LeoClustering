.PHONY: setup download prepare inventory features embeddings clean all

setup:
	uv sync

download:
	uv run scripts/download.py

prepare:
	uv run scripts/prepare.py

inventory: prepare
	uv run scripts/inventory.py

features:
	uv run scripts/extract_physics_features.py

embeddings:
	uv run scripts/extract_embeddings.py --model all

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results/step0/*
	rm -rf results/step1/*
	rm -rf results/step2/*
	rm -rf .venv

all: setup download inventory features embeddings
