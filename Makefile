.PHONY: setup download prepare inventory features embeddings clustering fewshot demo clean all

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

clustering:
	uv run scripts/run_clustering_benchmark.py --n_jobs -1

fewshot:
	uv run scripts/run_fewshot_benchmark.py --n_jobs -1

demo:
	uv run streamlit run scripts/run_demo_app.py

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results/step0/*
	rm -rf results/step1/*
	rm -rf results/step2/*
	rm -rf results/step3/*
	rm -rf results/step4/*
	rm -rf results/step6/*
	rm -rf .venv

all: setup download inventory features embeddings clustering fewshot demo
report:
	@echo "Generating synthesis reports..."
	uv run python scripts/build_report.py
all: setup simulate run_all report
