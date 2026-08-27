.PHONY: install data run evaluate test clean submission evaluate-ablation

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install ../ebnerd-benchmark/

data:
	python src/pipeline/build_pipeline.py --dataset mind
	python src/pipeline/build_pipeline.py --dataset ebnerd --scale demo
	python src/pipeline/build_pipeline.py --dataset ebnerd --scale small

test:
	pytest src/tests/ -v

quick-test:
	@echo "--- QUICK EVALUATION (BM25) ---"
	python src/pipeline/evaluate_recall.py --dataset all --retriever bm25 --limit 1000
	@echo "\n--- QUICK EVALUATION (SEMANTIC) ---"
	python src/pipeline/evaluate_recall.py --dataset all --retriever semantic --limit 1000

evaluate:
	python src/pipeline/evaluate_recall.py --dataset all --retriever bm25

embed:
	python src/pipeline/embedding_utils.py

evaluate-semantic:
	python src/pipeline/evaluate_recall.py --dataset all --retriever semantic

evaluate-harness:
	python src/pipeline/evaluate_harness.py --dataset all --retriever all

evaluate-ablation:
	python src/pipeline/evaluate_harness.py --dataset all --retriever all --ablation

data-large:
	python src/pipeline/build_pipeline.py --dataset mind --include-test
	python src/pipeline/build_pipeline.py --dataset ebnerd --scale demo --include-test

submission-mind-bm25:
	python src/pipeline/generate_predictions.py --dataset mind --retriever bm25

submission-mind-semantic:
	python src/pipeline/generate_predictions.py --dataset mind --retriever semantic

submission-mind: submission-mind-bm25 submission-mind-semantic

submission-ebnerd-bm25:
	python src/pipeline/generate_predictions.py --dataset ebnerd --retriever bm25

submission-ebnerd-semantic:
	python src/pipeline/generate_predictions.py --dataset ebnerd --retriever semantic

submission-ebnerd: submission-ebnerd-bm25 submission-ebnerd-semantic

submission: submission-mind submission-ebnerd

sbatch:
	sbatch run_submission.sbatch

clean:
	rm -rf data/ outputs/
