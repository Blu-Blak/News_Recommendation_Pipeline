.PHONY: install data run evaluate test clean submission

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install ../ebnerd-benchmark/

data:
	python src/pipeline/build_pipeline.py --dataset mind
	python src/pipeline/build_pipeline.py --dataset ebnerd --scale demo

test:
	pytest src/tests/ -v

evaluate:
	python src/pipeline/evaluate_recall.py --dataset all

clean:
	rm -rf data/ outputs/
