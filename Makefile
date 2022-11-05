.PHONY: help preprocess postprocess
.DEFAULT_GOAL := help
SHELL = /bin/bash

NOW = $(shell date '+%Y%m%d-%H%M%S-%N')
GROUP := $(shell date '+%Y%m%d-%H%M')
HEAD_COMMIT = $(shell git rev-parse HEAD)


build: ## Build training container image.
	docker build --build-arg PROXY=$(http_proxy) -t kaggle-gpu-with-custom-packages - < Dockerfile

preprocess: ## Preprocess.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus '"device=2,3"' \
		-v $(shell pwd):/app -w /app/working \
		--shm-size=256g \
		kaggle-gpu-with-custom-packages \
		python preprocess.py

postprocess: ## Postprocess.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus '"device=2,3"' \
		-v $(shell pwd):/app -w /app/working \
		--shm-size=256g \
		kaggle-gpu-with-custom-packages \
		python postprocess.py

# --gpus '"device=0,1,2,3,6,7"'
train: ## Run training.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus '"device=6,7"' \
		-v ~/.netrc:/home/jupyter/.netrc \
		-v $(shell pwd):/app -w /app/working \
		--shm-size=256g \
		kaggle-gpu-with-custom-packages \
		python train.py  # +settings.run_fold=0

debug: ## Run training debug mode.
	docker run -d --rm -u $(shell id -u):$(shell id -g) --gpus '"device=1,2"' \
		-v $(shell pwd):/app -w /app/working \
		--shm-size=256g \
		kaggle-gpu-with-custom-packages \
		python train.py settings.debug=True hydra.verbose=True +settings.run_fold=0

early-stop: ## Abort training gracefully.
	@touch abort-training.flag

push: clean-build ## Publish notebook.
	@rm -f ./notebook/inference.ipynb
	@python encode.py ./working/src ./working/config
	# @cd ./notebook && kaggle kernels push

push-model: ## Publish models.
	@cd ./dataset/training && \
		kaggle datasets version -m $(HEAD_COMMIT)-$(NOW) -r zip

clean: clean-build clean-pyc clean-training ## Remove all build and python artifacts.

clean-build: ## Remove build artifacts.
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove python artifacts.
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-training: ## Remove training artifacts.
	@rm -rf ./output ./multirun abort-training.flag

clean-preprocess:  ## Remove preprocess artifacts.
	@find preprocess -type f -name *.f -exec rm -rf {} \;
	@find preprocess -type f -name *.npy -exec rm -rf {} \;
	@find preprocess -type f -name *.pkl -exec rm -rf {} \;
	@find preprocess -type f -name *.pickle -exec rm -rf {} \;
	@find preprocess -type f -name *.index -exec rm -rf {} \;

test: ## Run tests.
	@cd ./working && pytest

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
