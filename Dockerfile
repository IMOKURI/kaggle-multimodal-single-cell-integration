FROM gcr.io/kaggle-gpu-images/python:v120

ARG PROXY

ENV http_proxy=$PROXY \
    https_proxy=$PROXY

RUN pip install \
    'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup' \
    hydra-core \
    iterative-stratification

RUN conda install -c pytorch faiss-gpu

RUN pip install \
    tables \
    pytorch-tabnet \
    ivis[gpu]
