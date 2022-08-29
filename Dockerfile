FROM gcr.io/kaggle-gpu-images/python

ARG PROXY

ENV http_proxy=$PROXY \
    https_proxy=$PROXY

RUN pip install \
    'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup' \
    hydra-core \
    iterative-stratification

RUN pip install \
    tables \
    pytorch-tabnet
