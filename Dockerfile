FROM gcr.io/kaggle-gpu-images/python:v122

ARG PROXY

ENV http_proxy=$PROXY \
    https_proxy=$PROXY

RUN pip install \
    hydra-core \
    iterative-stratification

# RUN conda install -c pytorch faiss-gpu

RUN pip install \
    tables \
    pytorch-tabnet \
    ivis[gpu] \
    scanpy
