import gc
import logging
import os

import faiss
import numpy as np
import pandas as pd

from .base import BaseTransformer

log = logging.getLogger(__name__)


class FaissKNeighbors(BaseTransformer):
    """
    https://gist.github.com/j-adamczyk/74ee808ffd53cd8545a49f185a908584
    https://www.ariseanalytics.com/activities/report/20210304/
    https://rest-term.com/archives/3414/
    """

    def __init__(self, c):
        # if c.global_params.data == "cite":
        #     self.n_components = c.preprocess_params.faiss_n_components_cite
        # elif c.global_params.data == "multi":
        #     self.n_components = c.preprocess_params.faiss_n_components_multi
        # else:
        #     raise Exception(f"Invalid data. {c.global_params.data}")

        self.index = None
        self.dim = 5
        self.k = 3

        self.save_dir = c.settings.dirs.preprocess
        self.c = c

    def fit(self, df):
        res = faiss.StandardGpuResources()

        config = faiss.GpuIndexFlatConfig()

        # df = df.loc[:, (df != 0).any(axis=0)]
        X = df.to_numpy()
        self.dim = X.shape[1]

        self.index = faiss.GpuIndexFlatL2(res, self.dim, config)
        self.index.train(np.ascontiguousarray(X, dtype=np.float32))
        self.index.add(np.ascontiguousarray(X, dtype=np.float32))

    def kneighbors(self, X, k: int = 2, nprobe: int = 100, return_distance: bool = True):
        if k <= 0:
            k = self.k

        self.index.nprobe = nprobe

        distances, indices = self.index.search(np.ascontiguousarray(X, dtype=np.float32), k=k)

        if return_distance:
            return distances, indices
        else:
            return indices

    def save(self, filename):
        index_cpu = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_cpu, os.path.join(self.save_dir, filename))

    def load(self, path):
        index_cpu = faiss.read_index(path)

        try:
            devices = self.c.settings.gpus.split(",")
            resources = [faiss.StandardGpuResources() for _ in devices]
            self.index = faiss.index_cpu_to_gpu_multiple_py(resources, index_cpu, gpus=devices)
        except Exception:
            self.index = faiss.index_cpu_to_all_gpus(index_cpu)
