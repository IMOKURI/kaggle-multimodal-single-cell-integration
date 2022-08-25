# https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart

import logging
import gc

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from .base import BaseTransformer

log = logging.getLogger(__name__)


class CustomPCA(BaseTransformer):
    """ """

    def __init__(self, c, kind):
        self.seed = c.global_params.seed
        if kind == "cite":
            self.n_components = c.preprocess_params.pca_n_components_cite
        elif kind == "multiome":
            self.n_components = c.preprocess_params.pca_n_components_multi
        else:
            raise Exception(f"Invalid kind. kind: {kind}")
        self.columns = [f"pca_{n}" for n in range(self.n_components)]
        self.pca = PCA(n_components=self.n_components, copy=False, random_state=self.seed)

    def fit(self, df):
        # df = df.loc[:, (df != 0).any(axis=0)]
        X = df.to_numpy()

        self.pca.fit(X)

    def transform(self, df):
        # df = df.loc[:, (df != 0).any(axis=0)]
        X = df.to_numpy()

        X = self.pca.transform(X)

        df = pd.DataFrame(X, columns=self.columns)
        return df
