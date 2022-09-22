# https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart

import gc
import logging

import numpy as np
import pandas as pd
from ivis import Ivis

from .base import BaseTransformer

log = logging.getLogger(__name__)


class CustomIvis(BaseTransformer):
    """
    https://bering-ivis.readthedocs.io/en/latest/supervised.html
    """

    def __init__(self, c):
        if c.global_params.data == "cite":
            self.n_components = c.preprocess_params.ivis_n_components_cite
        elif c.global_params.data == "multi":
            self.n_components = c.preprocess_params.ivis_n_components_multi
        else:
            raise Exception(f"Invalid data. {c.global_params.data}")
        self.columns = [f"ivis_{n}" for n in range(self.n_components)]
        self.ivis = Ivis(
            embedding_dims=self.n_components,
            k=5,
            n_epochs_without_progress=10,
            batch_size=1024,
            supervision_metric="mse",
        )

    def fit(self, df, label=None):
        # df = df.loc[:, (df != 0).any(axis=0)]
        X = df.to_numpy()

        if label is None:
            self.ivis.fit(X)

        else:
            y = label.to_numpy()
            self.ivis.fit(X, y)

    def transform(self, df):
        # df = df.loc[:, (df != 0).any(axis=0)]
        X = df.to_numpy()

        X = self.ivis.transform(X)

        df = pd.DataFrame(X, columns=self.columns)
        return df
