import logging

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from .base import BaseTransformer

log = logging.getLogger(__name__)


class CustomScanPy(BaseTransformer):
    """ """

    def __init__(self, c):
        self.seed = c.global_params.seed
        self.n_components = 240
        self.columns = [f"scanpy_pca_{n}" for n in range(self.n_components)]

    def fit(self, df):
        ...

    def transform(self, df):
        adata = AnnData(df.to_numpy(), dtype="float32")
        adata.obs_names = df.index
        adata.var_names = df.columns

        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)

        sc.pp.filter_genes(adata, min_cells=300)

        adata.var["mt"] = adata.var_names.str.contains("_MT-")
        adata.var["mt_a"] = adata.var_names.str.contains("_MT-A")
        adata.var["mt_c"] = adata.var_names.str.contains("_MT-C")
        adata.var["mt_n"] = adata.var_names.str.contains("_MT-N")
        adata.var["mt_r"] = adata.var_names.str.contains("_MT-R")
        adata.var["mt_t"] = adata.var_names.str.contains("_MT-T")

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "mt_a", "mt_c", "mt_n", "mt_r", "mt_t"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var["highly_variable"]]

        sc.pp.scale(adata, max_value=10)

        sc.pp.pca(adata, n_comps=self.n_components, random_state=self.seed)
        log.info(f"scanpy pca result: {type(adata.obsm['X_pca'])}, shape: {adata.obsm['X_pca'].shape}")

        # df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names, columns=self.columns)
        return df
