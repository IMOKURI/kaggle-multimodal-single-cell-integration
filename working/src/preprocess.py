# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import gc
import logging
import os
import pickle
import re
from functools import wraps
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .preprocesses.p001_dist_transformer import DistTransformer
from .preprocesses.p010_pca import CustomPCA
from .preprocesses.p011_ivis import CustomIvis

# from .preprocesses.p012_faiss import FaissKNeighbors
from .preprocesses.p020_scanpy import CustomScanPy

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame, stem: str) -> pd.DataFrame:
    # Convert None to NaN
    df = df.fillna(np.nan)

    ...

    # Convert None to NaN
    df = df.fillna(np.nan)

    return df


def preprocess_train_test(
    c, train_df: pd.DataFrame, test_df: pd.DataFrame, label_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = len(train_df)
    train_index = train_df.index
    test_index = test_df.index
    method = ""

    df = pd.concat([train_df, test_df])
    log.info(f"Shape before preprocess: {df.shape}")

    # if "neighbors" in c.preprocess_params.methods:
    #     method += "neighbors_"
    #     neighbor_df = pd.DataFrame(index=df.index)
    #     for n_col in range(len(df.columns) - 2):
    #         neighbor_df = pd.concat([neighbor_df, df.iloc[:, n_col : n_col + 3].mean(axis=1)], axis=1)
    #
    #     neighbor_df.columns = [f"neighbor_{n}" for n in range(len(df.columns) - 2)]
    #     df = neighbor_df

    # 列の中に1つの値しかない列は削除
    df = df.loc[:, df.nunique() != 1]

    if c.preprocess_params.cols in ["GL", "KI"]:
        log.info("Skip preprocess.")
        preprocessor = CustomPCA(c)

    elif "scanpy" in c.preprocess_params.methods:
        method += "scanpy"
        preprocessor = CustomScanPy(c)
        df = transform_data(
            c,
            f"{c.global_params.data}_{c.preprocess_params.cols}_{method}.pickle",
            df,
            preprocessor,
        )

    elif "pca" in c.preprocess_params.methods:
        method += "pca"
        preprocessor = CustomPCA(c)
        df = transform_data(
            c,
            f"{c.global_params.data}_{c.preprocess_params.cols}_{method}_{preprocessor.n_components}.pickle",
            df,
            preprocessor,
        )

    elif "ivis" in c.preprocess_params.methods:
        method += "ivis"
        preprocessor = DistTransformer(transform="min-max")
        df = transform_data(
            c,
            f"{c.global_params.data}_{c.preprocess_params.cols}_minmax.pickle",
            df,
            preprocessor,
        )
        preprocessor = CustomIvis(c)
        df = transform_data(
            c,
            f"{c.global_params.data}_{c.preprocess_params.cols}_{method}_{preprocessor.n_components}.pickle",
            df,
            preprocessor,
        )

    elif "ivis_supervised" in c.preprocess_params.methods:
        method += "ivis_supervised"
        preprocessor = DistTransformer(transform="min-max")
        df = transform_data(
            c,
            f"{c.global_params.data}_{c.preprocess_params.cols}_minmax.pickle",
            df,
            preprocessor,
        )

        train_df = df.iloc[:train_size, :]
        test_df = df.iloc[train_size:, :]

        preprocessor = CustomIvis(c)
        train_df = transform_data(
            c,
            f"train-{c.global_params.data}_{c.preprocess_params.cols}_{method}_{preprocessor.n_components}.pickle",
            train_df,
            preprocessor,
            label_df,
        )
        test_df = transform_data(
            c,
            f"test-{c.global_params.data}_{c.preprocess_params.cols}_{method}_{preprocessor.n_components}.pickle",
            test_df,
            preprocessor,
        )

        df = pd.concat([train_df, test_df])

    else:
        raise Exception(f"Invalid preprocess method.")

    # if "faiss" in c.preprocess_params.methods:
    #     log.info(f"faiss fit data: {df.shape}")
    #     index = FaissKNeighbors(c)
    #     index.fit(df)
    #     index.save(f"{c.global_params.data}_{c.preprocess_params.cols}_faiss_{index.dim}.index")

    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]

    train_df.index = train_index
    test_df.index = test_index

    train_df.index.name = "cell_id"
    test_df.index.name = "cell_id"

    train_df.to_pickle(
        os.path.join(
            c.settings.dirs.preprocess,
            f"train_{c.global_params.data}_{c.preprocess_params.cols}_inputs_{method}_{preprocessor.n_components}{'_raw' if c.preprocess_params.use_raw_data else ''}.pickle",
        )
    )
    test_df.to_pickle(
        os.path.join(
            c.settings.dirs.preprocess,
            f"test_{c.global_params.data}_{c.preprocess_params.cols}_inputs_{method}_{preprocessor.n_components}{'_raw' if c.preprocess_params.use_raw_data else ''}.pickle",
        )
    )

    return train_df, test_df


def load_or_fit(func: Callable):
    """
    前処理を行うクラスがすでに保存されていれば、それをロードする。
    保存されていなければ、 func で生成、学習する。
    与えられたデータを、学習済みクラスで前処理する。

    Args:
        func (Callable): 前処理を行うクラスのインスタンスを生成し、学習する関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1]) if args[1] is not None else None

        if path is not None and os.path.exists(path):
            instance = pickle.load(open(path, "rb"))

        else:
            instance = func(*args, **kwargs)

            if path is not None:
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                pickle.dump(instance, open(path, "wb"), protocol=4)

        return instance

    return wrapper


def load_or_transform(func: Callable):
    """
    前処理されたデータがすでに存在すれば、それをロードする。
    存在しなければ、 func で生成する。生成したデータは保存しておく。

    Args:
        func (Callable): 前処理を行う関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1])

        if os.path.exists(path) and os.path.splitext(path)[1] == ".npy":
            array = np.load(path, allow_pickle=True)
        elif os.path.exists(path) and os.path.splitext(path)[1] == ".pickle":
            array = pd.read_pickle(path)

        else:
            array = func(*args, **kwargs)

            if isinstance(array, np.ndarray):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                np.save(os.path.splitext(path)[0], array)
            elif isinstance(array, pd.DataFrame):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                array.to_pickle(path)

        return array

    return wrapper


@load_or_fit
def fit_instance(_, path, data: np.ndarray, instance, label=None):
    if label is None:
        instance.fit(data)
    else:
        instance.fit(data, label)

    log.info(f"Fit preprocess. -> {path}")
    return instance


@load_or_transform
def transform_data(
    c, path, data: Union[np.ndarray, pd.DataFrame], instance, label=None
) -> Union[np.ndarray, pd.DataFrame]:
    instance = fit_instance(
        c, re.sub("\w+-", "", path).replace(".npy", ".pkl").replace(".pickle", ".pkl"), data, instance, label
    )
    features = instance.transform(data)

    log.info(f"Transform data. -> {path}, shape: {features.shape}")
    return features
