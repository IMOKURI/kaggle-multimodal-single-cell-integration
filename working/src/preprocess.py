# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import gc
import logging
import os
import pickle
import re
from functools import wraps
from typing import Callable, Union

import numpy as np
import pandas as pd

from .preprocesses.p010_pca import CustomPCA

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame, stem: str) -> pd.DataFrame:
    # Convert None to NaN
    df = df.fillna(np.nan)

    ...

    # Convert None to NaN
    df = df.fillna(np.nan)

    return df


def preprocess_train_test(c, train_df: pd.DataFrame, test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train_size = len(train_df)

    df = pd.concat([train_df, test_df])

    preprocessor = CustomPCA(c)
    df = transform_data(c, f"{c.global_params.data}-pca.f", df, preprocessor)

    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]

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
        elif os.path.exists(path) and os.path.splitext(path)[1] == ".f":
            array = pd.read_feather(path)

        else:
            array = func(*args, **kwargs)

            if isinstance(array, np.ndarray):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                np.save(os.path.splitext(path)[0], array)
            elif isinstance(array, pd.DataFrame):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                array.to_feather(path)

        return array

    return wrapper


@load_or_fit
def fit_instance(_, path, data: np.ndarray, instance):
    instance.fit(data)

    log.info(f"Fit preprocess. -> {path}")
    return instance


@load_or_transform
def transform_data(c, path, data: Union[np.ndarray, pd.DataFrame], instance) -> Union[np.ndarray, pd.DataFrame]:
    instance = fit_instance(c, re.sub("\w+-", "", path).replace(".npy", ".pkl").replace(".f", ".pkl"), data, instance)
    features = instance.transform(data)

    log.info(f"Transform data. -> {path}, shape: {features.shape}")
    return features
