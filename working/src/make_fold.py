import itertools
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

log = logging.getLogger(__name__)


def make_fold(c, df):
    index = df.index
    df = df.reset_index(drop=True)

    if c.cv_params.n_fold == 0:
        df.loc[:, "fold"] = -1
    elif c.cv_params.fold == "kfold":
        df = kfold(c, df)
    elif c.cv_params.fold == "bins_stratified":
        df = bins_stratified_kfold(c, df, c.settings.label_name)
    elif c.cv_params.fold == "stratified":
        df = stratified_kfold(c, df, c.settings.label_name)
    elif c.cv_params.fold == "group":
        df = group_kfold(c, df, c.cv_params.group_name)
    elif c.cv_params.fold == "stratified_group":
        df = stratified_group_kfold(c, df, c.settings.label_name, c.cv_params.group_name)
    elif c.cv_params.fold in ["combinational_group", "combinational_purged"]:
        df = simple_combinational_purged_kfold(c, df, c.cv_params.group_name, c.cv_params.time_name)

    else:
        raise Exception("Invalid fold.")

    df.index = index
    return df


# https://github.com/IMOKURI/ubiquant-market-prediction/issues/1#issuecomment-1025375366
def make_cpcv_index(c):
    assert c.cv_params.n_fold == 5
    assert c.cv_params.n_validation in [1, 2]
    return {
        "val_group_id": {
            0: 1,
            1: 0,
            2: 2,
            3: 3,
            4: 4,
            5: 2,
            6: 4,
            7: 0,
            8: 3,
            9: 3,
            10: 4,
            11: 1,
            12: 1,
            13: 0,
            14: 2,
        },
        "val_time_id": {
            n: list(map(int, list(s)))
            for n, s in enumerate(itertools.combinations(range(c.cv_params.n_fold + 1), c.cv_params.n_validation))
        },
    }


def train_test_split(c, df, fold):
    embargo = 20  # num of SecuritiesCode * 0.01
    purged = 20

    if c.cv_params.fold == "combinational_group":
        # https://github.com/IMOKURI/ubiquant-market-prediction/issues/1#issuecomment-1025375366
        # ???????????????????????? group (SecuritiesCode) ?????????
        val_idx = df[df["time_fold"].isin(make_cpcv_index(c)["val_time_id"][fold])].index

        banned = []
        for f in make_cpcv_index(c)["val_time_id"][fold]:
            idx = df[df["time_fold"] == f].index
            banned += list(range(min(idx) - purged, max(idx) + purged + embargo))
        banned = list(set(banned))

        trn_idx = df[(~df.index.isin(banned)) & (df["group_fold"] != make_cpcv_index(c)["val_group_id"][fold])].index

    elif c.cv_params.fold == "combinational_purged":
        # ???????????? SecuritiesCode ??????????????????
        val_idx = df[df["time_fold"].isin(make_cpcv_index(c)["val_time_id"][fold])].index

        banned = []
        for f in make_cpcv_index(c)["val_time_id"][fold]:
            idx = df[df["time_fold"] == f].index
            banned += list(range(min(idx) - purged, max(idx) + purged + embargo))
        banned = list(set(banned))

        trn_idx = df[~df.index.isin(banned)].index

    else:
        trn_idx = df[df["fold"] != fold].index
        val_idx = df[df["fold"] == fold].index

    log.info(f"Num of training data: {len(trn_idx)}, num of validation data: {len(val_idx)}")

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx]  # .reset_index(drop=True)

    return train_folds, valid_folds


def kfold(c, df):
    fold_ = KFold(n_splits=c.cv_params.n_fold, shuffle=True, random_state=c.global_params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df)):
        df.loc[val_index, "fold"] = int(n)

    return df


def bins_stratified_kfold(c, df, col):
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[col], bins=num_bins, labels=False)

    fold_ = StratifiedKFold(n_splits=c.cv_params.n_fold, shuffle=True, random_state=c.global_params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df["bins"])):
        df.loc[val_index, "fold"] = int(n)

    return df


def stratified_kfold(c, df, col):
    fold_ = StratifiedKFold(n_splits=c.cv_params.n_fold, shuffle=True, random_state=c.global_params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


def group_kfold(c, df, col):
    fold_ = GroupKFold(n_splits=c.cv_params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df, groups=df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


def stratified_group_kfold(c, df, label, group):
    fold_ = StratifiedGroupKFold(n_splits=c.cv_params.n_fold, shuffle=True, random_state=c.global_params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df[label], df[group])):
        df.loc[val_index, "fold"] = int(n)

    return df


# https://blog.amedama.jp/entry/time-series-cv
class MovingWindowKFold(TimeSeriesSplit):
    """????????????????????????????????????????????????????????? iloc ????????? KFold"""

    def __init__(self, ts_column, clipping=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ???????????????????????????????????????
        self.ts_column = ts_column
        # ????????????????????????????????????????????????????????? Fold ?????????????????????
        self.clipping = clipping

    def split(self, X, *args, **kwargs):
        # ???????????????????????? DataFrame ???????????????
        assert isinstance(X, pd.DataFrame)

        # clipping ???????????????????????????????????????
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # ????????????????????????????????????
        ts = X[self.ts_column]
        # ????????????????????????????????????????????? iloc ????????????????????? (0, 1, 2...) ?????????
        ts_df = ts.reset_index()
        # ???????????????????????????
        sorted_ts_df = ts_df.sort_values(by=self.ts_column)
        # ????????????????????????????????????????????????????????????
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # ?????????????????? DataFrame ??? iloc ????????????????????????????????????
            train_iloc_index = sorted_ts_df.iloc[train_index].index
            test_iloc_index = sorted_ts_df.iloc[test_index].index

            if self.clipping:
                # TimeSeriesSplit.split() ??????????????? Fold ??????????????????????????????????????????????????????????????????
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])


def simple_combinational_purged_kfold(c, df, group_col, time_col):
    fold_ = GroupKFold(n_splits=c.cv_params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df, groups=df[group_col])):
        df.loc[val_index, "group_fold"] = int(n)

    fold_ = MovingWindowKFold(time_col, clipping=False, n_splits=c.cv_params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df)):
        df.loc[df.index[val_index], "time_fold"] = int(n)

    df = df.fillna({"time_fold": -1})
    df["time_fold"] += 1

    return df
