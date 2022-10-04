import logging
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import wandb
import xgboost as xgb
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import accuracy_score, auc, log_loss, mean_squared_error

log = logging.getLogger(__name__)


def get_score(scoring, y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.to_numpy()
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.to_numpy()

    if scoring == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == "auc":
        return auc(y_true, y_pred)
    elif scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif scoring == "logloss":
        return log_loss(y_true, y_pred)
    elif scoring == "pearson":
        """
        Scores the predictions according to the competition rules.
        It is assumed that the predictions are not constant.
        Returns the average of each sample's Pearson correlation coefficient
        https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart#The-scoring-function
        """
        corrsum = 0
        for i in range(len(y_true)):
            corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
        return corrsum / len(y_true)

    else:
        raise Exception("Invalid scoring.")


def record_result(c, df, fold, label_df=None, loss=None):
    if c.settings.scoring == "mean":
        score = df["preds"].mean()
    elif c.settings.scoring == "pearson" and label_df is not None:
        score = get_score("pearson", label_df, df)
        # score = np.mean(df[["time_id", "target", "preds"]].groupby("time_id").apply(pearson_coef))
    else:
        preds = df["preds"].to_numpy()
        labels = df[c.settings.label_name].to_numpy()
        # preds = df[[f"preds_{n}" for n in df[c.settings.label_name].unique()]].to_numpy()
        # labels = df[["label_CE", "label_LAA"]].to_numpy()
        score = get_score(c.settings.scoring, labels, preds)

    log.info(f"Score: {score:<.5f}")
    if c.wandb.enabled:
        wandb.log({"score": score, "fold": fold})
        if loss is not None:
            wandb.log({"loss": loss, "fold": fold})

    return score


def pearson_coef(data):
    return data.corr()["target"]["preds"]


class PearsonCCTabNetScore(Metric):
    def __init__(self):
        self._name = "pearson"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        return get_score("pearson", y_true, y_pred)


def pearson_cc_xgb_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_true は 1次元で与えられるので reshape する
    n_col = y_pred.shape[1]
    score = get_score("pearson", y_true.reshape(-1, n_col), y_pred)
    return 1.0 - score


# def optimize_function(c, y_true, y_pred):
#     def optimize_score(x):
#         return -accuracy_score(y_true, y_pred > x)
#
#     return optimize_score


def optimize_func(target, predictions, index=None):
    """
    最小化するパラメータを引数とし、最小化したい値を返す関数を生成する
    """

    def optimize_score(weights):
        df = None
        for weight, prediction in zip(weights, predictions):
            if df is None:
                df = prediction.sort_index() * weight
            else:
                df += prediction.sort_index() * weight

        if index is not None:
            df = df.loc[index, :]
            target_df = target.loc[index, :]
        else:
            target_df = target

        score = get_score("pearson", target_df.sort_index(), df.sort_index())
        return -score

        # score = get_score("rmse", target_df.sort_index(), df.sort_index())
        # return score

    return optimize_score
