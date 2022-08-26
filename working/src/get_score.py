import logging
import traceback

import numpy as np
import pandas as pd
import scipy.stats as stats
import wandb
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

log = logging.getLogger(__name__)


def get_score(scoring, y_true, y_pred):
    if scoring == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
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
        if type(y_true) == pd.DataFrame: y_true = y_true.values
        if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
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
        # preds = df["preds"].to_numpy()
        # labels = df[c.settings.label_name].to_numpy()
        preds = df[[f"preds_{n}" for n in df[c.settings.label_name].unique()]].to_numpy()
        labels = df[["label_CE", "label_LAA"]].to_numpy()
        score = get_score(c.settings.scoring, labels, preds)

    log.info(f"Score: {score:<.5f}")
    if c.wandb.enabled:
        wandb.log({"score": score, "fold": fold})
        if loss is not None:
            wandb.log({"loss": loss, "fold": fold})

    return score


def pearson_coef(data):
    return data.corr()["target"]["preds"]


def optimize_function(c, y_true, y_pred):
    assert c.settings.scoring == "accuracy"

    def optimize_score(x):
        return -accuracy_score(y_true, y_pred > x)

    return optimize_score
