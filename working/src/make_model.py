import io
import logging
import os
import zipfile
from typing import Any

import joblib
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import xgboost as xgb
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from .get_score import pearson_cc_xgb_score
from .models.image import ImageBaseModel
from .models.node import DenseBlock, Lambda, entmax15, entmoid15

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.model_params.model == "base":
        pretrained = True if model_path is None else False
        model = ImageBaseModel(c, pretrained)
    elif c.model_params.model == "node":
        model = nn.Sequential(DenseBlock())
    else:
        raise Exception("Invalid model.")

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    if device:
        model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model


def make_model_ridge(c, ds=None, model_path=None):

    if c.global_params.method == "kernel_ridge":
        kernel = RBF(length_scale=10.0)
        kernel_ridge_params = dict(alpha=0.1, kernel=kernel)  # type: dict[str, Any]

        clf = KernelRidge(**kernel_ridge_params)

    else:
        ridge_params = dict(
            random_state=c.global_params.seed,
        )  # type: dict[str, Any]

        clf = Ridge(**ridge_params)

    # if model_path is not None:
    #     clf.load_model(model_path)

    return clf


def make_model_xgboost(c, ds=None, model_path=None):

    xgb_params = dict(
        n_estimators=1000,
        early_stopping_rounds=20,
        # learning_rate=0.05,
        objective="reg:squarederror",  # "binary:logistic", "reg:squarederror",
        eval_metric=pearson_cc_xgb_score,  # "logloss", "rmse",
        random_state=c.global_params.seed,
        tree_method="gpu_hist",
    )  # type: dict[str, Any]

    # if ds is not None:
    #     num_data = len(ds)
    #     num_steps = num_data // (c.training_params.batch_size * c.training_params.gradient_acc_step) * c.training_params.epoch + 5
    #
    #     xgb_params["scheduler_params"] = dict(T_0=num_steps, T_mult=1, eta_min=c.training_params.min_lr, last_epoch=-1)
    #     xgb_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    clf = xgb.XGBRegressor(**xgb_params)
    # clf = xgb.XGBClassifier(**xgb_params)

    if model_path is not None:
        clf.load_model(model_path)

    return clf


def make_pre_model_tabnet(c, c_index=None, c_features=None):
    tabnet_params = dict(
        n_d=32,
        n_a=32,
        n_steps=1,
        n_independent=2,  # 2 is better CV than 1, but need more time
        n_shared=2,  # same above
        gamma=1.3,
        lambda_sparse=0,
        cat_idxs=c_index,
        cat_dims=c_features,
        cat_emb_dim=[1],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=c.training_params.lr, weight_decay=c.training_params.weight_decay),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=dict(
            min_lr=c.training_params.min_lr, patience=c.training_params.es_patience // 5, verbose=True
        ),
        mask_type="entmax",
        seed=c.global_params.seed,
        verbose=5,
    )  # type: dict[str, Any]

    clf = TabNetPretrainer(**tabnet_params)
    return clf


def make_model_tabnet(c, ds=None, model_path=None, c_index=None, c_features=None):

    tabnet_params = dict(
        n_d=32,
        n_a=32,
        n_steps=1,
        n_independent=2,  # 2 is better CV than 1, but need more time
        n_shared=2,  # same above
        gamma=1.3,
        lambda_sparse=0,
        cat_idxs=c_index,
        cat_dims=c_features,
        cat_emb_dim=[1],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=c.training_params.lr, weight_decay=c.training_params.weight_decay),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=dict(
            mode="max", min_lr=c.training_params.min_lr, patience=c.training_params.es_patience // 5, verbose=True
        ),
        mask_type="entmax",
        seed=c.global_params.seed,
        verbose=5,
    )  # type: dict[str, Any]

    if "adversarial" in c.global_params.method or "cell_type_" in c.global_params.method:
        clf = TabNetClassifier(**tabnet_params)
    else:
        clf = TabNetRegressor(**tabnet_params)

    if model_path is not None:
        clf.load_model(model_path)

    return clf


def load_model(c, device, pretrained=None):
    if pretrained is None:
        pretrained = c.inference_params.pretrained
    models = []

    for training in pretrained:
        try:
            c.model_params.model = training.model
            c.model_params.model_name = training.model_name
            # c.model_params.model_input = training.model_input
            # c.training_params.feature_set = training.feature_set
        except Exception:
            pass

        # if training.model == "lightgbm":
        #     model = joblib.load(os.path.join(training.dir, "lightgbm.pkl"))
        # elif training.model == "xgboost":
        #     model = make_model_xgboost(c, model_path=os.path.join(training.dir, "xgboost.pkl"))
        # elif training.model == "tabnet":
        #     tabnet_zip = io.BytesIO()
        #     with zipfile.ZipFile(tabnet_zip, "w") as z:
        #         z.write(os.path.join(training.dir, "model_params.json"), arcname="model_params.json")
        #         z.write(os.path.join(training.dir, "network.pt"), arcname="network.pt")
        #     model = make_model_tabnet(c, model_path=tabnet_zip)
        # else:
        #     model = make_model(c, device, training.dir)
        model = make_model(c, device, training.dir)

        models.append(model)

    return models
