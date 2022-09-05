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

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.model_params.model == "base":
        pretrained = True if model_path is None else False
        model = ImageBaseModel(c, pretrained)
    else:
        raise Exception("Invalid model.")

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    if device:
        model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model


# def make_model_xgboost(c, ds=None, model_path=None):
#
#     xgb_params = dict(
#         n_estimators=10000,
#         # learning_rate=0.05,
#         objective="binary:logistic",  # "reg:squarederror",
#         eval_metric="logloss",  # "rmse",
#         random_state=c.global_params.seed,
#         tree_method="gpu_hist",
#     )  # type: dict[str, Any]
#
#     # if ds is not None:
#     #     num_data = len(ds)
#     #     num_steps = num_data // (c.training_params.batch_size * c.training_params.gradient_acc_step) * c.training_params.epoch + 5
#     #
#     #     xgb_params["scheduler_params"] = dict(T_0=num_steps, T_mult=1, eta_min=c.training_params.min_lr, last_epoch=-1)
#     #     xgb_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#
#     clf = xgb.XGBRegressor(**xgb_params)
#     # clf = xgb.XGBClassifier(**xgb_params)
#
#     if model_path is not None:
#         clf.load_model(model_path)
#
#     return clf


def make_model_tabnet(c, ds=None, model_path=None, c_index=None, c_features=None):

    tabnet_params = dict(
        n_d=16,
        n_a=16,
        n_steps=2,
        n_independent=2,  # 2 is better CV than 1, but need more time
        n_shared=2,  # same above
        gamma=1.3,
        lambda_sparse=0,
        cat_idxs=c_index,
        cat_dims=c_features,
        cat_emb_dim=[1],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=c.training_params.lr, weight_decay=c.training_params.weight_decay),
        mask_type="entmax",
        seed=c.global_params.seed,
        verbose=10,
    )  # type: dict[str, Any]

    if ds is not None:
        num_data = len(ds)
        num_steps = (
            num_data // (c.training_params.batch_size * c.training_params.gradient_acc_step) * c.training_params.epoch
            + 5
        )

        tabnet_params["scheduler_params"] = dict(
            T_0=num_steps, T_mult=1, eta_min=c.training_params.min_lr, last_epoch=-1
        )
        tabnet_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    clf = TabNetRegressor(**tabnet_params)
    # clf = TabNetClassifier(**tabnet_params)

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


class ImageBaseModel(nn.Module):
    def __init__(self, c, pretrained=True):
        super().__init__()
        # self.model = timm.create_model(c.model_params.model_name, pretrained=pretrained, num_classes=c.settings.n_class)
        self.model = None

    def forward(self, x):
        x = self.model(x)
        return x
