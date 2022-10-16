import io
import logging
import os
import zipfile
from functools import partial
from typing import Any

import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .get_score import pearson_cc_xgb_score
from .make_loss import pearson_cc_loss, torch_autodiff_grad_hess
from .models.auto_encoder import DenoisingAutoEncoder
from .models.image import ImageBaseModel
from .models.mlp import MlpBaseModel, MlpDropoutModel, MlpResnetModel
from .models.node import DenseBlock, Lambda, entmax15, entmoid15
from .models.one_d_cnn import OneDCNNModel

# import jax

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.model_params.model == "image_base":
        model = ImageBaseModel(c)
    elif c.model_params.model == "mlp_base":
        model = MlpBaseModel(c)
    elif c.model_params.model == "mlp_dropout":
        model = MlpDropoutModel(c)
    elif c.model_params.model == "mlp_resnet":
        model = MlpResnetModel(c)
    elif c.model_params.model == "denoising_auto_encoder":
        model = DenoisingAutoEncoder(c)
    elif c.model_params.model == "one_d_cnn":
        model = OneDCNNModel(c)
    elif c.model_params.model == "node":
        model = nn.Sequential(
            DenseBlock(
                c.model_params.model_input,
                layer_dim=512,
                num_layers=2,
                tree_dim=c.settings.n_class + 1,
                flatten_output=False,
                depth=4,
                choice_function=entmax15,
                bin_function=entmoid15,
            ),
            Lambda(lambda x: x[..., : c.settings.n_class].mean(dim=-2)),
        )
    else:
        raise Exception("Invalid model.")

    if c.settings.debug:
        model_state_dict = model.state_dict()
        log.debug(f"Model layers: {model_state_dict.keys()}")

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


def make_model_catboost(c, ds=None, model_path=None):

    cat_params = dict(
        iterations=10000,
        early_stopping_rounds=20,
        # learning_rate=0.05,
        objective="RMSE",
        eval_metric="RMSE",
        random_state=c.global_params.seed,
        task_type="GPU",  # Catboost does not support multitarget on GPU yet
    )  # type: dict[str, Any]

    # if ds is not None:
    #     num_data = len(ds)
    #     num_steps = num_data // (c.training_params.batch_size * c.training_params.gradient_acc_step) * c.training_params.epoch + 5
    #
    #     cat_params["scheduler_params"] = dict(T_0=num_steps, T_mult=1, eta_min=c.training_params.min_lr, last_epoch=-1)
    #     cat_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    clf = MultiOutputRegressor(CatBoostRegressor(**cat_params), n_jobs=-1)
    # clf = CatBoostClassifier(**cat_params)

    if model_path is not None:
        clf.load_model(model_path)

    return clf


def make_model_xgboost(c, ds=None, model_path=None):

    custom_objective = partial(torch_autodiff_grad_hess, pearson_cc_loss)
    # jax_custom_objective = jax.jit(partial(jax_autodiff_grad_hess, jax_pearson_cc_loss))

    xgb_params = dict(
        n_estimators=10000,
        early_stopping_rounds=20,
        # learning_rate=0.05,
        objective=custom_objective,  # "binary:logistic", "reg:squarederror",
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
        # cat_idxs=c_index,
        # cat_dims=c_features,
        # cat_emb_dim=[1],
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

    if c.preprocess_params.use_cell_type:
        tabnet_params["cat_idxs"] = c_index
        tabnet_params["cat_dims"] = c_features
        tabnet_params["cat_emb_dim"] = [1]

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
        # cat_idxs=c_index,
        # cat_dims=c_features,
        # cat_emb_dim=[1],
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

    if c.preprocess_params.use_cell_type:
        tabnet_params["cat_idxs"] = c_index
        tabnet_params["cat_dims"] = c_features
        tabnet_params["cat_emb_dim"] = [1]

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
