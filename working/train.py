import logging
import os

import hydra
import numpy as np
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from omegaconf.errors import ConfigAttributeError
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from src.get_score import optimize_function, record_result
from src.load_data import LoadData
from src.run_loop import (  # train_fold_lightgbm,
    adversarial_train_fold_tabnet,
    train_fold_nn,
    train_fold_ridge,
    train_fold_tabnet,
    train_fold_xgboost,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    run = utils.setup_wandb(c)

    utils.fix_seed(utils.choice_seed(c))
    device = utils.gpu_settings(c)

    input = LoadData(c)

    oof_df = pd.DataFrame()
    label_df = pd.DataFrame()
    inference_df = pd.DataFrame()

    losses = utils.AverageMeter()
    single_run = False

    # if c.cv_params.fold in ["combinational_group", "combinational_purged"]:
    #     if c.cv_params.n_validation == 1:
    #         num_fold = c.cv_params.n_fold + 1
    #     elif c.cv_params.n_validation == 2:
    #         num_fold = c.cv_params.n_fold * 3
    #     else:
    #         raise Exception("Invalid n_validation.")
    # else:
    #     num_fold = c.cv_params.n_fold
    num_fold = c.cv_params.n_fold if c.cv_params.n_fold > 0 else 1

    for fold in range(num_fold):
        try:
            fold = int(c.settings.run_fold)
            single_run = True
        except ConfigAttributeError:
            pass

        log.info(f"========== fold {fold} training start ==========")
        utils.fix_seed(c.global_params.seed + fold)

        if False:
            raise
        # elif c.global_params.method == "lightgbm":
        #     _oof_df, _label_df, loss = train_fold_lightgbm(c, input, fold)
        elif c.global_params.method == "ridge":
            _oof_df, _label_df, loss, _inference_df = train_fold_ridge(c, input, fold)
        elif c.global_params.method == "xgboost":
            _oof_df, _label_df, loss, _inference_df = train_fold_xgboost(c, input, fold)
        elif c.global_params.method == "tabnet":
            _oof_df, _label_df, loss, _inference_df = train_fold_tabnet(c, input, fold)
        elif c.global_params.method == "adversarial_tabnet":
            _oof_df, _label_df, loss, _inference_df = adversarial_train_fold_tabnet(c, input, fold)
        elif c.global_params.method == "nn":
            _oof_df, _label_df, loss, _inference_df = train_fold_nn(c, input, fold, device)
        else:
            raise Exception(f"Invalid training method. {c.global_params.method}")

        log.info(f"========== fold {fold} training result ==========")
        log.debug(f"_oof_df: {_oof_df.shape}, _label_df: {_label_df.shape}")
        record_result(c, _oof_df, fold, _label_df, loss)

        oof_df = pd.concat([oof_df, _oof_df])
        label_df = pd.concat([label_df, _label_df])
        inference_df = pd.concat([inference_df, _inference_df])
        losses.update(loss)

        if c.settings.debug or single_run:
            break

    log.info("========== postprocess ==========")
    if c.global_params.data == "multi":
        oof_df[oof_df < 0] = 0

    log.info("========== training result ==========")
    score = record_result(c, oof_df, c.cv_params.n_fold, label_df, losses.avg)

    # log.info("========== optimize training result ==========")
    # minimize_result = minimize(
    #     optimize_function(c, oof_df[c.settings.label_name].to_numpy(), oof_df["base_preds"].to_numpy()),
    #     np.array([0.5]),
    #     method="Nelder-Mead",
    # )
    # log.info(f"optimize result. -> \n{minimize_result}")
    # score = record_result(c, oof_df, c.cv_params.n_fold, losses.avg)
    # oof_df["preds"] = (oof_df["base_preds"] > minimize_result["x"].item()).astype(np.int8)

    oof_path = os.path.join(HydraConfig.get().run.dir, "oof.pickle")
    oof_df.to_pickle(oof_path)

    log.info(f"oof -> \n{oof_df}")

    if "adversarial" in c.global_params.method:
        cm = confusion_matrix(oof_df[c.settings.label_name], oof_df["preds"])
        log.info(f"confusion matrix: {cm}")

    if not c.settings.skip_inference:
        log.info("========== inference result ==========")

        inference_df = inference_df.groupby("cell_id").mean()

        inference_path = os.path.join(HydraConfig.get().run.dir, f"{c.global_params.data}_inference.pickle")
        inference_df.to_pickle(inference_path)

        log.info(f"inference -> \n{inference_df}")

    #     cols_base_preds = [col for col in input.test.columns if "base_preds" in col]
    #     input.test["base_preds"] = nanmean(input.test[cols_base_preds].to_numpy(), axis=1)
    #     input.test["preds"] = (input.test[f"base_preds"] > minimize_result["x"].item()).astype(bool)
    #
    #     input.sample_submission[c.settings.label_name] = input.test["preds"]
    #
    #     input.test.to_feather("inference.f")
    #     input.sample_submission.to_csv("submission.csv", index=False)
    #
    #     log.info(f"inference -> \n{input.test}")

    log.info("Done.")

    utils.teardown_wandb(c, run, losses.avg, score)


if __name__ == "__main__":
    main()
