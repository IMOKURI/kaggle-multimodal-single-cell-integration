import logging
import os

import hydra
import numpy as np
import optuna
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from src.get_score import get_score
from src.load_data import PostprocessData, std

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    utils.fix_seed(utils.choice_seed(c))

    input = PostprocessData(c)

    ################################################################################
    # Ensemble by Average
    ################################################################################
    cite_oof = None
    multi_oof = None

    for df in input.cite_oof:
        if cite_oof is None:
            cite_oof = df
        else:
            cite_oof += df.sort_index()
    for df in input.multi_oof:
        if multi_oof is None:
            multi_oof = df
        else:
            multi_oof += df.sort_index()

    assert cite_oof is not None
    assert multi_oof is not None

    # cite_oof = cite_oof / len(input.cite_oof)
    # multi_oof = multi_oof / len(input.multi_oof)

    # https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349591
    cv_cite = get_score(c.settings.scoring, input.train_cite_targets.sort_index(), cite_oof.sort_index())
    cv_multi = get_score(c.settings.scoring, input.train_multi_targets.sort_index(), multi_oof.sort_index())
    cv = 0.712 * cv_cite + 0.288 * cv_multi
    log.info(f"All training data CV: {cv:.5f} (cite: {cv_cite:.5f}, multi: {cv_multi:.5f})")

    # validation data from adversarial training
    cite_good_validation = input.cite_adversarial_oof[
        (input.cite_adversarial_oof["label"] == 0) & (input.cite_adversarial_oof["preds"] == 1)
    ]
    multi_good_validation = input.multi_adversarial_oof[
        (input.multi_adversarial_oof["label"] == 0) & (input.multi_adversarial_oof["preds"] == 1)
    ]

    assert cite_good_validation is not None
    assert multi_good_validation is not None

    cv_cite = get_score(
        c.settings.scoring,
        input.train_cite_targets.loc[cite_good_validation.index, :].sort_index(),
        cite_oof.loc[cite_good_validation.index, :].sort_index(),
    )
    cv_multi = get_score(
        c.settings.scoring,
        input.train_multi_targets.loc[multi_good_validation.index, :].sort_index(),
        multi_oof.loc[multi_good_validation.index, :].sort_index(),
    )
    cv = 0.712 * cv_cite + 0.288 * cv_multi
    log.info(
        f"training data that similar test data CV: {cv:.5f} (cite: {cv_cite:.5f}(size: {len(cite_good_validation)}), multi: {cv_multi:.5f}(size: {len(multi_good_validation)}))"
    )

    ################################################################################
    # Optimize ensemble weights
    ################################################################################
    if len(input.cite_oof) > 1 and c.inference_params.cite_ensemble_weight_optimization:
        log.info(f"Optimize cite ensemble weight.")

        objective = Objective(input.train_cite_targets, input.cite_oof, cite_good_validation.index)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=400)

        optimized_cv_cite = study.best_trial.value
        best_weight_cite = list(study.best_trial.params.values())
        log.info(f"cite optimization result. CV: {optimized_cv_cite:.5f}, weight: {best_weight_cite}")
    else:
        optimized_cv_cite = cv_cite
        best_weight_cite = [1.0] * len(input.cite_oof)

    if len(input.multi_oof) > 1 and c.inference_params.multi_ensemble_weight_optimization:
        log.info(f"Optimize multi ensemble weight.")
        objective = Objective(input.train_multi_targets, input.multi_oof, multi_good_validation.index)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        optimized_cv_multi = study.best_trial.value
        best_weight_multi = list(study.best_trial.params.values())
        log.info(f"multi optimization result. CV: {optimized_cv_multi:.5f}, weight: {best_weight_multi}")
    else:
        optimized_cv_multi = cv_multi
        best_weight_multi = [1.0] * len(input.multi_oof)

    if c.inference_params.cite_ensemble_weight_optimization or c.inference_params.multi_ensemble_weight_optimization:
        assert optimized_cv_cite is not None
        assert optimized_cv_multi is not None

        cv = 0.712 * (optimized_cv_cite) + 0.288 * (optimized_cv_multi)
        log.info(f"training data that similar test data optimized CV: {cv:.5f}")

    ################################################################################
    # Inference
    ################################################################################
    cite_inf = None
    multi_inf = None

    for weight, df in zip(best_weight_cite, input.cite_inference):
        if cite_inf is None:
            cite_inf = df * weight
        else:
            cite_inf += df * weight
    for weight, df in zip(best_weight_multi, input.multi_inference):
        if multi_inf is None:
            multi_inf = df * weight
        else:
            multi_inf += df * weight

    assert cite_inf is not None
    assert multi_inf is not None

    cite_inf = pd.DataFrame(std(cite_inf.to_numpy()), index=cite_inf.index, columns=cite_inf.columns)
    multi_inf = pd.DataFrame(std(multi_inf.to_numpy()), index=multi_inf.index, columns=multi_inf.columns)

    inference = pd.concat([cite_inf, multi_inf])

    for row_id, cell_id, gene_id in zip(
        input.evaluation_ids["row_id"], input.evaluation_ids["cell_id"], input.evaluation_ids["gene_id"]
    ):
        input.sample_submission.at[row_id, "target"] = inference.at[cell_id, gene_id]

    if c.inference_params.pretrained is not None:
        target = input.sample_submission["target"]
        for df in input.public_inference:
            target = target + df["target"].to_numpy()
        input.sample_submission["target"] = target

    assert input.sample_submission["target"].isnull().sum() == 0

    submission_path = os.path.join(HydraConfig.get().run.dir, "submission.csv")
    input.sample_submission.to_csv(submission_path, index=False)

    log.info("Done.")


class Objective:
    def __init__(self, target, predictions, index=None):
        self.num = len(predictions)
        self.target = target
        self.predictions = predictions
        self.index = index

    def __call__(self, trial):
        weights = [0] * self.num

        for n in range(self.num):
            weights[n] = trial.suggest_float(f"weight_{n}", 0, 1, step=1e-5)

        df = None
        for weight, prediction in zip(weights, self.predictions):
            if df is None:
                df = prediction.sort_index() * weight
            else:
                df += prediction.sort_index() * weight

        assert df is not None

        if self.index is not None:
            df = df.loc[self.index, :]
            target_df = self.target.loc[self.index, :]
        else:
            target_df = self.target

        score = get_score("pearson", target_df.sort_index(), df.sort_index())
        return score


if __name__ == "__main__":
    main()
