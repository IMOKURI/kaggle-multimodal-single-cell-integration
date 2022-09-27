import logging
import os
from typing import List

import hydra
import numpy as np
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from scipy.optimize import minimize
from src.get_score import get_score, optimize_func
from src.load_data import PostprocessData

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

    cite_oof = cite_oof / len(input.cite_oof)
    multi_oof = multi_oof / len(input.multi_oof)

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
    # weight の合計が 1 になるようにする
    cons = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    if len(input.cite_oof) > 1 and c.inference_params.ensemble_weight_optimization:
        log.info(f"Optimize cite ensemble weight.")
        scores = []
        weights = []
        for i in range(10):
            starting_values = np.random.uniform(size=len(input.cite_oof))
            bounds = [(0, 1)] * len(input.cite_oof)

            res = minimize(
                optimize_func(input.train_cite_targets, input.cite_oof, cite_good_validation.index),
                starting_values,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            scores.append(res["fun"])
            weights.append(res["x"])
            log.info(f"optimization epoch {i+1}, score: {-res['fun']}, weight: {res['x']}")

        optimized_cv_cite = np.min(scores)
        best_weight_cite = weights[np.argmin(scores)]
        log.info(f"cite optimization result. CV: {-optimized_cv_cite:.5f}, weight: {best_weight_cite}")
    else:
        optimized_cv_cite = -cv_cite
        best_weight_cite = [1.0] * len(input.cite_oof)

    if len(input.multi_oof) > 1 and c.inference_params.ensemble_weight_optimization:
        log.info(f"Optimize multi ensemble weight.")
        scores = []
        weights = []
        for i in range(10):
            starting_values = np.random.uniform(size=len(input.multi_oof))
            bounds = [(0, 1)] * len(input.multi_oof)

            res = minimize(
                optimize_func(input.train_multi_targets, input.multi_oof, multi_good_validation.index),
                starting_values,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            scores.append(res["fun"])
            weights.append(res["x"])
            log.info(f"optimization epoch {i+1}, score: {-res['fun']}, weight: {res['x']}")

        optimized_cv_multi = np.min(scores)
        best_weight_multi = weights[np.argmin(scores)]
        log.info(f"multi optimization result. CV: {-optimized_cv_multi:.5f}, weight: {best_weight_multi}")
    else:
        optimized_cv_multi = -cv_multi
        best_weight_multi = [1.0] * len(input.multi_oof)

    if c.inference_params.ensemble_weight_optimization:
        cv = 0.712 * (-optimized_cv_cite) + 0.288 * (-optimized_cv_multi)
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

    inference = pd.concat([cite_inf, multi_inf])

    for row_id, cell_id, gene_id in zip(
        input.evaluation_ids["row_id"], input.evaluation_ids["cell_id"], input.evaluation_ids["gene_id"]
    ):
        input.sample_submission.at[row_id, "target"] = inference.at[cell_id, gene_id]

    if c.inference_params.pretrained is not None:
        ...
        # input.sample_submission["target"] = input.sample_submission["target"] + input.public_inference["target"]

    submission_path = os.path.join(HydraConfig.get().run.dir, "submission.csv")
    input.sample_submission.to_csv(submission_path, index=False)

    log.info("Done.")


if __name__ == "__main__":
    main()
