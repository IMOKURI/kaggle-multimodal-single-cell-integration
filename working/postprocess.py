import logging
import os

import hydra
import numpy as np
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from src.get_score import get_score
from src.load_data import PostprocessData

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    utils.fix_seed(utils.choice_seed(c))

    input = PostprocessData(c)

    # Multiome の target は 非負
    input.multi_oof[input.multi_oof < 0] = 0
    input.multi_inference[input.multi_inference < 0] = 0
    assert (input.multi_oof < 0).sum().sum() == 0
    assert (input.multi_inference < 0).sum().sum() == 0

    # https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349591
    cv_cite = get_score(c.settings.scoring, input.train_cite_targets.sort_index(), input.cite_oof.sort_index())
    cv_multi = get_score(c.settings.scoring, input.train_multi_targets.sort_index(), input.multi_oof.sort_index())
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
        input.cite_oof.loc[cite_good_validation.index, :].sort_index(),
    )
    cv_multi = get_score(
        c.settings.scoring,
        input.train_multi_targets.loc[multi_good_validation.index, :].sort_index(),
        input.multi_oof.loc[multi_good_validation.index, :].sort_index(),
    )
    cv = 0.712 * cv_cite + 0.288 * cv_multi
    log.info(
        f"training data that similar test data CV: {cv:.5f} (cite: {cv_cite:.5f}(size: {len(cite_good_validation)}), multi: {cv_multi:.5f}(size: {len(multi_good_validation)}))"
    )

    inference = pd.concat([input.cite_inference, input.multi_inference])

    for row_id, cell_id, gene_id in zip(
        input.evaluation_ids["row_id"], input.evaluation_ids["cell_id"], input.evaluation_ids["gene_id"]
    ):
        input.sample_submission.at[row_id, "target"] = inference.at[cell_id, gene_id]

    submission_path = os.path.join(HydraConfig.get().run.dir, "submission.csv")
    input.sample_submission.to_csv(submission_path, index=False)

    log.info("Done.")


if __name__ == "__main__":
    main()
