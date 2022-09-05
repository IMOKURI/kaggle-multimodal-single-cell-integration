import logging
import os

import hydra
import numpy as np
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from src.load_data import PostprocessData
from src.get_score import get_score

if utils.is_env_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    utils.fix_seed(utils.choice_seed(c))

    input = PostprocessData(c)

    # Cite の day2-donor27678 の値を train の day2-donor32606 の target に変換する (Public test leak...)
    leak_32606_cell_id = input.metadata[
        (input.metadata["donor"] == 32606) & (input.metadata["technology"] == "citeseq") & (input.metadata["day"] == 2)
    ]["cell_id"]
    leak_27678_cell_id = input.metadata[
        (input.metadata["donor"] == 27678) & (input.metadata["technology"] == "citeseq") & (input.metadata["day"] == 2)
    ]["cell_id"]
    input.cite_inference.loc[leak_27678_cell_id, :] = input.train_cite_targets.loc[leak_32606_cell_id, :].to_numpy()
    assert len(leak_32606_cell_id) == 7476
    assert len(leak_27678_cell_id) == 7476
    assert np.array_equal(
        input.cite_inference.loc[leak_27678_cell_id, :].to_numpy(),
        input.train_cite_targets.loc[leak_32606_cell_id, :].to_numpy(),
    )

    # Multiome の target は 非負
    input.multi_oof[input.multi_oof < 0] = 0
    input.multi_inference[input.multi_inference < 0] = 0
    assert (input.multi_oof < 0).sum().sum() == 0
    assert (input.multi_inference < 0).sum().sum() == 0

    # https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349591
    cv_cite = get_score(c.settings.scoring, input.train_cite_targets.sort_index(), input.cite_oof.sort_index())
    cv_multi = get_score(c.settings.scoring, input.train_multi_targets.sort_index(), input.multi_oof.sort_index())
    cv = 0.743 * cv_cite + 0.257 * cv_multi
    log.info(f"CV: {cv} (cite: {cv_cite}, multi: {cv_multi})")

    inference = pd.concat([input.cite_inference, input.multi_inference])

    for row_id, cell_id, gene_id in tqdm(
        zip(input.evaluation_ids["row_id"], input.evaluation_ids["cell_id"], input.evaluation_ids["gene_id"]),
        total=len(inference),
    ):
        input.sample_submission.at[row_id, "target"] = inference.at[cell_id, gene_id]

    submission_path = os.path.join(HydraConfig.get().run.dir, "submission.csv")
    input.sample_submission.to_csv(submission_path, index=False)

    log.info("Done.")


if __name__ == "__main__":
    main()
