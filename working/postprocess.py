import logging
import os

import hydra
import pandas as pd
import src.utils as utils
from hydra.core.hydra_config import HydraConfig
from src.load_data import PostprocessData

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

    # Multiome の target は 非負
    input.multi_inference[input.multi_inference < 0] = 0
    assert (input.multi_inference < 0).sum().sum() == 0

    inference = pd.concat([input.multi_inference, input.cite_inference])

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
