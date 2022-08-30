import logging

import hydra
import src.utils as utils
from src.load_data import PreprocessData

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    utils.fix_seed(utils.choice_seed(c))

    PreprocessData(c)

    log.info("Done.")


if __name__ == "__main__":
    main()
