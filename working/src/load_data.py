import logging
import os

import pandas as pd

from .make_fold import make_fold
from .preprocess import preprocess, preprocess_train_test
from .utils import reduce_mem_usage

log = logging.getLogger(__name__)


class InputData:
    def __init__(self, c, use_fold=True, do_preprocess=True):
        self.c = c

        for file_name in c.settings.inputs:
            stem = os.path.splitext(file_name)[0].replace("/", "__")
            extension = os.path.splitext(file_name)[1]

            original_file_path = os.path.join(c.settings.dirs.input, file_name)
            f_file_path = original_file_path.replace(extension, ".f")

            if os.path.exists(f_file_path):
                log.info(f"Load feather file. path: {f_file_path}")
                df = pd.read_feather(f_file_path)

            elif os.path.exists(original_file_path):
                log.info(f"Load original file. path: {original_file_path}")

                if extension == ".csv":
                    df = pd.read_csv(original_file_path, low_memory=False)
                    df.to_feather(f_file_path)

                elif extension == ".h5":
                    df = pd.read_hdf(original_file_path)
                    # df.to_feather(f_file_path)

                else:
                    raise Exception(f"Invalid extension to load file. filename: {original_file_path}")

            else:
                log.warning(f"File does not exist. path: {original_file_path}")
                continue

            if c.settings.debug:
                df = sample_for_debug(c, df)

            if stem in [] and do_preprocess:
                df = preprocess(c, df, stem)

            if stem in [] and use_fold:
                df = make_fold(c, df)

            df = reduce_mem_usage(df)

            setattr(self, stem, df)

        if hasattr(self, "train_cite_inputs") and do_preprocess:
            train, test = preprocess_train_test(
                c, getattr(self, "train_cite_inputs"), getattr(self, "test_cite_inputs"), "cite"
            )

            train = make_fold(c, train)

            train.index = getattr(self, "train_cite_inputs").index
            test.index = getattr(self, "test_cite_inputs").index

            setattr(self, "train_cite_inputs", train)
            setattr(self, "test_cite_inputs", test)

            self.train_cite_targets["fold"] = self.train_cite_inputs["fold"]

        elif hasattr(self, "train_multi_inputs") and do_preprocess:
            train, test = preprocess_train_test(
                c, getattr(self, "train_multi_inputs"), getattr(self, "test_multi_inputs"), "multiome"
            )

            train = make_fold(c, train)

            train.index = getattr(self, "train_multi_inputs").index
            test.index = getattr(self, "test_multi_inputs").index

            setattr(self, "train_multi_inputs", train)
            setattr(self, "test_multi_inputs", test)

            self.train_multi_targets["fold"] = self.train_multi_inputs["fold"]


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data and c.settings.n_debug_data > 0:
        df = df.sample(n=c.settings.n_debug_data, random_state=c.global_params.seed).reset_index(drop=True)
    return df
