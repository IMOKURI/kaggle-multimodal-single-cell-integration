import logging
import os

import pandas as pd

from .make_fold import make_fold
from .preprocess import preprocess, preprocess_train_test
from .utils import reduce_mem_usage

log = logging.getLogger(__name__)


class PreprocessData:
    def __init__(self, c, do_preprocess=True):
        self.c = c

        for file_name in c.settings.inputs:
            stem = os.path.splitext(file_name)[0].replace("/", "__")
            extension = os.path.splitext(file_name)[1]

            if c.global_params.data == "cite" and "multi" in stem:
                continue
            if c.global_params.data == "multi" and "cite" in stem:
                continue

            original_file_path = os.path.join(c.settings.dirs.input, file_name)
            p_file_path = original_file_path.replace(extension, ".pickle")

            if os.path.exists(p_file_path):
                log.info(f"Load pickle file. path: {p_file_path}")
                df = pd.read_pickle(p_file_path)

            elif os.path.exists(original_file_path):
                log.info(f"Load original file. path: {original_file_path}")

                if extension == ".csv":
                    df = pd.read_csv(original_file_path, low_memory=False)
                    df.to_pickle(p_file_path)

                elif extension == ".h5":
                    df = pd.read_hdf(original_file_path)
                    df.to_pickle(p_file_path)

                else:
                    raise Exception(f"Invalid extension to load file. filename: {original_file_path}")

            else:
                log.warning(f"File does not exist. path: {original_file_path}")
                continue

            if c.preprocess_params.cols != "all":
                cols = [co for co in df.columns if co.startswith(c.preprocess_params.cols)]
                df = df[cols]

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            # if stem in [] and do_preprocess:
            #     df = preprocess(c, df, stem)

            setattr(self, stem, df)

        if do_preprocess:
            train, test = preprocess_train_test(
                c,
                getattr(self, f"train_{c.global_params.data}_inputs"),
                getattr(self, f"test_{c.global_params.data}_inputs"),
            )

            train.to_pickle(os.path.join(c.settings.dirs.preprocess, f"train_{c.global_params.data}_inputs.pickle"))
            test.to_pickle(os.path.join(c.settings.dirs.preprocess, f"test_{c.global_params.data}_inputs.pickle"))


class LoadData:
    def __init__(self, c, use_fold=True):
        self.c = c

        train_inputs = pd.DataFrame()
        train_targets = pd.DataFrame()
        test_inputs = pd.DataFrame()

        for file_name in c.settings.preprocesses:
            stem = os.path.splitext(file_name)[0].replace("/", "__")

            if c.global_params.data == "cite" and "multi" in stem:
                continue
            if c.global_params.data == "multi" and "cite" in stem:
                continue

            file_path = os.path.join(c.settings.dirs.preprocess, file_name)

            if os.path.exists(file_path):
                log.info(f"Load preprocessed file. path: {file_path}")
                df = pd.read_pickle(file_path)

            else:
                log.warning(f"File does not exist. path: {file_path}")
                continue

            if "targets" in stem:
                train_targets = df
            elif "train" in stem:
                train_inputs = pd.concat([train_inputs, df], axis=1)
                train_inputs.index = df.index
                train_inputs.index.name = df.index.name
            elif "test" in stem:
                test_inputs = pd.concat([test_inputs, df], axis=1)
                test_inputs.index = df.index
                test_inputs.index.name = df.index.name
            else:
                raise Exception("Invalid filename")

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            # df = reduce_mem_usage(df)

            # setattr(self, stem, df)

        assert train_inputs.index.equals(train_targets.index)
        assert train_inputs.index.name == train_targets.index.name
        assert train_inputs.columns.equals(test_inputs.columns)

        if use_fold:
            train_inputs = make_fold(c, train_inputs)
            train_targets["fold"] = train_inputs["fold"]

        setattr(self, f"train_{c.global_params.data}_inputs", train_inputs)
        setattr(self, f"train_{c.global_params.data}_targets", train_targets)
        setattr(self, f"test_{c.global_params.data}_inputs", test_inputs)


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data and c.settings.n_debug_data > 0:
        df = df.sample(n=c.settings.n_debug_data, random_state=c.global_params.seed).reset_index(drop=True)
    return df
