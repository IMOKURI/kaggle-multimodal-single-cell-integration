import logging
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
            train = (getattr(self, f"train_{c.global_params.data}_inputs"),)
            test = (getattr(self, f"test_{c.global_params.data}_inputs"),)

            # 過学習のもとになりそうなカラムを削除
            if c.global_params.data == "cite":
                train = train.drop(c.preprocess_params.cite_drop_cols, axis=1)
                test = test.drop(c.preprocess_params.cite_drop_cols, axis=1)

            train, test = preprocess_train_test(c, train, test)

            train.to_pickle(os.path.join(c.settings.dirs.preprocess, f"train_{c.global_params.data}_inputs.pickle"))
            test.to_pickle(os.path.join(c.settings.dirs.preprocess, f"test_{c.global_params.data}_inputs.pickle"))

            if c.global_params.data == "cite":
                ...
                # cite: target の column名を含む inputs の column を抽出する
                # cols = []
                # for target_col in self.train_cite_targets.columns:
                #     cols += [col for col in self.train_cite_inputs.columns if target_col in col]
                #
                # train = self.train_cite_inputs[cols]
                # test = self.test_cite_inputs[cols]
                #
                # log.info(f"cite no pca data: {train.shape}")
                #
                # train.to_pickle(
                #     os.path.join(c.settings.dirs.preprocess, f"train_{c.global_params.data}_no_pca_inputs.pickle")
                # )
                # test.to_pickle(
                #     os.path.join(c.settings.dirs.preprocess, f"test_{c.global_params.data}_no_pca_inputs.pickle")
                # )

                # cite: RNA Type ごとの和を特徴量とする
                # 他に、 mean, std, skew も
                # rna_annot = pd.read_table(os.path.join(c.settings.dirs.input, "catrapid_rnas.txt"))
                # rna_annot_human = rna_annot[rna_annot["species"] == "human"].reset_index(drop=True)
                # rna_biotype_dict = {
                #     ens: biotype for ens, biotype in zip(rna_annot_human["ensg"], rna_annot_human["biotype"])
                # }
                #
                # annot_cols = []
                # for co in [col.split("_")[0] for col in self.train_cite_inputs.columns]:
                #     try:
                #         annot_cols.append(rna_biotype_dict[co])
                #     except KeyError:
                #         annot_cols.append(co)
                #
                # train = self.train_cite_inputs
                # test = self.test_cite_inputs
                #
                # train.columns = annot_cols
                # test.columns = annot_cols
                # train.columns.name = "gene_id"
                # test.columns.name = "gene_id"
                #
                # train = train.T.groupby("gene_id").sum().T
                # test = test.T.groupby("gene_id").sum().T
                #
                # train.columns = [f"{col}_sum" for col in train.columns]
                # test.columns = [f"{col}_sum" for col in test.columns]
                #
                # train = train.dropna(axis=1)
                # test = test.dropna(axis=1)
                #
                # log.info(f"cite sum by RNA type: {train.shape}")  # 列数は 205
                #
                # train.to_pickle(
                #     os.path.join(
                #         c.settings.dirs.preprocess, f"train_{c.global_params.data}_sum_by_rna_type_inputs.pickle"
                #     )
                # )
                # test.to_pickle(
                #     os.path.join(
                #         c.settings.dirs.preprocess, f"test_{c.global_params.data}_sum_by_rna_type_inputs.pickle"
                #     )
                # )


class LoadData:
    def __init__(self, c, use_fold=True):
        self.c = c

        train_inputs = pd.DataFrame()
        train_targets = pd.DataFrame()
        test_inputs = pd.DataFrame()
        metadata = pd.DataFrame()

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
                if c.global_params.data == "multi":
                    id = stem.split("_")[2]
                    df.columns = [f"{id}_{col}" for col in df.columns]
                train_inputs = pd.concat([train_inputs, df], axis=1)
                train_inputs.index = df.index
                train_inputs.index.name = df.index.name
            elif "test" in stem:
                if c.global_params.data == "multi":
                    id = stem.split("_")[2]
                    df.columns = [f"{id}_{col}" for col in df.columns]
                test_inputs = pd.concat([test_inputs, df], axis=1)
                test_inputs.index = df.index
                test_inputs.index.name = df.index.name
            elif "metadata" in stem:
                if c.global_params.data == "cite":
                    metadata = df[df["technology"] == "citeseq"].set_index("cell_id")
                elif c.global_params.data == "multi":
                    metadata = df[df["technology"] == "multiome"].set_index("cell_id")

                # le = LabelEncoder()
                # metadata["cell_type_num"] = le.fit_transform(metadata["cell_type"].to_numpy())
                setattr(self, "metadata", metadata)
                # setattr(self, "metadata_cell_type_num", le.classes_)

            else:
                raise Exception("Invalid filename")

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            # df = reduce_mem_usage(df)

            # setattr(self, stem, df)

        # train_inputs = train_inputs.join(metadata["cell_type_num"])
        # test_inputs = test_inputs.join(metadata["cell_type_num"])

        # 過学習のもとになりそうなカラムを削除
        if c.global_params.data == "cite" and c.preprocess_params.cite_drop_cols != []:
            train_inputs = train_inputs.drop(c.preprocess_params.cite_drop_cols, axis=1)
            test_inputs = test_inputs.drop(c.preprocess_params.cite_drop_cols, axis=1)
        elif c.global_params.data == "multi" and c.preprocess_params.multi_drop_cols != []:
            train_inputs = train_inputs.drop(c.preprocess_params.multi_drop_cols, axis=1)
            test_inputs = test_inputs.drop(c.preprocess_params.multi_drop_cols, axis=1)
            log.info(
                f"Removed columns count: {pd.Series([col.split('_')[0] for col in c.preprocess_params.multi_drop_cols]).value_counts()}"
            )

        assert train_inputs.index.equals(train_targets.index)
        assert train_inputs.index.name == train_targets.index.name
        assert train_inputs.columns.equals(test_inputs.columns)

        log.info(f"Data size, train: {train_inputs.shape}, target: {train_targets.shape}, test: {test_inputs.shape}")
        log.debug(f"input columns: {train_inputs.columns}")

        if use_fold:
            train_inputs = make_fold(c, train_inputs)
            train_targets["fold"] = train_inputs["fold"]
            test_inputs = make_fold(c, test_inputs)

        setattr(self, f"train_{c.global_params.data}_inputs", train_inputs)
        setattr(self, f"train_{c.global_params.data}_targets", train_targets)
        setattr(self, f"test_{c.global_params.data}_inputs", test_inputs)


class PostprocessData:
    def __init__(self, c):
        self.c = c
        self.evaluation_ids = pd.DataFrame()
        self.metadata = pd.DataFrame()
        self.sample_submission = pd.DataFrame()
        self.cite_inference = pd.DataFrame()
        self.cite_oof = pd.DataFrame()
        self.multi_inference = pd.DataFrame()
        self.multi_oof = pd.DataFrame()
        self.train_cite_targets = pd.DataFrame()
        self.train_multi_targets = pd.DataFrame()

        for file_name in c.settings.postprocesses:
            stem = os.path.splitext(file_name)[0].replace("/", "__")
            extension = os.path.splitext(file_name)[1]

            original_file_path = os.path.join(c.settings.dirs.postprocess, file_name)
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

            setattr(self, stem, df)


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data and c.settings.n_debug_data > 0:
        df = df.sample(n=c.settings.n_debug_data, random_state=c.global_params.seed).reset_index(drop=True)
    return df
