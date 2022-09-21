import logging
import os

import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.preprocessing import LabelEncoder

from .make_fold import make_fold
from .preprocess import preprocess, preprocess_train_test
from .preprocesses.p001_dist_transformer import DistTransformer
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
            train = getattr(self, f"train_{c.global_params.data}_inputs")
            test = getattr(self, f"test_{c.global_params.data}_inputs")

            if c.global_params.data == "cite":
                test = pd.concat([self.test_cite_inputs, self.test_cite_inputs_day_2_donor_27678])
                test = test.dropna(axis=1)
                setattr(self, "test_cite_inputs", test)

            # 過学習のもとになりそうなカラムを削除
            # if c.global_params.data == "cite":
            #     train = train.drop(c.preprocess_params.cite_drop_cols, axis=1)
            #     test = test.drop(c.preprocess_params.cite_drop_cols, axis=1)

            train, test = preprocess_train_test(c, train, test)

            # train.to_pickle(os.path.join(c.settings.dirs.preprocess, f"train_{c.global_params.data}_inputs.pickle"))
            # test.to_pickle(os.path.join(c.settings.dirs.preprocess, f"test_{c.global_params.data}_inputs.pickle"))

            if c.global_params.data == "cite":
                ...
                # cite: target の column名を含む inputs の column を抽出する
                cols = []
                for target_col in self.train_cite_targets.columns:
                    cols += [col for col in self.train_cite_inputs.columns if target_col in col]

                # 明示的に温存する column を抽出する
                if c.preprocess_params.cite_no_pca_cols != []:
                    cols += c.preprocess_params.cite_no_pca_cols

                cols = list(set(cols))

                train = self.train_cite_inputs[cols]
                test = self.test_cite_inputs[cols]

                log.info(f"cite no pca data: {train.shape}")

                train.to_pickle(
                    os.path.join(c.settings.dirs.preprocess, f"train_{c.global_params.data}_no_pca_inputs.pickle")
                )
                test.to_pickle(
                    os.path.join(c.settings.dirs.preprocess, f"test_{c.global_params.data}_no_pca_inputs.pickle")
                )

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
    def __init__(self, c, use_fold=True, do_preprocess=True):
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

            elif "adversarial" in stem:
                setattr(self, "adversarial", df)

            else:
                raise Exception("Invalid filename")

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            # df = reduce_mem_usage(df)

            # setattr(self, stem, df)

        # train_inputs = train_inputs.join(metadata["cell_type_num"])
        # test_inputs = test_inputs.join(metadata["cell_type_num"])

        # RNA アノテーションによるカラム抽出
        # rna_annot = pd.read_table(os.path.join(c.settings.dirs.input, "catrapid_rnas.txt"))
        # rna_annot_human = rna_annot[rna_annot["species"] == "human"].reset_index(drop=True)
        # rna_biotype_dict = {ens: biotype for ens, biotype in zip(rna_annot_human["ensg"], rna_annot_human["biotype"])}
        # biotype_cols = [
        #     col
        #     for col in train_inputs.columns
        #     if (col.split("_")[0] in rna_biotype_dict) and (rna_biotype_dict[col.split("_")[0]] == "protein_coding")
        # ]
        # train_inputs = train_inputs[biotype_cols]
        # test_inputs = test_inputs[biotype_cols]

        # 過学習のもとになりそうなカラムを削除
        if do_preprocess:
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

        # Standard Scaler
        if c.global_params.method in ["ridge"]:
            ss = DistTransformer()
            ss.fit(pd.concat([train_inputs, test_inputs]))
            train_inputs = ss.transform(train_inputs)
            test_inputs = ss.transform(test_inputs)

        log.info(f"Data size, train: {train_inputs.shape}, target: {train_targets.shape}, test: {test_inputs.shape}")
        log.debug(f"input columns: {train_inputs.columns}")

        if use_fold:
            good_validation = self.adversarial[(self.adversarial["label"] == 0) & (self.adversarial["preds"] == 1)]
            train_inputs[c.settings.label_name] = 0
            train_inputs.loc[good_validation.index, :][c.settings.label_name] = 1
            test_inputs[c.settings.label_name] = 0

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
        self.cite_adversarial_oof = pd.DataFrame()
        self.cite_inference = None
        self.cite_oof = None
        self.multi_adversarial_oof = pd.DataFrame()
        self.multi_inference = None
        self.multi_oof = None
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

        # inference_df から leak データを除外する
        leak_27678_cell_id = self.metadata[
            (self.metadata["donor"] == 27678)
            & (self.metadata["technology"] == "citeseq")
            & (self.metadata["day"] == 2)
        ]["cell_id"]
        assert len(leak_27678_cell_id) == 7476

        log.info("Load CITEseq inference data.")
        for dir, weight in c.inference_params.cite_pretrained.items():
            log.info(f"  -> {dir}")
            path = os.path.join(c.settings.dirs.output, dir, "oof.pickle")
            df = pd.read_pickle(path)
            if self.cite_oof is None:
                self.cite_oof = pd.DataFrame(std(df.to_numpy()) * weight)
                self.cite_oof.index = df.index
                self.cite_oof.columns = df.columns
            else:
                self.cite_oof += std(df.to_numpy()) * weight

            path = os.path.join(c.settings.dirs.output, dir, "cite_inference.pickle")
            df = pd.read_pickle(path)
            df = df.drop(leak_27678_cell_id)
            if self.cite_inference is None:
                self.cite_inference = pd.DataFrame(std(df.to_numpy()) * weight)
                self.cite_inference.index = df.index
                self.cite_inference.columns = df.columns
            else:
                self.cite_inference += std(df.to_numpy()) * weight

        self.cite_inference = pd.concat(
            [
                self.cite_inference,
                pd.DataFrame(
                    np.zeros((len(leak_27678_cell_id), len(self.cite_inference.columns))), index=leak_27678_cell_id
                ),
            ]
        )

        log.info("Load Multiome inference data.")
        for dir, weight in c.inference_params.multi_pretrained.items():
            log.info(f"  -> {dir}")
            path = os.path.join(c.settings.dirs.output, dir, "oof.pickle")
            df = pd.read_pickle(path)
            if self.multi_oof is None:
                self.multi_oof = pd.DataFrame(std(df.to_numpy()) * weight)
                self.multi_oof.index = df.index
                self.multi_oof.columns = df.columns
            else:
                self.multi_oof += std(df.to_numpy()) * weight

            path = os.path.join(c.settings.dirs.output, dir, "multi_inference.pickle")
            df = pd.read_pickle(path)
            if self.multi_inference is None:
                self.multi_inference = pd.DataFrame(std(df.to_numpy()) * weight)
                self.multi_inference.index = df.index
                self.multi_inference.columns = df.columns
            else:
                self.multi_inference += std(df.to_numpy()) * weight


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data and c.settings.n_debug_data > 0:
        df = df.sample(n=c.settings.n_debug_data, random_state=c.global_params.seed).reset_index(drop=True)
    return df


def std(x):
    return (x - np.mean(x)) / np.std(x)
