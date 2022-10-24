import logging

from .run_loop import train_fold_nn, train_fold_tabnet

log = logging.getLogger(__name__)


class ObjectiveTabnet:
    def __init__(self, c, input, fold):
        self.c = c
        self.input = input
        self.fold = fold

        self.best_preds_df = None
        self.best_valid_label_df = None
        self.best_inference_df = None

        self.preds_df = None
        self.valid_label_df = None
        self.inference_df = None

    def __call__(self, trial):
        self.c.model_params.tabnet.n_d = trial.suggest_int("n_d", 8, 64, step=8)
        self.c.model_params.tabnet.n_steps = trial.suggest_int("n_steps", 1, 5)
        self.c.model_params.tabnet.n_independent = trial.suggest_int("n_independent", 1, 3)
        self.c.model_params.tabnet.gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
        self.c.model_params.tabnet.mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])

        self.preds_df, self.valid_label_df, loss, self.inference_df = train_fold_tabnet(self.c, self.input, self.fold)

        return loss

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_preds_df = self.preds_df
            self.best_valid_label_df = self.valid_label_df
            self.best_inference_df = self.inference_df


class ObjectiveNN:
    def __init__(self, c, input, fold, device):
        self.c = c
        self.input = input
        self.fold = fold
        self.device = device

        self.best_preds_df = None
        self.best_valid_label_df = None
        self.best_inference_df = None

        self.preds_df = None
        self.valid_label_df = None
        self.inference_df = None

    def __call__(self, trial):
        if self.c.model_params.model == "one_d_cnn":
            self.c.model_params.one_d_cnn.dropout_1 = trial.suggest_float("dropout_1", 0.0, 0.3, step=0.05)
            self.c.model_params.one_d_cnn.dropout_2 = trial.suggest_float("dropout_2", 0.0, 0.5, step=0.05)
            self.c.model_params.one_d_cnn.dropout_3 = trial.suggest_float("dropout_3", 0.0, 0.3, step=0.05)
            # self.c.model_params.one_d_cnn.weight_norm = trial.suggest_categorical("weight_norm", [True, False])

        self.preds_df, self.valid_label_df, loss, self.inference_df = train_fold_nn(
            self.c, self.input, self.fold, self.device
        )

        return loss

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_preds_df = self.preds_df
            self.best_valid_label_df = self.valid_label_df
            self.best_inference_df = self.inference_df
