import torch
from torch.utils.data import Dataset


class TableBaseDataset(Dataset):
    def __init__(self, c, df, label, label_df=None):
        self.use_label = label

        if label_df is None:
            if c.settings.n_class == 1:
                self.label_name = [c.settings.label_name]
            else:
                self.label_name = [f"{c.settings.label_name}_{n}" for n in range(c.settings.n_class)]

            try:
                self.labels = df[self.label_name].to_numpy().squeeze()
            except KeyError:
                self.labels = None
        else:
            self.label_name = []
            self.labels = None

        for col in [
            "index",
            "fold",
            "group_fold",
            "time_fold",
            c.settings.label_name,
            c.cv_params.group_name,
        ] + self.label_name:
            try:
                df = df.drop(col, axis=1)
                if label_df is not None:
                    label_df = label_df.drop(col, axis=1)
            except KeyError:
                pass

        if label_df is not None:
            self.labels = label_df.to_numpy()

        self.ds = df.to_numpy()

    def __len__(self):
        # return len(self.df)
        # return len(self.features)
        return len(self.ds)

    def __getitem__(self, idx):
        features = torch.tensor(self.ds[idx]).float()
        if self.use_label:
            label = torch.tensor(self.labels[idx]).float()
            return features, label
        return features
