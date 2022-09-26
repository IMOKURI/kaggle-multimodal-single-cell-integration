import lightgbm as lgb
from torch.utils.data import DataLoader

from .datasets.image import ImageBaseDataset, get_transforms
from .datasets.table import TableBaseDataset


def make_dataset_nn(c, df, label=True, label_df=None, transform="simple"):
    if c.model_params.dataset == "image_base":
        ds = ImageBaseDataset(c, df, label, get_transforms(c, transform))
    elif c.model_params.dataset == "table_base":
        ds = TableBaseDataset(c, df, label, label_df)

    else:
        raise Exception("Invalid dataset.")
    return ds


def make_dataset(c, df, label_df=None, lightgbm=False):
    if label_df is None:
        if c.settings.n_class == 1:
            label_name = [c.settings.label_name]
        else:
            label_name = [f"{c.settings.label_name}_{n}" for n in range(c.settings.n_class)]

        try:
            labels = df[label_name].to_numpy().squeeze()
        except KeyError:
            labels = None
    else:
        label_name = []
        labels = None

    for col in ["index", "fold", "group_fold", "time_fold", c.settings.label_name, c.cv_params.group_name] + label_name:
        try:
            df = df.drop(col, axis=1)
            if label_df is not None:
                label_df = label_df.drop(col, axis=1)
        except KeyError:
            pass

    if label_df is not None:
        labels = label_df.to_numpy()

    if lightgbm:
        ds = lgb.Dataset(data=df, label=labels)
        return ds, df.to_numpy()

    else:
        ds = df.to_numpy()
        return ds, labels


def make_dataloader(c, ds, shuffle, drop_last):
    dataloader = DataLoader(
        ds,
        batch_size=c.training_params.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader
