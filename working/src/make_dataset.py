import glob
import os

import albumentations as A
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


def make_dataset_nn(c, df, label=True, transform="simple"):
    if c.model_params.dataset == "base":
        ds = BaseDataset(c, df, label, get_transforms(c, transform))

    else:
        raise Exception("Invalid dataset.")
    return ds


def make_dataset(c, train_df, valid_df, train_label_df=None, valid_label_df=None, is_training=True, lightgbm=False):
    if train_label_df is None or valid_label_df is None:
        if c.settings.n_class == 1:
            labels = [c.settings.label_name]
        else:
            labels = [f"{c.settings.label_name}_{n}" for n in range(c.settings.n_class)]

        if is_training:
            train_labels = train_df[labels].to_numpy().squeeze()
            valid_labels = valid_df[labels].to_numpy().squeeze()
        else:
            train_labels = None
            valid_labels = None
    else:
        labels = []
        train_labels = None
        valid_labels = None

    for col in ["index", "fold", "group_fold", "time_fold", c.settings.label_name] + labels:
        try:
            train_df = train_df.drop(col, axis=1)
            valid_df = valid_df.drop(col, axis=1)
            if train_label_df is not None and valid_label_df is not None:
                train_label_df = train_label_df.drop(col, axis=1)
                valid_label_df = valid_label_df.drop(col, axis=1)
        except KeyError:
            pass

    if train_label_df is not None and valid_label_df is not None:
        train_labels = train_label_df.to_numpy()
        valid_labels = valid_label_df.to_numpy()

    if lightgbm:
        train_ds = lgb.Dataset(data=train_df, label=train_labels)
        valid_ds = lgb.Dataset(data=valid_df, label=valid_labels)

        return train_ds, train_df.to_numpy(), valid_ds, valid_df.to_numpy()

    else:
        train_ds = train_df.to_numpy()
        valid_ds = valid_df.to_numpy()

        return train_ds, train_labels, valid_ds, valid_labels


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


class BaseDataset(Dataset):
    def __init__(self, c, df, label=True, transform=None):
        # self.df = df
        self.image_ids = df["image_id"]
        self.use_label = label
        self.transform = transform

        self.preprocess_dir = c.settings.dirs.preprocess
        self.image_size = c.model_params.model_input

        if self.use_label:
            if c.settings.n_class == 1:
                # labels = [c.settings.label_name]
                # self.labels = df[c.settings.label_name].to_numpy()
                self.labels = df[c.settings.label_name]
            else:
                # labels = [f"{c.settings.label_name}_{n}" for n in range(c.settings.n_class)]
                # labels = [f"{c.settings.label_name}_{n}" for n in df[c.settings.label_name].unique()]
                # self.labels = df[labels].to_numpy()
                self.labels = df[c.settings.label_name]
        else:
            # labels = []
            ...

        # for col in [
        #     # "primary_key",
        #     "time_id",
        #     "fold",
        #     "group_fold",
        #     "time_fold",
        #     c.settings.label_name,
        # ] + labels:
        #     try:
        #         df = df.drop(col, axis=1)
        #     except KeyError:
        #         pass

        # self.features = df.to_numpy()

    def __len__(self):
        # return len(self.df)
        # return len(self.features)
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.use_label:
            label = self.labels[idx]
            if label == 2:  # Unknown
                image_path = f"{self.preprocess_dir}/other/{image_id}_{self.image_size}.npy"
            else:
                image_path = f"{self.preprocess_dir}/train/{image_id}_{self.image_size}.npy"
        else:
            image_path = f"{self.preprocess_dir}/test/{image_id}_{self.image_size}.npy"

        image = np.load(image_path, allow_pickle=True)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.use_label:
            # label = torch.tensor(self.labels[idx]).float()
            label = torch.tensor(label).long()
            return image, label
        return image


def get_transforms(c, transform):
    if transform == "light":
        return A.Compose(
            [
                A.Resize(c.model_params.model_input, c.model_params.model_input),
                # A.RandomResizedCrop(c.model_params.model_input, c.model_params.model_input),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                # A.CoarseDropout(p=0.5),
                # A.Cutout(p=0.5),
                A.Normalize(),
                ToTensorV2(),  # to channel first.
            ]
        )

    elif transform == "simple":
        return A.Compose(
            [
                A.Resize(c.model_params.model_input, c.model_params.model_input),
                # A.CenterCrop(c.model_params.model_input, c.model_params.model_input),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    else:
        return None
