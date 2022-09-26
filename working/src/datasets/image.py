import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class ImageBaseDataset(Dataset):
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
