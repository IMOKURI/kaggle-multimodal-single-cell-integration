import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn

log = logging.getLogger(__name__)


def weight_norm(layer, dim=None, enabled=True):
    return nn.utils.weight_norm(layer, dim=dim) if enabled else layer


# https://www.kaggle.com/c/lish-moa/discussion/202256
# https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
class OneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.model_params.model_input

        self.hidden_size = 1024
        self.ch_1 = 128
        self.ch_2 = 384
        self.ch_3 = 384

        self.ch_po_1 = int(self.hidden_size / self.ch_1 / 2)
        self.ch_po_2 = int(self.hidden_size / self.ch_1 / 2 / 2) * self.ch_3

        self.expand = nn.Sequential(
            nn.BatchNorm1d(self.input), weight_norm(nn.Linear(self.input, self.hidden_size)), nn.CELU(0.06)
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(self.ch_1),
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=False)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=self.ch_po_1),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True)),
            nn.ReLU(),
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.ch_po_2),
            nn.Dropout(0.1),
            nn.utils.weight_norm(nn.Linear(self.ch_po_2, c.settings.n_class)),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.conv2(x) * x

            x = self.max_po_c2(x)
            x = self.flt(x)

            x = self.head(x).squeeze(1)

        return x


# https://www.kaggle.com/ghostcxs/fork-prediction-including-spatial-info-with-conv1d
class SmallOneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.model_params.model_input

        self.hidden_size = 256
        self.ch_1 = 1
        self.ch_2 = 16
        self.ch_3 = 64
        self.head_size_1 = int(self.hidden_size / 4 / 4 / 2) * self.ch_3  # 512
        self.head_size_2 = 128
        self.head_size_3 = 32

        self.expand = nn.Sequential(
            nn.Linear(self.input, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),
            nn.Conv1d(self.ch_2, self.ch_2, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),
            nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),
            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),
            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),
        )

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(self.head_size_1, self.head_size_1),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_size_1, self.head_size_2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_size_2, self.head_size_3),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_size_3, c.settings.n_class),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.flt(x)

            x = self.head(x).squeeze(1)

        return x
