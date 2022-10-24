import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn

log = logging.getLogger(__name__)


# https://www.kaggle.com/c/lish-moa/discussion/202256
# https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
class OneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.model_params.model_input

        if "tuning" not in c.global_params.method:
            c.model_params.one_d_cnn.hidden_size = c.model_params.one_d_cnn[c.global_params.data].hidden_size
            c.model_params.one_d_cnn.ch_1 = c.model_params.one_d_cnn[c.global_params.data].ch_1
            c.model_params.one_d_cnn.ch_2 = c.model_params.one_d_cnn[c.global_params.data].ch_2
            c.model_params.one_d_cnn.ch_3 = c.model_params.one_d_cnn[c.global_params.data].ch_3

        hidden_size = c.model_params.one_d_cnn.hidden_size
        self.ch_1 = c.model_params.one_d_cnn.ch_1
        self.ch_2 = c.model_params.one_d_cnn.ch_2
        self.ch_3 = c.model_params.one_d_cnn.ch_3
        dropout_1 = c.model_params.one_d_cnn.dropout_1
        dropout_2 = c.model_params.one_d_cnn.dropout_2
        dropout_3 = c.model_params.one_d_cnn.dropout_3
        self.weight_norm = c.model_params.one_d_cnn.weight_norm

        ch_po_1 = int(hidden_size / self.ch_1 / 2)
        ch_po_2 = int(hidden_size / self.ch_1 / 2 / 2) * self.ch_3

        self.expand = nn.Sequential(
            nn.BatchNorm1d(self.input),
            nn.Dropout(dropout_1),
            nn.utils.weight_norm(nn.Linear(self.input, hidden_size)),
            nn.SiLU(),
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(self.ch_1),
            nn.Dropout(dropout_1),
            self._norm(nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=False)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=ch_po_1),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(dropout_1),
            self._norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(dropout_2),
            self._norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(dropout_2),
            self._norm(nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True)),
            nn.LeakyReLU(),
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.BatchNorm1d(ch_po_2),
            nn.Dropout(dropout_3),
            self._norm(nn.Linear(ch_po_2, c.settings.n_class), dim=0),
        )

        if c.model_params.tf_initialization:
            self._tf_reinitialize()

    def _norm(self, layer, dim=None):
        return nn.utils.weight_norm(layer, dim=dim) if self.weight_norm else layer

    def _tf_reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.conv2(x) * x

            x = self.max_po_c2(x)
            x = self.flt(x)

            x = self.head(x).squeeze(1)

        return x
