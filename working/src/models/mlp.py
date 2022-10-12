import torch
import torch.cuda.amp as amp
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class MlpBaseModel(nn.Module):
    def __init__(self, c, tf_initialization=False):
        super().__init__()
        self.amp = c.settings.amp
        self.model_input = c.model_params.model_input
        self.model_output = c.settings.n_class
        diff = (self.model_output - self.model_input) // 3

        self.bn_1 = nn.BatchNorm1d(self.model_input)

        self.fc_1 = nn.Linear(self.model_input, self.model_input + diff)
        self.fc_2 = nn.Linear(self.model_input + diff, self.model_input + diff * 2)
        self.fc_3 = nn.Linear(self.model_input + diff * 2, self.model_output)

        if tf_initialization:
            self._tf_reinitialize()

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
            x = self.bn_1(x)

            x = swish(self.fc_1(x))
            x = swish(self.fc_2(x))
            x = self.fc_3(x)

        return x
