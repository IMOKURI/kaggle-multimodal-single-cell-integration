import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class MlpBaseModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.model_input = c.model_params.model_input
        self.model_output = c.settings.n_class
        diff = (self.model_output - self.model_input) // 3

        self.bn_1 = nn.BatchNorm1d(self.model_input)

        if c.global_params.data == "multi":
            self.fc_1 = nn.Linear(self.model_input, self.model_input)
            self.fc_2 = nn.Linear(self.model_input, self.model_input + diff)
            self.fc_3 = nn.Linear(self.model_input + diff, self.model_output)
        else:
            self.fc_1 = nn.Linear(self.model_input, self.model_input + diff)
            self.fc_2 = nn.Linear(self.model_input + diff, self.model_input + diff * 2)
            self.fc_3 = nn.Linear(self.model_input + diff * 2, self.model_output)

        if c.model_params.tf_initialization:
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


class MlpDropoutModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.model_input = c.model_params.model_input
        self.model_output = c.settings.n_class
        hidden_size = self.model_input * 3 // 2

        self.batch_norm1 = nn.BatchNorm1d(self.model_input)
        self.dense1 = nn.utils.weight_norm(nn.Linear(self.model_input, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, self.model_output))

        if c.model_params.tf_initialization:
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
            x = self.batch_norm1(x)
            x = F.relu(self.dense1(x))

            x = self.batch_norm2(x)
            x = self.dropout2(x)
            x = F.relu(self.dense2(x))

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = self.dense3(x)

        return x


class MlpResnetModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.model_input = c.model_params.model_input
        self.model_output = c.settings.n_class
        hidden_size = self.model_input // 3

        self.batch_norm1 = nn.BatchNorm1d(self.model_input)
        self.dense1 = nn.utils.weight_norm(nn.Linear(self.model_input, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(self.model_input + hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.utils.weight_norm(nn.Linear(self.model_input + hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size * 2, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout4 = nn.Dropout(0.3)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size * 2, self.model_output))

        if c.model_params.tf_initialization:
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
            x1 = self.batch_norm1(x)
            x1 = F.leaky_relu(self.dense1(x1))
            x = torch.cat([x, x1], 1)

            x2 = self.batch_norm2(x)
            x2 = self.dropout2(x2)
            x2 = F.leaky_relu(self.dense2(x2))
            x = torch.cat([x1, x2], 1)

            x3 = self.batch_norm3(x)
            x3 = self.dropout3(x3)
            x3 = F.leaky_relu(self.dense3(x3))
            x = torch.cat([x2, x3], 1)

            x4 = self.batch_norm4(x)
            x4 = self.dropout4(x4)
            x = self.dense4(x4)

        return x
