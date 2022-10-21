import torch
import torch.cuda.amp as amp
import torch.nn as nn


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        in_out_channels = c.model_params.model_input
        hidden_channels = in_out_channels * 3 // 5

        self.encoder = nn.Sequential(
            nn.Linear(in_out_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(True),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels // 4, hidden_channels // 2),
            nn.ReLU(True),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, in_out_channels),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.encoder(x)
            x = self.decoder(x)

        return x
