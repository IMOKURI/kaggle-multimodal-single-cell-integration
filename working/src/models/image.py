import torch
import torch.cuda.amp as amp
import torch.nn as nn


class ImageBaseModel(nn.Module):
    def __init__(self, c, pretrained=True):
        super().__init__()
        # self.model = timm.create_model(c.model_params.model_name, pretrained=pretrained, num_classes=c.settings.n_class)
        self.model = None

    def forward(self, x):
        x = self.model(x)
        return x
