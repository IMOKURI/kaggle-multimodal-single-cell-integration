import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

log = logging.getLogger(__name__)


# ====================================================
# Criterion
# ====================================================
def make_criterion(c):
    if c.training_params.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif c.training_params.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif c.training_params.criterion == "MSELoss":
        criterion = nn.MSELoss()
    elif c.training_params.criterion == "RMSELoss":
        criterion = RMSELoss()
    elif c.training_params.criterion == "PearsonCCLoss":
        criterion = PearsonCCLoss()
    elif c.training_params.criterion == "ConcordanceCCLoss":
        criterion = ConcordanceCCLoss()
    elif c.training_params.criterion == "LabelSmoothCrossEntropyLoss":
        criterion = LabelSmoothCrossEntropyLoss(smoothing=c.training_params.label_smoothing)
    elif c.training_params.criterion == "LabelSmoothBCEWithLogitsLoss":
        criterion = LabelSmoothBCEWithLogitsLoss(smoothing=c.training_params.label_smoothing)
    elif c.training_params.criterion == "MarginRankingLoss":
        criterion = nn.MarginRankingLoss(margin=c.training_params.margin)

    else:
        raise Exception("Invalid criterion.")
    return criterion


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, inputs, targets):
        loss = torch.sqrt(self.mse(inputs, targets) + self.eps)
        return loss


# https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/9
class PearsonCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=eps)

    def forward(self, inputs, targets):
        pcc = self.cos(inputs - inputs.mean(dim=1, keepdim=True), targets - targets.mean(dim=1, keepdim=True))
        # return 1.0 - pcc
        return 1.0 - pcc.mean()


# https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/8
class ConcordanceCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=eps)
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_mean = torch.mean(inputs)
        targets_mean = torch.mean(targets)

        inputs_var = torch.var(inputs)
        targets_var = torch.var(targets)

        inputs_std = torch.std(inputs)
        targets_std = torch.std(targets)

        pcc = self.cos(inputs - inputs.mean(dim=1, keepdim=True), targets - targets.mean(dim=1, keepdim=True))
        ccc = (2 * pcc * inputs_std * targets_std) / (
            inputs_var + targets_var + (targets_mean - inputs_mean) ** 2 + self.eps
        )
        return 1.0 - ccc


# https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch/blob/main/label_smothing_cross_entropy_loss.py
class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# https://www.kaggle.com/felipebihaiek/torch-continued-from-auxiliary-targets-smoothing
class LabelSmoothBCEWithLogitsLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothBCEWithLogitsLoss._smooth(targets, self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# ====================================================
# Optimizer
# ====================================================
def make_optimizer(c, model):
    if c.training_params.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=c.training_params.lr, weight_decay=c.training_params.weight_decay)
    elif c.training_params.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=c.training_params.lr, weight_decay=c.training_params.weight_decay)
    elif c.training_params.optimizer == "RAdam":
        optimizer = RAdam(model.parameters(), lr=c.training_params.lr, weight_decay=c.training_params.weight_decay)
    else:
        raise Exception("Invalid optimizer.")
    return optimizer


# ====================================================
# Scheduler
# ====================================================
def make_scheduler(c, optimizer, ds):
    num_data = len(ds)
    num_steps = (
        num_data // (c.training_params.batch_size * c.training_params.gradient_acc_step) * c.training_params.epoch + 2
    )

    if c.training_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=num_steps, T_mult=1, eta_min=c.training_params.min_lr, last_epoch=-1
        )
    elif c.training_params.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, min_lr=c.training_params.min_lr, patience=c.training_params.es_patience // 5, verbose=True
        )
    elif c.training_params.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=c.training_params.min_lr, last_epoch=-1)
    elif c.training_params.scheduler == "CosineAnnealingWarmupRestarts":
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_steps,
            max_lr=c.training_params.lr,
            min_lr=c.training_params.min_lr,
            warmup_steps=(num_steps // 10),
        )

    else:
        raise Exception("Invalid scheduler.")
    return scheduler
