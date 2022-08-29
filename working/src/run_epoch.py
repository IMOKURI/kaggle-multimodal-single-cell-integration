import gc
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp

# from memory_profiler import profile

from .utils import AverageMeter, compute_grad_norm, timeSince

log = logging.getLogger(__name__)


def train_epoch(c, train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device, verbose=False):
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for step, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with amp.autocast(enabled=c.settings.amp):
            y_preds = model(features)

            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)
            loss = loss / c.training_params.gradient_acc_step

        scaler.scale(loss).backward()

        if (step + 1) % c.training_params.gradient_acc_step == 0:
            scaler.unscale_(optimizer)

            # error_if_nonfinite に関する warning を抑止する
            # pytorch==1.10 で不要となりそう
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c.training_params.max_grad_norm  # , error_if_nonfinite=False
                )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            grad_norm = compute_grad_norm(model.parameters())

        if c.training_params.scheduler == "CosineAnnealingWarmupRestarts":
            last_lr = scheduler.get_lr()[0]
        else:
            last_lr = scheduler.get_last_lr()[0]

        # end = time.time()
        if verbose and (step % c.settings.print_freq == 0 or step == (len(train_loader) - 1)):
            log.info(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {losses.avg:.4f} "
                f"Grad: {grad_norm:.4f} "
                f"LR: {last_lr:.2e}  "
            )

    return losses.avg


def validate_epoch(c, valid_loader, model, criterion, device, verbose=False):
    model.eval()
    losses = AverageMeter()

    size = len(valid_loader.dataset)
    if c.settings.n_class == 1:
        preds = np.zeros((size,))
    else:
        preds = np.zeros((size, c.settings.n_class))
    start = time.time()

    for step, (features, labels) in enumerate(valid_loader):
        features = features.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.inference_mode():
            # with torch.no_grad():
            y_preds = model(features)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        begin = step * c.training_params.batch_size
        end = begin + batch_size
        if c.settings.n_class == 1:
            preds[begin:end] = y_preds.squeeze().to("cpu").numpy()
        elif c.settings.n_class > 1:
            preds[begin:end, :] = y_preds.softmax(1).to("cpu").numpy()
            # preds[begin:end] = y_preds[:, -1].squeeze().to("cpu").numpy()
        else:
            raise Exception("Invalid n_class.")

        # end = time.time()
        if verbose and (step % c.settings.print_freq == 0 or step == (len(valid_loader) - 1)):
            log.info(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    return losses.avg, preds


# @profile
def inference_epoch(c, inference_loader, model, device):
    model.eval()

    size = len(inference_loader.dataset)
    preds = np.zeros((size, c.settings.n_class), dtype=np.float32).squeeze()
    # start = time.time()

    for step, features in enumerate(inference_loader):
        features = features.to(device)
        batch_size = features.size(0)

        with torch.inference_mode():
            # with torch.no_grad():
            # y_preds = model(images)
            y_preds = model(features)

        begin = step * c.training_params.batch_size
        end = begin + batch_size
        if c.settings.n_class == 1:
            preds[begin:end] = y_preds.squeeze().to("cpu").numpy()
        elif c.settings.n_class > 1:
            preds[begin:end, :] = y_preds.softmax(1).to("cpu").numpy()
            # preds[begin:end] = y_preds[:, -1].squeeze().to("cpu").numpy()
        else:
            raise Exception("Invalid n_class.")

        # end = time.time()
        # if step % c.settings.print_freq == 0 or step == (len(inference_loader) - 1):
        #     log.info(
        #         f"EVAL: [{step}/{len(inference_loader)}] "
        #         f"Elapsed {timeSince(start, float(step + 1) / len(inference_loader)):s} "
        #     )

        del y_preds
        gc.collect()

    return preds
