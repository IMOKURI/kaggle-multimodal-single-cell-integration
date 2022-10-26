import logging
import math
import os
import random
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Optional

import git
import numpy as np
import pandas as pd
import psutil
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

log = logging.getLogger(__name__)

_ALWAYS_CATCH = False


def basic_logger():
    logging.basicConfig(
        # filename=__file__.replace('.py', '.log'),
        level=logging.getLevelName("INFO"),
        format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
    )


def set_always_catch(catch: bool):
    global _ALWAYS_CATCH
    _ALWAYS_CATCH = catch


def in_kaggle() -> bool:
    return "kaggle_web_client" in sys.modules


@contextmanager
def catch_everything_in_kaggle(name: Optional[str] = None):
    try:
        yield
    except Exception:
        msg = f"WARNINGS: exception occurred in {name or '(unknown)'}: {traceback.format_exc()}"
        warnings.warn(msg)
        log.warning(msg)

        if in_kaggle() or _ALWAYS_CATCH:
            # ...catch and suppress if this is executed in kaggle
            pass
        else:
            # re-raise if this is executed outside of kaggle
            raise


def choice_seed(c):
    try:
        method = len(c.global_params.method)
    except ConfigAttributeError:
        method = 0
    try:
        return c.global_params.seed + method
    except ConfigAttributeError:
        with open_dict(c):
            c.global_params.seed = random.randint(1, 10_000)
        return c.global_params.seed


def fix_seed(seed=42):
    log.info(f"Fix seed: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def basic_environment_info():
    log.info(f"Output directory: {HydraConfig.get().run.dir}")
    log.info(f"Python version: {sys.version}")
    try:
        log.info(f"PyTorch version: {torch.__version__}")
    except Exception:
        pass


def debug_settings(c):
    if c.settings.debug:
        log.info("Enable debug mode.")
        c.wandb.enabled = False
        c.settings.print_freq = c.settings.print_freq // 10
        if c.cv_params.n_fold > 3:
            c.cv_params.n_fold = 3
        c.training_params.epoch = 1


def gpu_settings(c):
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = c.settings.gpus
        log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    except ConfigAttributeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"torch device: {device}, device count: {torch.cuda.device_count()}")
    return device


def df_stats(df):
    stats = []
    for col in df.columns:
        try:
            stats.append(
                (
                    col,
                    df[col].nunique(),
                    df[col].value_counts().index[0],
                    df[col].value_counts().values[0],
                    df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                    df[col].isnull().sum(),
                    df[col].isnull().sum() * 100 / df.shape[0],
                    df[col].dtype,
                )
            )
        except TypeError:
            log.warning(f"Skip column. {col}: {df[col].dtype}")
    return pd.DataFrame(stats, columns=["カラム名", "ユニーク値数", "最頻値", "最頻値の出現回数", "最頻値の割合", "欠損値の数", "欠損値の割合", "タイプ"])


def analyze_column(input_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(input_series):
        return "numeric"
    else:
        return "categorical"


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        log.info(f"Mem. usage decreased to {str(end_mem)[:3]}Mb: {num_reduction[:2]}% reduction")

    return df_out


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def get_gpu_memory(cmd_path="nvidia-smi", target_properties=("memory.total", "memory.used")):
    """
    ref: https://www.12-technology.com/2022/01/pythongpu.html
    Returns
    -------
    gpu_total : ndarray,  "memory.total"
    gpu_used: ndarray, "memory.used"
    """

    # format option
    format_option = "--format=csv,noheader,nounits"

    cmd = "%s --query-gpu=%s %s" % (cmd_path, ",".join(target_properties), format_option)

    # Command execution in sub-processes
    cmd_res = subprocess.check_output(cmd, shell=True)

    gpu_lines = cmd_res.decode().split("\n")[0].split(", ")

    gpu_total = int(gpu_lines[0]) / 1024
    gpu_used = int(gpu_lines[1]) / 1024

    gpu_total = np.round(gpu_used, 1)
    gpu_used = np.round(gpu_used, 1)
    return gpu_total, gpu_used


@contextmanager
def timer(name, gpu=True):
    s = time.time()
    p = psutil.Process(os.getpid())
    ram_m0 = p.memory_info().rss / 2.0**30
    if gpu:
        gpu_m0 = get_gpu_memory()[0]

    yield

    elapsed = time.time() - s
    ram_m1 = p.memory_info().rss / 2.0**30
    if gpu:
        gpu_m1 = get_gpu_memory()[0]

    ram_delta = ram_m1 - ram_m0
    ram_sign = "+" if ram_delta >= 0 else "-"
    ram_delta = math.fabs(ram_delta)
    if gpu:
        gpu_delta = gpu_m1 - gpu_m0
        gpu_sign = "+" if gpu_delta >= 0 else "-"
        gpu_delta = math.fabs(gpu_delta)

    ram_message = f"{ram_m1:.1f}GB({ram_sign}{ram_delta:.1f}GB)"
    if gpu:
        gpu_message = f"{gpu_m1:.1f}GB({gpu_sign}{gpu_delta:.1f}GB)"
        log.info(f"[{name}] {asMinutes(elapsed)} ({elapsed:.3f}s), RAM: {ram_message}, GPU Mem: {gpu_message}")
    else:
        log.info(f"[{name}] {asMinutes(elapsed)} ({elapsed:.3f}s), RAM: {ram_message}")


def use_first_element_as_int(x):
    try:
        return int(x)
    except:
        return int(x[0])


def use_last_element_as_int(x):
    try:
        return int(x)
    except:
        return int(x[-1])


def compute_grad_norm(parameters, norm_type=2.0):
    """Refer to torch.nn.utils.clip_grad_norm_"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )

    return total_norm


def setup_wandb(c):
    if c.wandb.enabled:
        os.makedirs(os.path.abspath(c.wandb.dir), exist_ok=True)
        c_dict = OmegaConf.to_container(c.params, resolve=True)
        c_dict["commit"] = get_commit_hash(c.settings.dirs.working)
        run = wandb.init(
            entity=c.wandb.entity,
            project=c.wandb.project,
            dir=os.path.abspath(c.wandb.dir),
            config=c_dict,
            group=c.wandb.group,
        )
        log.info(f"WandB initialized. name: {run.name}, id: {run.id}")
    else:
        run = wandb.init(entity=c.wandb.entity, project=c.wandb.project, mode="disabled")
    return run


def teardown_wandb(c, run, loss, score):
    if c.wandb.enabled:
        wandb.summary["loss"] = loss
        artifact = wandb.Artifact(
            c.model_params.model_name.replace("/", "-"),
            # c.model_params.model.replace("/", "-"),
            type="model",
        )
        artifact.add_dir(".")
        run.log_artifact(artifact)
        log.info(f"WandB recorded. name: {run.name}, id: {run.id}")
        wandb.alert(
            title="Result",
            text=f"Run name: {run.name}, id: {run.id}, score: {score:.5}, loss: {loss:.5f}",
            level=wandb.AlertLevel.INFO,
        )


def get_commit_hash(dir_):
    repo = git.Repo(dir_, search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha
