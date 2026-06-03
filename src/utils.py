"""通用工具:种子、日志、配置、设备管理。"""
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """固定所有随机种子,保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logger(name: str = "qsba", level: str = "INFO",
                 log_file: str = None) -> logging.Logger:
    """设置 logger,可同时输出到 console 和文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers = []  # 清空,避免重复

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_device(prefer: str = "cuda") -> torch.device:
    """选设备。"""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    """确保目录存在。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 归一化。"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)
