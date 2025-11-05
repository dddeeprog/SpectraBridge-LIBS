#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seed.py：统一随机种子与确定性开关

功能：
- set_seed(seed, deterministic=True)：统一设置 Python / NumPy / PyTorch 随机种子；
- 提供 DataLoader 的 worker_init_fn（seed_worker）与生成器（get_generator）；
- 控制 cuDNN 的 benchmark / deterministic 选项，兼顾稳定性与吞吐。

建议：
- 训练阶段若更重视复现性，设 deterministic=True, benchmark=False；
- 若更重视速度，可设 deterministic=False, benchmark=True（轻微非确定性）。
"""
from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42,
             deterministic: bool = True,
             benchmark: bool = False) -> None:
    """统一设置随机种子与确定性。

    参数：
    - seed: 全局随机种子
    - deterministic: 是否启用确定性（cudnn.deterministic=True；可能略降速）
    - benchmark: 是否让 cuDNN 对卷积等做算法搜索（与确定性冲突，通常在 deterministic=False 时打开）
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 行为
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark and not deterministic)


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 worker_init_fn：确保每个 worker 的 NumPy 随机数不同且可复现。
    用法：DataLoader(..., worker_init_fn=seed_worker, generator=get_generator(seed))
    """
    # PyTorch 会把 base_seed 派生到每个 worker；这里同步到 numpy/python
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed)
    random.seed(base_seed)


def get_generator(seed: int) -> torch.Generator:
    """返回一个带固定种子的 PyTorch 生成器，用于 DataLoader 随机采样等。"""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
