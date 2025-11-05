#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mgtl：多模态/迁移学习工具包（S1 单模态光谱版）

对外统一导出常用 API，便于外部脚本：

    from mgtl import (
        NpyDataset, NpyDualIter,
        RegressionMetrics, ClassificationMetrics,
        set_seed, seed_worker, get_generator,
        get_logger, Progress,
        SpecEncoder, ClassifierHead, PerClassRegressors, GlobalRegressor,
        coral_loss, FocalLoss,
        Compose, build_default_train_augs,
    )
"""
from __future__ import annotations

__all__ = [
    # data
    "NpyDataset", "NpyDualIter",
    # utils
    "RegressionMetrics", "ClassificationMetrics",
    "set_seed", "seed_worker", "get_generator",
    "get_logger", "Progress", "human_time", "human_num",
    # models
    "SpecEncoder", "ClassifierHead", "PerClassRegressors", "GlobalRegressor",
    # losses
    "coral_loss", "FocalLoss",
    # transforms
    "Compose", "build_default_train_augs",
]

__version__ = "0.1.0"

# —— data ——
from .data import NpyDataset, NpyDualIter

# —— utils ——
from .utils.metrics import RegressionMetrics, ClassificationMetrics
from .utils.seed import set_seed, seed_worker, get_generator
from .utils.logging import get_logger, Progress, human_time, human_num

# —— models ——
from .models.spec_encoder import SpecEncoder
from .models.classifier import ClassifierHead
from .models.regressors import PerClassRegressors, GlobalRegressor

# —— losses ——
from .losses.coral import coral_loss
from .losses.focal import FocalLoss

# —— transforms（可选增广） ——
from .transforms import Compose, build_default_train_augs
