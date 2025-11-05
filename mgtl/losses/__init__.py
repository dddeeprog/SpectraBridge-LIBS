#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mgtl.losses 子包导出

公开 API：
- coral_loss：Deep CORAL 二阶统计对齐损失
- FocalLoss ：多类 Focal Loss（支持按类/标量 alpha、γ 聚焦参数）

用法：
    from mgtl.losses import coral_loss, FocalLoss
"""
from __future__ import annotations

__all__ = [
    "coral_loss",
    "FocalLoss",
]

# 显式导出，便于静态分析与自动补全
from .coral import coral_loss
from .focal import FocalLoss
