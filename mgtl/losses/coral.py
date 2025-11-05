#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep CORAL 损失（域自适应对齐二阶统计）

参考思想：
- 将源域/目标域的批特征 z_s, z_t ∈ R[B, D] 做中心化，计算协方差矩阵 C_s/C_t；
- 以 Frobenius 范数最小化两者差异：||C_s - C_t||_F^2；
- 可选加入均值对齐项（L2）：||μ_s - μ_t||_2^2；
- 为避免与维度/尺度强绑定，通常对损失做归一化（除以 D^2 或 4D^2 等常见配方）。

用法：
  loss = coral_loss(z_s, z_t, mean_align_weight=0.0, normalize='d2')
"""
from __future__ import annotations
from typing import Literal

import torch
import torch.nn.functional as F


def _covariance(z: torch.Tensor) -> torch.Tensor:
    """计算批协方差矩阵（无偏/有偏影响很小，这里采用 1/(B-1)）。
    输入：z ∈ R[B, D]
    输出：C ∈ R[D, D]
    """
    if z.dim() != 2:
        raise ValueError(f"coral_loss 期望特征形状 [B,D]，收到 {tuple(z.shape)}")
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)  # 中心化
    # 1/(B-1) * Z^T Z
    cov = (zc.t() @ zc) / max(B - 1, 1)
    return cov


def coral_loss(z_s: torch.Tensor,
               z_t: torch.Tensor,
               *,
               mean_align_weight: float = 0.0,
               normalize: Literal['none', 'd', 'd2', 'frob'] = 'd2') -> torch.Tensor:
    """Deep CORAL：二阶统计对齐 +（可选）均值对齐。

    参数：
    - z_s, z_t: 源/目标特征，形状 [B, D]
    - mean_align_weight: 均值对齐项的权重（0 表示不加）
    - normalize: 损失归一化方式
        'none' : 不缩放
        'd'    : 除以 D
        'd2'   : 除以 D^2（常见做法，弱化维度影响）
        'frob' : 对差矩阵的 Frobenius 范数再做 1/D 缩放
    返回：
    - 标量损失张量（device 与输入一致）
    """
    if z_s.shape != z_t.shape:
        raise ValueError(f"源/目标特征形状需一致，收到 {tuple(z_s.shape)} vs {tuple(z_t.shape)}")

    D = z_s.shape[1]
    Cs = _covariance(z_s)
    Ct = _covariance(z_t)
    diff = Cs - Ct

    # 二阶统计项
    if normalize == 'frob':
        loss_2nd = torch.norm(diff, p='fro') ** 2 / max(D, 1)
    else:
        loss_2nd = (diff ** 2).sum()
        if normalize == 'd':
            loss_2nd = loss_2nd / max(D, 1)
        elif normalize == 'd2':
            loss_2nd = loss_2nd / max(D * D, 1)

    # 可选：均值对齐
    if mean_align_weight > 0.0:
        ms = z_s.mean(dim=0)
        mt = z_t.mean(dim=0)
        loss_mean = F.mse_loss(ms, mt, reduction='sum') / max(D, 1)
        loss = loss_2nd + mean_align_weight * loss_mean
    else:
        loss = loss_2nd

    return loss
