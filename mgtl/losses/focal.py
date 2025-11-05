#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多类 Focal Loss（用于类别不平衡的分类头）

公式：
  FL(p_t) = - α_t * (1 - p_t)^γ * log(p_t)
其中：
  - p_t 为目标类别的预测概率
  - α_t 可为标量（全类同权）或向量（按类权重）
  - γ≥0 调整易分类样本的降权力度

用法：
  crit = FocalLoss(alpha=[0.2,0.3,0.5], gamma=2.0)
  loss = crit(logits, target)  # target 为 LongTensor（0..C-1）
"""
from __future__ import annotations
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Iterable[float]] | float = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 eps: float = 1e-8):
        """
        参数：
        - alpha: None | 标量 | 长度为 C 的可迭代（每类权重）。
        - gamma: 焦点参数 γ≥0，越大越强调难样本。
        - reduction: 'none' | 'mean' | 'sum'
        - eps: 数值稳定性项，避免 log(0)
        """
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = eps

        if alpha is None:
            self.register_buffer('alpha', None, persistent=False)
        elif isinstance(alpha, (float, int)):
            self.register_buffer('alpha', torch.tensor(float(alpha)), persistent=False)
        else:
            a = torch.tensor(list(alpha), dtype=torch.float32)
            self.register_buffer('alpha', a, persistent=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向：
        - logits: [B, C] 未归一化分类分数
        - target: [B]    LongTensor（0..C-1）
        返回：标量损失或逐样本损失（取决于 reduction）
        """
        if logits.dim() != 2:
            raise ValueError(f"FocalLoss 期望 logits 形状 [B,C]，收到 {tuple(logits.shape)}")
        if target.dim() != 1:
            raise ValueError(f"FocalLoss 期望 target 形状 [B]，收到 {tuple(target.shape)}")
        B, C = logits.shape

        # 概率与目标概率
        log_p = F.log_softmax(logits, dim=1)                # [B,C]
        p = log_p.exp()                                     # [B,C]
        log_pt = log_p[torch.arange(B, device=logits.device), target]   # [B]
        pt = p[torch.arange(B, device=logits.device), target]           # [B]

        # 分类权重 α_t
        if self.alpha is None:
            alpha_t = 1.0
        elif self.alpha.dim() == 0:
            alpha_t = float(self.alpha.item())
        else:
            # 按类取权重
            alpha_t = self.alpha.to(logits.device)[target]  # [B]

        # Focal 调制因子 (1 - p_t)^γ
        mod = (1.0 - pt).clamp(min=0.0).pow(self.gamma)     # [B]

        # 损失
        loss = -alpha_t * mod * log_pt                      # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
