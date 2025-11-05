#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归头（统一版）

- PerClassRegressors：每个类别一个回归器，输出形状 [B, C]
- GlobalRegressor   ：全局回归器，输出形状 [B]

设计要点：
- 默认 `independent`：为每个类别单独建一个轻量 MLP（更灵活，便于差异化拟合）；
- 可选 `shared_mlp` ：先通过共享 MLP 得到特征，再接一个 Linear 输出 C 维（参数更省）。
- 可选输出夹紧（clamp）：用 tanh 映射到 [y_min, y_max]，用于已知物理范围的回归场景；
- 与 `train.py / eval.py / infer.py / export_onnx.py` 完全兼容。
"""
from __future__ import annotations
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


# ============================= 工具与初始化 ============================= #

def _get_act(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


def _init_weights(m: nn.Module):
    """权重初始化：
    - Linear: kaiming_uniform（GELU/RELU 友好），bias=0
    - LayerNorm: weight=1, bias=0
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ============================= 每类回归器 ============================= #
class PerClassRegressors(nn.Module):
    """每类回归器。

    参数
    - d_model : 编码器输出维度 D
    - n_classes: 类别数 C
    - hidden  : 隐藏层维度（默认=D）
    - dropout : 随机失活比例
    - head_type: 'independent' | 'shared_mlp'
    - activation: 'gelu' | 'relu'
    - clamp  : Optional[(y_min, y_max)] 若提供，则将输出压到该区间
    """
    def __init__(self,
                 d_model: int,
                 n_classes: int,
                 hidden: Optional[int] = None,
                 dropout: float = 0.1,
                 head_type: str = 'independent',
                 activation: str = 'gelu',
                 clamp: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.h = int(hidden or d_model)
        self.act = _get_act(activation)
        self.clamp = clamp
        self.head_type = head_type.lower()

        if self.head_type not in {"independent", "shared_mlp"}:
            raise ValueError("head_type 必须是 'independent' 或 'shared_mlp'")

        if self.head_type == 'independent':
            # 为每个类别建立独立 MLP
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, self.h),
                    self.act,
                    nn.Dropout(dropout),
                    nn.Linear(self.h, 1),
                ) for _ in range(self.n_classes)
            ])
            self.apply(_init_weights)
        else:
            # 共享 MLP + 类别线性输出
            self.backbone = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.h),
                self.act,
                nn.Dropout(dropout),
            )
            self.fc_out = nn.Linear(self.h, self.n_classes)
            self.apply(_init_weights)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        if self.clamp is None:
            return y
        y_min, y_max = float(self.clamp[0]), float(self.clamp[1])
        # tanh 映射到 [-1,1] 再缩放到 [y_min,y_max]
        y = torch.tanh(y)
        return (y * 0.5 + 0.5) * (y_max - y_min) + y_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向：
        - 输入：z:[B,D]
        - 输出：y_pc:[B,C]（按类别的回归标量）
        """
        if z.dim() != 2:
            raise ValueError(f"PerClassRegressors 期望输入形状 [B,D]，收到 {tuple(z.shape)}")

        if self.head_type == 'independent':
            ys: List[torch.Tensor] = []
            for head in self.heads:
                ys.append(head(z))  # [B,1]
            y = torch.cat(ys, dim=1)  # [B,C]
        else:
            h = self.backbone(z)  # [B,h]
            y = self.fc_out(h)    # [B,C]

        y = self._clamp(y)
        return y


# ============================= 全局回归器 ============================= #
class GlobalRegressor(nn.Module):
    """全局回归器（不区分类别）。

    参数
    - d_model : 编码器输出维度 D
    - hidden  : 隐藏层维度（默认=D）
    - dropout : 随机失活比例
    - activation: 'gelu' | 'relu'
    - clamp  : Optional[(y_min, y_max)] 若提供，则将输出压到该区间
    """
    def __init__(self,
                 d_model: int,
                 hidden: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 clamp: Optional[Tuple[float, float]] = None):
        super().__init__()
        h = int(hidden or d_model)
        self.clamp = clamp
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h),
            _get_act(activation),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )
        self.apply(_init_weights)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        if self.clamp is None:
            return y
        y_min, y_max = float(self.clamp[0]), float(self.clamp[1])
        y = torch.tanh(y)
        return (y * 0.5 + 0.5) * (y_max - y_min) + y_min

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向：z:[B,D] → y:[B]"""
        if z.dim() != 2:
            raise ValueError(f"GlobalRegressor 期望输入形状 [B,D]，收到 {tuple(z.shape)}")
        y = self.net(z).squeeze(-1)  # [B]
        y = self._clamp(y)
        return y
