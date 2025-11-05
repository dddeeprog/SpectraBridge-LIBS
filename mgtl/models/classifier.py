#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ClassifierHead：分类头（完善版）

输入：z ∈ R[B, D]（编码器输出的全局表征）
输出：logits ∈ R[B, C]（未归一化分类分数）

设计要点：
- 轻量两层 MLP（LayerNorm → Linear → GELU → Dropout → Linear），稳健易训；
- 可配置隐藏层维度与 dropout；
- **可选温度缩放（temperature scaling）**：部署阶段用于置信度校准；
- 支持 `return_feat=True` 返回中间特征，便于可解释/蒸馏等二次开发；
- 保持与训练/推理脚本接口兼容（默认仅返回 logits）。
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_classes: int,
                 hidden: Optional[int] = None,
                 dropout: float = 0.1,
                 temperature: float = 1.0):
        """
        参数
        - d_model: 编码器输出维度 D
        - n_classes: 类别数 C
        - hidden: 隐藏层维度（默认=D）
        - dropout: 随机失活比例（0~1）
        - temperature: 温度缩放初始值（>0，部署时用于置信度校准）
        """
        super().__init__()
        h = int(hidden or d_model)

        # 明确拆分，便于拿到中间特征
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, h)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h, n_classes)

        # 温度缩放参数（默认不训练，只在部署后通过校准过程设定）
        self.register_buffer("_temperature", torch.tensor(float(temperature)), persistent=False)
        self._temperature_trainable = False

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """权重初始化：
        - Linear: kaiming_uniform（GELU 友好），bias=0
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # --------------------------- 公共方法 --------------------------- #
    def set_temperature(self, value: float, trainable: bool = False) -> None:
        """设置温度缩放值；若 trainable=True，将其作为可训练参数暴露。"""
        value = float(max(value, 1e-6))
        if hasattr(self, "temperature"):
            # 若已创建可训练参数，则直接赋值
            with torch.no_grad():
                self.temperature.fill_(value)
            self.temperature.requires_grad_(trainable)
        else:
            if trainable:
                self.temperature = nn.Parameter(torch.tensor(value, dtype=torch.float32))
                self._temperature_trainable = True
            else:
                # 使用 buffer 保存，保持与 state_dict 兼容
                self.register_buffer("temperature", torch.tensor(value, dtype=torch.float32), persistent=True)
                self._temperature_trainable = False

    def get_temperature(self) -> float:
        """返回当前温度缩放值（float）。"""
        if hasattr(self, "temperature"):
            return float(self.temperature.detach().cpu())
        return float(self._temperature.detach().cpu())

    # --------------------------- 前向计算 --------------------------- #
    def forward(self, z: torch.Tensor, return_feat: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """前向：
        - 输入：z:[B,D]
        - 输出：logits:[B,C]（默认）；若 return_feat=True，同时返回中间特征 feat:[B,h]
        说明：温度缩放仅在**导出/部署**时用于置信度校准，不改变训练损失定义。
        """
        if z.dim() != 2:
            raise ValueError(f"ClassifierHead 期望输入形状 [B,D]，收到 {tuple(z.shape)}")

        h = self.ln(z)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)

        # 应用温度缩放到 logits（p = softmax(logits / T)；此处只返回 logits_T）
        T = self.get_temperature()
        if abs(T - 1.0) > 1e-6:
            logits = logits / T

        if return_feat:
            return logits, h
        return logits
