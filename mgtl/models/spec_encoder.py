#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpecEncoder：单模态光谱编码器（Conv1d + Transformer）

输入：x ∈ R[B, L]（单通道光谱，已在 Dataset 中完成标准化/长度对齐）
输出：z ∈ R[B, D]（全局表征），D=d_model

结构要点：
- 多层 1D 卷积堆叠（可设 dilation/stride），提取局部峰形与线比等短程特征；
- 可选池化（avg/max/none）；
- 1×1 卷积将通道投影到 d_model；
- 位置编码（sinusoidal | learned）：为 Transformer 提供序位信息；
- TransformerEncoder（batch_first=True）：聚合全局上下文（跨峰/跨段相关性）；
- 池化输出（mean | cls）：按配置汇聚为定长向量。

备注：
- 为避免不必要复杂性，序列长度 L 设计为固定（由 config.spectral_length 控制）。
- 若 conv_stride>1 或池化启用，Transformer 输入序列长度将缩短（自动处理）。
"""
from __future__ import annotations
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================= 工具函数 ============================= #

def _get_act(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


def _same_pad(kernel: int, dilation: int = 1, stride: int = 1) -> int:
    """近似 SAME padding（对 stride=1 完全保持长度）。
    stride>1 时按中心对齐给出对称 padding，长度将按 stride 下采样。
    """
    eff = dilation * (kernel - 1) + 1
    return max((eff - stride) // 2, 0)


# ============================= 位置编码 ============================= #
class SinusoidalPE(nn.Module):
    """经典正弦位置编码，按需截取到给定长度。"""
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，避免被优化器更新
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D] 加位置信息"""
        B, T, D = x.shape
        return x + self.pe[:T, :D].unsqueeze(0)


class LearnedPE(nn.Module):
    """可学习位置编码（Embedding），上限 max_len。"""
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)
        pe = self.emb(pos)  # [T, D]
        return x + pe.unsqueeze(0)


# ============================= 主编码器 ============================= #
class SpecEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 conv_channels: List[int] = [64, 128],
                 conv_kernels: List[int] = [5, 3],
                 conv_dilation: List[int] = [1, 2],
                 conv_stride: List[int] = [1, 1],
                 conv_pool: str = 'none',           # 'none' | 'avg' | 'max'
                 transformer_layers: int = 2,
                 nhead: int = 4,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'gelu',          # 'gelu' | 'relu'
                 pooling: str = 'mean',             # 'mean' | 'cls'
                 positional_encoding: str = 'learned',  # 'learned' | 'sinusoidal'
                 pe_max_len: int = 16384):
        super().__init__()
        assert len(conv_channels) == len(conv_kernels) == len(conv_dilation) == len(conv_stride), \
            "conv_* 列表长度必须一致"

        self.pooling = pooling.lower()
        self.act = _get_act(activation)

        # —— 卷积堆叠 ——
        in_ch = 1
        blocks = []
        for i, (ch, k, d, s) in enumerate(zip(conv_channels, conv_kernels, conv_dilation, conv_stride)):
            pad = _same_pad(k, dilation=d, stride=s)
            blocks += [
                nn.Conv1d(in_ch, ch, kernel_size=k, stride=s, padding=pad, dilation=d),
                nn.BatchNorm1d(ch),
                self.act,
            ]
            if conv_pool == 'avg':
                blocks.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif conv_pool == 'max':
                blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = ch
        self.conv = nn.Sequential(*blocks)

        # 将通道映射到 d_model（序列维不变）
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=1)

        # —— 位置编码 ——
        pe_type = (positional_encoding or 'learned').lower()
        if pe_type == 'sinusoidal':
            self.pe = SinusoidalPE(d_model, max_len=pe_max_len)
        else:
            self.pe = LearnedPE(d_model, max_len=pe_max_len)

        # —— Transformer 编码器 ——
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu' if activation.lower() == 'gelu' else 'relu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # —— CLS token（仅当 pooling=cls 时使用）——
        if self.pooling == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # —— 最终归一化 ——
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：x:[B, L] -> z:[B, D]
        注意：输入需为 float32，并已标准化；长度 L 需与 config 保持一致。
        """
        if x.dim() != 2:
            raise ValueError(f"SpecEncoder 期望输入形状 [B,L]，收到 {tuple(x.shape)}")

        # [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)
        # 卷积堆叠: [B, C, T]
        h = self.conv(x)
        # 通道投影到 d_model: [B, D, T]
        h = self.proj(h)
        # 变换为 Transformer 接口: [B, T, D]
        h = h.transpose(1, 2)

        # 位置编码
        h = self.pe(h)

        # 可选 CLS token
        if self.cls_token is not None:
            cls = self.cls_token.expand(h.size(0), -1, -1)  # [B,1,D]
            h = torch.cat([cls, h], dim=1)  # [B, 1+T, D]

        # Transformer 聚合
        h = self.transformer(h)  # [B, T' , D]

        # 池化到全局向量
        if self.pooling == 'cls':
            # 取 CLS 位置
            z = h[:, 0, :]
        else:
            # 序列平均
            z = h.mean(dim=1)

        # 归一化
        z = self.out_norm(z)
        return z
