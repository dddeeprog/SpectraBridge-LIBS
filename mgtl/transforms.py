#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transforms.py：光谱一维增强/预处理变换（纯 PyTorch/NumPy 实现，无第三方依赖）

用途：
- 训练增广：强度微扰、加噪、随机遮挡、波长轻微平移等（稳健性提升、减少过拟合）。
- 预处理：移动平均平滑、裁剪、标准化/归一化（若需覆盖 Dataset 内置策略）。

范式：
- 单样本变换：输入/输出均为 shape [L] 的 torch.Tensor（float32）。
- Compose 支持将多个变换串联。

注意：
- 与 `mgtl/data.py` 的标准化不冲突；若你在 transforms 中重复标准化，请在 config 中将 Dataset 的 normalize 设为 'none'。
- 平滑采用“简单移动平均”(SMA) 实现，避免额外依赖；若需 Savitzky-Golay，可后续引入 SciPy。
"""
from __future__ import annotations
from typing import Iterable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F


# ============================= 组合器 ============================= #
class Compose:
    """顺序执行一组单样本变换。"""
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# ============================= 预处理 ============================= #
class ToTensor:
    """将 numpy.ndarray 或 list 转换为 torch.float32 Tensor（一维）。"""
    def __call__(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(torch.float32)
        return torch.tensor(np.asarray(x).reshape(-1), dtype=torch.float32)


class Standardize:
    """样本级标准化（零均值/单位方差）。"""
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean()
        s = x.std()
        if s < self.eps:
            return x - m
        return (x - m) / s


class MinMaxNormalize:
    """样本级 Min-Max 归一化到 [0,1]。"""
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lo = x.min()
        hi = x.max()
        if (hi - lo) < self.eps:
            return torch.zeros_like(x)
        return (x - lo) / (hi - lo)


class ClipRange:
    """裁剪到给定范围。"""
    def __init__(self, lo: float = -float('inf'), hi: float = float('inf')):
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.lo, self.hi)


class MovingAverage:
    """简单移动平均平滑（窗口 k 奇数）。边界使用“复制延拓”策略。"""
    def __init__(self, k: int = 5):
        assert k >= 1 and (k % 2 == 1), "k 必须为奇数且 >=1"
        self.k = int(k)
        # 卷积核
        w = torch.ones(1, 1, self.k, dtype=torch.float32) / float(self.k)
        self.registered_w = w  # 存 buffer 的替代做法（类无 nn.Module）

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 1:
            x = x.reshape(-1)
        # 复制边界
        pad = self.k // 2
        x2 = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate')
        y = torch.nn.functional.conv1d(x2, self.registered_w).squeeze(0).squeeze(0)
        return y


# ============================= 训练增强 ============================= #
class RandomScaleShift:
    """随机仿射强度扰动：x' = a * x + b。
    - a ∈ [1-s, 1+s], b ∈ [-t, t]
    """
    def __init__(self, scale: float = 0.05, shift: float = 0.02):
        self.scale = float(scale)
        self.shift = float(shift)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        a = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale
        b = (torch.rand(1).item() * 2 - 1) * self.shift
        return x * a + b


class RandomGaussianNoise:
    """加性高斯噪声，σ= noise_std * (x 的标准差)。"""
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = float(noise_std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = float(x.std().item())
        noise = torch.randn_like(x) * (self.noise_std * (std if std > 1e-8 else 1.0))
        return x + noise


class RandomCutout1D:
    """随机遮挡一段连续波长区间（模拟局部缺失/异常）。
    - frac: 遮挡长度相对比例（0~1）或绝对长度（当 use_frac=False）
    - value: 填充值（默认 0）
    """
    def __init__(self, frac: float = 0.05, use_frac: bool = True, value: float = 0.0):
        self.frac = float(frac)
        self.use_frac = bool(use_frac)
        self.value = float(value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[0]
        k = int(self.frac * L) if self.use_frac else int(self.frac)
        k = max(1, min(L, k))
        s = np.random.randint(0, L - k + 1)
        y = x.clone()
        y[s:s+k] = self.value
        return y


class RandomWavelengthShift:
    """随机小幅循环移位（模拟波长轴微偏差/温漂校准误差）。"""
    def __init__(self, max_shift: int = 2):
        self.max_shift = int(max_shift)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x
        k = np.random.randint(-self.max_shift, self.max_shift + 1)
        if k == 0:
            return x
        return torch.roll(x, shifts=int(k), dims=0)


# ============================= 工厂函数 ============================= #

def build_default_train_augs() -> Compose:
    """给出一套稳健的增广组合（保守默认），可在 config 中开关。"""
    return Compose([
        RandomScaleShift(scale=0.05, shift=0.02),
        RandomGaussianNoise(noise_std=0.01),
        RandomCutout1D(frac=0.02, use_frac=True, value=0.0),
        RandomWavelengthShift(max_shift=1),
    ])


# ## 6. 一键运行示例（快速上手）

# > 假设你已完成 CSV→NPY 转换，并在 `config.yaml` 中正确设置了 `paths.*` 与 `n_classes`、`spectral_length`。

# # 0) 可选：将现场 CSV 转换为标准 NPY（若已是 NPY 可跳过）
# python csv_to_npy.py \
#   --csv_dir data_raw/source --out_dir data/source --spectral_length 2048 --has_y --has_c
# python csv_to_npy.py \
#   --csv_dir data_raw/target --out_dir data/target --spectral_length 2048 --has_y --has_c=false

# # 1) 预训练（仅源域监督）
# python train.py --config S1/config.yaml --save artifacts/best_pretrain.pt

# # 2) UDA 域自适应（默认 Deep CORAL），从预训练权重继续
# python train.py --config S1/config.yaml --resume artifacts/best_pretrain.pt --save artifacts/best_uda.pt

# # 3) （可选）SSDA 少量目标标注微调
# python train.py --config S1/config.yaml --resume artifacts/best_uda.pt --save artifacts/best_ssda.pt

# # 4) 评估（source & target）
# python eval.py --config S1/config.yaml --ckpt artifacts/best_uda.pt --domains source target

# # 5) 推理（目标域），软加权回退 + 指数平滑
# python infer.py --config S1/config.yaml --ckpt artifacts/best_uda.pt \
#   --input_dir data/target --out artifacts/infer_target.csv --alpha 0.2 --tau 0.6

# # 6) 导出 ONNX（含 logits / y_per_class / y_global / y_soft）
# python export_onnx.py --config S1/config.yaml --ckpt artifacts/best_uda.pt \
#   --out artifacts/model.onnx --opset 17 --dynamic
# ```
