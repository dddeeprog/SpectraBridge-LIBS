#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPY/NPZ 数据集与辅助迭代器

- NpyDataset: 统一从 X.npy / y.npy / c.npy 读取，支持按需归一化（none/standard/minmax），
              自动 dtype 转换、长度校验与（必要时）线性插值到 L。
- NpyDualIter: 将一个 DataLoader 包装为“无限迭代器”，用于 UDA/SSDA 训练中同步抽取目标域批次。

约定：
- X.npy: [N, L] float32  光谱强度序列
- y.npy: [N]    float32  回归标签（可选）
- c.npy: [N]    int64    类别标签（可选）

注意：
- 训练前推荐用 csv_to_npy.py 将现场 CSV 统一转换，以确保 L 一致无 NaN/Inf。
- 若仍存在 L 不一致，Dataset 会做线性插值到 config.spectral_length，但建议源头修正。
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================= 工具函数 ============================= #

def _ensure_1d(a: np.ndarray) -> np.ndarray:
    """确保返回一维数组（处理 [L] 或 [1,L] 或 [L,1]）。"""
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if 1 in a.shape:
            return a.reshape(-1)
    raise ValueError(f"期望一维数组，收到形状 {a.shape}")


def _interp_to_length(x: np.ndarray, L: int) -> np.ndarray:
    """将一维序列线性插值/裁剪到长度 L（无外推）。
    - 若 len(x)==L: 直接返回；
    - 若 len(x)> L: 均匀采样裁剪；
    - 若 len(x)< L: 线性插值到 L；
    """
    n = x.shape[0]
    if n == L:
        return x
    if n <= 1:
        # 极端兜底：常量填充
        return np.full((L,), float(x[0] if n == 1 else 0.0), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=n)
    fp = x.astype(np.float32)
    xq = np.linspace(0.0, 1.0, num=L)
    yq = np.interp(xq, xp, fp).astype(np.float32)
    return yq


def _normalize(x: np.ndarray, how: str) -> np.ndarray:
    """样本级归一化：none/standard/minmax。"""
    how = (how or 'none').lower()
    if how == 'none':
        return x.astype(np.float32)
    x = x.astype(np.float32)
    if how == 'standard':
        m = float(x.mean())
        s = float(x.std())
        if not np.isfinite(s) or s < 1e-8:
            return (x - m).astype(np.float32)
        return ((x - m) / s).astype(np.float32)
    if how == 'minmax':
        lo = float(x.min())
        hi = float(x.max())
        if not np.isfinite(hi - lo) or hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)
    raise ValueError(f"未知归一化方式：{how}")


# ============================= 主数据集 ============================= #
class NpyDataset(Dataset):
    """从 NPY/NPZ 文件构建的数据集。

    参数：
    - x_path: X.npy/npz 路径
    - y_path: y.npy/npz 路径（可选）
    - c_path: c.npy/npz 路径（可选）
    - spectral_length: 期望长度 L
    - normalize: 'none' | 'standard' | 'minmax'
    - mmap: 是否使用内存映射（对超大数据友好）
    """
    def __init__(self,
                 x_path: Path,
                 y_path: Optional[Path] = None,
                 c_path: Optional[Path] = None,
                 spectral_length: int = 1024,
                 normalize: str = 'standard',
                 mmap: bool = False):
        super().__init__()
        self.x_path = Path(x_path)
        self.y_path = Path(y_path) if y_path is not None else None
        self.c_path = Path(c_path) if c_path is not None else None
        self.L = int(spectral_length)
        self.normalize = normalize
        self.mmap = mmap

        # 载入数组（支持 .npz/.npy）
        self.X = self._load_array(self.x_path)
        self.y = self._load_array(self.y_path) if self.y_path and self.y_path.exists() else None
        self.c = self._load_array(self.c_path) if self.c_path and self.c_path.exists() else None

        # 基础校验
        if self.X.ndim != 2:
            raise ValueError(f"X 期望形状 [N,L]，收到 {self.X.shape}")
        self.N, self.Lx = self.X.shape
        if self.y is not None and self.y.shape[0] != self.N:
            raise ValueError("y 的样本数与 X 不一致")
        if self.c is not None and self.c.shape[0] != self.N:
            raise ValueError("c 的样本数与 X 不一致")

        # 记录类别统计（若可用），便于 focal alpha 设置
        self.class_counts = None
        if self.c is not None:
            # 强制 int64
            self.c = self.c.astype(np.int64)
            # 统计各类数量
            max_c = int(self.c.max()) if self.c.size > 0 else -1
            counts = np.zeros((max(0, max_c + 1),), dtype=np.int64)
            for k in range(counts.shape[0]):
                counts[k] = int((self.c == k).sum())
            self.class_counts = counts

    # --------------------------- 私有工具 --------------------------- #
    def _load_array(self, path: Optional[Path]):
        if path is None:
            return None
        if not path.exists():
            return None
        if str(path).endswith('.npz'):
            # 兼容性：期望 npz 内键为 'arr_0' 或 'X/y/c'
            z = np.load(path, allow_pickle=False)
            for k in ('X', 'y', 'c', 'arr_0'):
                if k in z:
                    return z[k]
            raise ValueError(f"无法从 {path} 中识别键，期望 'X'/'y'/'c' 或 'arr_0'")
        else:
            return np.load(path, mmap_mode=('r' if self.mmap else None))

    # --------------------------- Dataset 接口 --------------------------- #
    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 读取并确保为 float32
        x = self.X[idx]
        x = _ensure_1d(x)
        if x.shape[0] != self.L:
            x = _interp_to_length(x, self.L)
        x = _normalize(x, self.normalize)
        x_t = torch.from_numpy(x.astype(np.float32))

        item: Dict[str, torch.Tensor] = {"x": x_t}

        if self.y is not None:
            y = float(self.y[idx])
            item["y"] = torch.tensor(y, dtype=torch.float32)
        if self.c is not None:
            c = int(self.c[idx])
            item["c"] = torch.tensor(c, dtype=torch.long)
        return item

    # --------------------------- 实用方法 --------------------------- #
    def get_class_weights(self, mode: str = 'inverse_freq') -> Optional[torch.Tensor]:
        """基于数据集中类别分布返回权重向量（用于 Focal/CE 的 alpha）。
        - 'inverse_freq'：1/(freq+eps) 再归一化到和为 C
        """
        if self.class_counts is None:
            return None
        cnt = self.class_counts.astype(np.float32)
        eps = 1e-6
        w = 1.0 / np.maximum(cnt, eps)
        w = w * (len(w) / w.sum())
        return torch.tensor(w, dtype=torch.float32)


# ============================= 无限迭代器 ============================= #
class NpyDualIter:
    """包装一个 DataLoader，提供 next() 无限取批次。
    常用于 UDA/SSDA：每个源域 step 同步取一个目标域批次。
    """
    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

    # 兼容手动调用
    def next(self):
        return self.__next__()
