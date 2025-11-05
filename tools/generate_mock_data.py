#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成用于单模态光谱迁移学习（源域=煤饼、目标域=煤粉）的**模拟数据**。
支持可控的**域移位**（峰位微移、展宽、漂移、噪声、基线/散射等），
并同时给出**分类标签**（煤类/配煤簇）与**回归目标**（元素含量等）。

输出目录结构（NPY）：
  data/
    source/{X.npy:[N,L], y.npy:[N], c.npy:[N]}
    target/{X.npy:[N,L], (y.npy:[N]), (c.npy:[N])}

使用示例：
  python generate_mock_data.py --root data --n 2000 --length 1024 --classes 4
  python generate_mock_data.py --root data --n 1500 --length 4096 --classes 5 --no_target_labels

注意：
- X 为 float32，且不含 NaN/Inf；
- L（光谱长度）需与 config.yaml:spectral_length 保持一致；
- 该脚本仅用于**流程联调/算法验证**，不代表真实物理光谱。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# ============================= 工具函数 ============================= #

def set_seed(seed: int = 42) -> np.random.Generator:
    """固定随机种子，返回 numpy 随机数生成器。"""
    rng = np.random.default_rng(seed)
    return rng


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """高斯函数，用于模拟谱线峰形。"""
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def make_class_line_table(
    C: int,
    L: int,
    grid_min: float = 160.0,
    grid_max: float = 890.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """为每个类别生成一张“谱线中心表”（索引空间）。

    思路：
    - 先在波长轴上放置若干公共锚点（anchors）；
    - 将波长映射到索引空间 [0, L-1]；
    - 不同类别在每个锚点处加入微小位移（类内规律差异）。
    返回形状：[C, K]，K 为每类谱线数量。
    """
    rng = rng or np.random.default_rng(42)
    K = 10  # 每个类别的谱线条数（可按需调整）
    anchors = np.linspace(grid_min + 20, grid_max - 20, K)          # 公共谱线“参考位置”
    anchor_idx = np.interp(anchors, [grid_min, grid_max], [0, L - 1])  # 映射到索引
    class_lines = []
    for _ in range(C):
        # 类间微小差异：对每条线施加 ~N(0, 2.5) 的位移抖动
        shift = rng.normal(0.0, 2.5, size=K)
        jitter = anchor_idx + shift
        class_lines.append(jitter)
    return np.array(class_lines)  # [C, K]


def synth_one_spectrum(
    L: int,
    class_id: int,
    lines_idx: np.ndarray,
    base_conc: float,
    domain: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """合成**一条**光谱及其回归目标。

    参数：
    - L: 光谱长度；
    - class_id: 类别 ID（0..C-1）；
    - lines_idx: 该类的谱线中心索引数组，形如 [K]；
    - base_conc: 基础含量（0..1），将决定峰强与非线性饱和程度；
    - domain: 'source' 或 'target'，决定噪声/展宽/漂移等域特性；
    - rng: 随机数生成器。

    返回：
    - x: 模拟光谱（float32, 形状 [L]）；
    - y: 回归目标标量（例如元素含量，经轻微偏置和噪声扰动）。
    """
    x = np.zeros(L, dtype=np.float32)

    # —— 连续谱/基线（简化的一阶趋势）——
    slope = rng.normal(0.0005, 0.0003)
    intercept = rng.uniform(0.0, 0.02)
    baseline = intercept + slope * np.arange(L)

    # —— 叠加若干高斯峰（峰宽/幅值受类与浓度影响）——
    K = lines_idx.shape[0]
    idx_grid = np.arange(L)
    for k in range(K):
        mu = lines_idx[k]
        # 目标域（煤粉）更宽的峰，用于模拟成像模糊/粉尘与分辨率差异
        width = rng.uniform(1.2, 2.0) if domain == 'source' else rng.uniform(1.8, 3.0)
        amp = (0.6 + 0.4 * base_conc) * rng.uniform(0.8, 1.2)
        x += amp * gaussian(idx_grid, mu, width)

    # —— 类别相关的线比调制：让部分谱线强度随浓度/类别发生规律性变化 ——
    mod_idx = (class_id * 3 + np.arange(3)) % K
    x[mod_idx.clip(0, L - 1)] *= (1.1 + 0.2 * np.tanh(2 * (base_conc - 0.5)))

    # —— 自吸/饱和的简化非线性（高浓度时峰顶被压扁）——
    nonlin = 1.0 if domain == 'source' else 1.1
    x = x / (1.0 + nonlin * 0.6 * np.maximum(x - 0.7, 0.0))

    # —— 目标域特有扰动：波长微移 + 轻微拉伸 + 粉尘散射 + 低频漂移 ——
    if domain == 'target':
        # 波长标定微偏移与轻度拉伸
        shift = rng.normal(0.2, 0.15)
        stretch = 1.0 + rng.normal(0.0008, 0.0005)
        src_idx = np.arange(L)
        dst_idx = (src_idx - shift) / stretch
        dst_idx = np.clip(dst_idx, 0, L - 1)
        x = np.interp(src_idx, dst_idx, x).astype(np.float32)
        # 粉尘散射（高斯白噪）与低频漂移（简化为正弦）
        dust = rng.normal(0.0, 0.008, size=L).astype(np.float32)
        drift_amp = rng.uniform(0.0, 0.02)
        drift = drift_amp * np.sin(np.linspace(0, 3.14 * rng.uniform(1.0, 2.0), L))
        x = x + dust + drift.astype(np.float32)

    # —— 叠加基线与测量噪声；裁到非负 ——
    x = x + baseline.astype(np.float32)
    noise = rng.normal(0.0, 0.01 if domain == 'source' else 0.015, size=L).astype(np.float32)
    x = np.maximum(x + noise, 0.0)

    # —— 构造回归目标 y：浓度的线性函数 + 域偏置 + 噪声 ——
    a = 1.0
    b = 0.0 if domain == 'source' else 0.05  # 目标域存在系统偏置
    y = a * base_conc + b + rng.normal(0.0, 0.03)
    return x.astype(np.float32), float(y)


def build_dataset(N: int, L: int, C: int, domain: str, line_table: np.ndarray, rng: np.random.Generator):
    """构建一个域的数据集（X,y,c）。"""
    X = np.zeros((N, L), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)
    c = np.zeros(N, dtype=np.int64)
    for i in range(N):
        class_id = int(rng.integers(0, C))              # 随机采样类别
        conc = float(np.clip(rng.beta(2.0, 2.0), 0, 1))  # 0..1 的“含量”
        lines_idx = line_table[class_id]
        xi, yi = synth_one_spectrum(L, class_id, lines_idx, conc, domain, rng)
        X[i] = xi
        y[i] = yi
        c[i] = class_id
    return X.astype(np.float32), y.astype(np.float32), c.astype(np.int64)


# ============================= 主流程 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="输出数据集的根目录")
    ap.add_argument("--n", type=int, default=2000, help="每个域的样本数")
    ap.add_argument("--length", type=int, default=1024, help="光谱长度 L")
    ap.add_argument("--classes", type=int, default=4, help="类别数 C")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--no_target_labels", action="store_true", help="不保存目标域的 y/c 标签（仅用于无监督对齐与线上推理）")
    args = ap.parse_args()

    root = Path(args.root)
    (root / "source").mkdir(parents=True, exist_ok=True)
    (root / "target").mkdir(parents=True, exist_ok=True)

    rng = set_seed(args.seed)

    # 构建“谱线中心表”，源/目标域共享以体现相同物理，再由各自扰动形成域差异
    line_table = make_class_line_table(args.classes, args.length, rng=rng)

    # —— 源域（煤饼）——
    Xs, ys, cs = build_dataset(args.n, args.length, args.classes, domain="source", line_table=line_table, rng=rng)
    np.save(root / "source" / "X.npy", Xs)
    np.save(root / "source" / "y.npy", ys)
    np.save(root / "source" / "c.npy", cs)

    # —— 目标域（煤粉，含域移位）——
    Xt, yt, ct = build_dataset(args.n, args.length, args.classes, domain="target", line_table=line_table, rng=rng)
    np.save(root / "target" / "X.npy", Xt)
    if not args.no_target_labels:
        np.save(root / "target" / "y.npy", yt)
        np.save(root / "target" / "c.npy", ct)

    # —— 简单健康检查与摘要 ——
    def stats(x: np.ndarray):
        return dict(shape=tuple(x.shape), min=float(x.min()), max=float(x.max()), mean=float(x.mean()))

    print("数据已保存到:", root.resolve())
    print("source/X:", stats(Xs))
    print("source/y:", dict(shape=ys.shape, mean=float(ys.mean())))
    print("target/X:", stats(Xt))
    if not args.no_target_labels:
        print("target/y:", dict(shape=yt.shape, mean=float(yt.mean())))


if __name__ == "__main__":
    main()
