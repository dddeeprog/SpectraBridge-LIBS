#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将实际采样的 CSV 光谱数据转换为本项目所需的 NPY 格式：
  data/<domain>/{X.npy:[N,L], y.npy:[N](可选), c.npy:[N](可选)}

支持两种输入形态：
1) table 模式：一个 CSV 内包含多条样本（通常一行一条样本），列中可包含 y/c 标签；
2) files 模式：一个目录下每个 CSV 是一条样本（通常是 [波长, 强度] 两列或仅强度一列）。

常用示例：
  # 表格：每行一条样本；强度列是 i_ 开头；y,c 两列为标签；目标域
  python csv_to_npy.py --mode table --input raw/target.csv --domain target \
      --spectral_length 1024 --wavemin 160 --wavemax 890 \
      --intensity_prefix i_ --y_col y --c_col c --out_root data

  # 文件夹：每个 CSV 为两列 [wavelength,intensity]；标签从 label_map.csv 读取；源域
  python csv_to_npy.py --mode files --input raw/source_dir --domain source \
      --spectral_length 1024 --label_map raw/label_map.csv --out_root data

注意：
- 输出的 X.npy 统一为 float32，y.npy 为 float32，c.npy 为 int64；
- 若输入不含波长列，将按等间隔栅格处理（wavemin/wavemax 用于元数据记录与后续一致性）；
- 若未提供 y/c，将只保存 X.npy（训练/评估时相应能力受限）。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================= 基础工具 ============================= #

def set_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resample_to_grid(wave: np.ndarray, inten: np.ndarray, L: int,
                     wavemin: Optional[float] = None,
                     wavemax: Optional[float] = None) -> np.ndarray:
    """将任意波长采样的光谱重采样到统一长度 L 的等间隔栅格。
    - wave: [M]（递增）; inten: [M]
    - 返回: [L]
    """
    wave = np.asarray(wave, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    if wavemin is None:
        wavemin = float(np.nanmin(wave))
    if wavemax is None:
        wavemax = float(np.nanmax(wave))
    # 防止边界重合
    if wavemax <= wavemin:
        wavemax = wavemin + 1e-6
    grid = np.linspace(wavemin, wavemax, L)
    # 缺测内插，边界外推为最近值
    inten = np.asarray(inten)
    mask = ~np.isfinite(inten)
    if mask.any():
        # 简单线性插值填充内部 NaN
        notnan_idx = np.where(~mask)[0]
        if notnan_idx.size >= 2:
            inten = pd.Series(inten).interpolate(limit_direction="both").to_numpy()
        else:
            inten[mask] = 0.0
    out = np.interp(grid, wave, inten, left=inten[0], right=inten[-1])
    return out.astype(np.float32)


def standardize(x: np.ndarray, method: str = "none") -> np.ndarray:
    """按样本维（行）做标准化/归一化。x: [N,L]"""
    if method == "none":
        return x
    x = x.astype(np.float32)
    if method == "standard":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mu) / sd
    if method == "minmax":
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        denom = (mx - mn)
        denom[denom < 1e-8] = 1.0
        return (x - mn) / denom
    return x

# ============================= table 模式 ============================= #

def load_table_mode(file: Path,
                    spectral_length: int,
                    wavemin: Optional[float],
                    wavemax: Optional[float],
                    intensity_prefix: Optional[str],
                    intensity_cols: Optional[List[str]],
                    wavelength_col: Optional[str],
                    y_col: Optional[str],
                    c_col: Optional[str],
                    normalize: str = "none") -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """从一个多样本 CSV 生成 X/y/c。默认每行一条样本。
    - 若提供 wavelength_col，则每行的强度列需对应相同的波长向量（常见是所有样本共用同一波长列，不同列代表不同样本的强度，这种情况建议先转置为行样本形式再使用）。
    - 更常见的工程格式：每行一条样本，强度分布在若干列（i_ 开头），y/c 分别为标签列。
    """
    df = pd.read_csv(file)
    # 选择强度列
    if intensity_cols:
        I_cols = intensity_cols
    elif intensity_prefix:
        I_cols = [c for c in df.columns if str(c).startswith(intensity_prefix)]
    else:
        # 回退：自动选择所有数值列，排除明显是标签的列
        exclude = set([y_col, c_col])
        I_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not I_cols:
        raise ValueError("未找到强度列，请通过 --intensity_prefix 或 --intensity_cols 指定。")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    c_list: List[int] = []

    # 如果提供波长列，则对每一行做重采样；否则直接取数值并在末端对齐/裁剪到 L
    if wavelength_col and wavelength_col in df.columns:
        wave = df[wavelength_col].to_numpy()
        if wave.ndim != 1:
            raise ValueError("wavelength_col 应为一维列。")
        for idx, row in df.iterrows():
            inten = row[I_cols].to_numpy(dtype=np.float64)
            x = resample_to_grid(wave, inten, spectral_length, wavemin, wavemax)
            X_list.append(x)
            if y_col and y_col in df.columns:
                y_list.append(float(row[y_col]))
            if c_col and c_col in df.columns:
                c_list.append(int(row[c_col]))
    else:
        # 直接从强度列取值，必要时截断/填零到 L
        X_raw = df[I_cols].to_numpy(dtype=np.float32)  # [N, K]
        # 对齐到 L
        K = X_raw.shape[1]
        if K == spectral_length:
            X_list = [row.astype(np.float32) for row in X_raw]
        else:
            # 按列索引线性插值到 L（等距假定）
            src_idx = np.linspace(0, 1, K)
            dst_idx = np.linspace(0, 1, spectral_length)
            for i in range(X_raw.shape[0]):
                X_list.append(np.interp(dst_idx, src_idx, X_raw[i]).astype(np.float32))
        if y_col and y_col in df.columns:
            y_list = [float(v) for v in df[y_col].to_numpy()]
        if c_col and c_col in df.columns:
            c_list = [int(v) for v in df[c_col].to_numpy()]

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N,L]
    X = standardize(X, normalize)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    c = np.array(c_list, dtype=np.int64) if c_list else None
    return X, y, c

# ============================= files 模式 ============================= #

def parse_labels_by_regex(name: str, regex: Optional[str]) -> Tuple[Optional[float], Optional[int]]:
    """从文件名通过正则提取 y/c（可选）。例如: r"y(?P<y>[0-9.]+)_c(?P<c>\d+)""" 
    if not regex:
        return None, None
    m = re.search(regex, name)
    if not m:
        return None, None
    y = m.groupdict().get("y")
    c = m.groupdict().get("c")
    return (float(y) if y is not None else None, int(c) if c is not None else None)


def load_files_mode(folder: Path,
                    spectral_length: int,
                    wavemin: Optional[float],
                    wavemax: Optional[float],
                    label_map: Optional[Path],
                    filename_regex: Optional[str],
                    normalize: str = "none") -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """从目录读取多个 CSV 文件。默认每个文件包含 >=1 条样本。
    支持两列 [wavelength,intensity]、单列强度（默认等距），
    以及“首列为波长、其余列为多条强度”的批量记录格式。
    标签可由 label_map.csv 或文件名正则提取；两者皆无时仅导出 X.npy。
    label_map.csv 需要包含列：file(文件名或相对路径)，可选 y、c。
    """
    files = sorted([p for p in folder.glob("**/*.csv") if p.is_file()])
    if not files:
        raise ValueError(f"目录中未找到 CSV 文件: {folder}")

    lm = None
    if label_map and label_map.exists():
        lm = pd.read_csv(label_map)
        # 统一为仅文件名键匹配（避免不同子目录影响）
        lm["key"] = lm["file"].apply(lambda s: Path(str(s)).name)
        lm = lm.set_index("key")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    c_list: List[int] = []

    for f in files:
        df = pd.read_csv(f, header=None)
        # 将所有值转为数值，方便自动过滤“wavelength,Spec1,...”之类的表头；
        # 若整行/整列均为 NaN，则视为无效并剔除。
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(f"文件 {f} 转成数值后为空，请确认是否包含有效光谱。")

        wave_values: Optional[np.ndarray] = None
        wave_mask: Optional[np.ndarray] = None
        if df.shape[1] >= 2:
            wave_values = df.iloc[:, 0].to_numpy(dtype=np.float64)
            if not np.isfinite(wave_values).all():
                wave_mask = np.isfinite(wave_values)
                valid_cnt = int(wave_mask.sum())
                if valid_cnt < 2:
                    raise ValueError(f"文件 {f} 的波长列有效值不足，无法重采样。")
            else:
                wave_mask = None

        def append_sample(inten_arr: np.ndarray):
            if inten_arr.ndim != 1:
                inten = inten_arr.to_numpy()
            else:
                inten = inten_arr
            inten = np.asarray(inten, dtype=np.float64)
            if wave_values is not None:
                wave = wave_values
                if wave_mask is not None:
                    inten = inten[wave_mask]
                    wave = wave[wave_mask]
                if inten.shape[0] != wave.shape[0]:
                    raise ValueError(f"文件 {f} 中波长与强度长度不一致：{wave.shape[0]} vs {inten.shape[0]}")
                x = resample_to_grid(wave, inten, spectral_length, wavemin, wavemax)
            else:
                K = inten.shape[0]
                if K == spectral_length:
                    x = inten.astype(np.float32)
                else:
                    src_idx = np.linspace(0, 1, K)
                    dst_idx = np.linspace(0, 1, spectral_length)
                    x = np.interp(dst_idx, src_idx, inten).astype(np.float32)
            X_list.append(x)
            if y_val is not None:
                y_list.append(float(y_val))
            if c_val is not None:
                c_list.append(int(c_val))

        # 标签：label_map 优先；否则正则尝试
        y_val: Optional[float] = None
        c_val: Optional[int] = None
        if lm is not None:
            key = f.name
            if key in lm.index:
                row = lm.loc[key]
                if "y" in row and pd.notna(row["y"]):
                    y_val = float(row["y"])
                if "c" in row and pd.notna(row["c"]):
                    c_val = int(row["c"])
        if y_val is None or c_val is None:
            y2, c2 = parse_labels_by_regex(f.name, filename_regex)
            y_val = y_val if y_val is not None else y2
            c_val = c_val if c_val is not None else c2

        if df.shape[1] >= 2:
            # 每一列（除首列波长）代表一个光谱。
            for col_idx in range(1, df.shape[1]):
                append_sample(df.iloc[:, col_idx])
        else:
            append_sample(df.iloc[:, 0])

    X = np.stack(X_list, axis=0).astype(np.float32)
    X = standardize(X, normalize)
    y = np.array(y_list, dtype=np.float32) if len(y_list) == len(X_list) else None
    c = np.array(c_list, dtype=np.int64) if len(c_list) == len(X_list) else None
    return X, y, c

# ============================= 主入口 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["table", "files"], required=True, help="CSV 输入模式：table=一个文件多样本；files=目录下多个文件各一条样本")
    ap.add_argument("--input", type=str, required=True, help="table: CSV 文件路径；files: 目录路径")
    ap.add_argument("--domain", choices=["source", "target"], required=True, help="输出到哪个域的目录")

    # 光谱口径
    ap.add_argument("--spectral_length", type=int, required=True, help="目标统一长度 L")
    ap.add_argument("--wavemin", type=float, default=None, help="波长下界（可选，files 有波长列时常用）")
    ap.add_argument("--wavemax", type=float, default=None, help="波长上界（可选，files 有波长列时常用）")

    # 标准化
    ap.add_argument("--normalize", choices=["none", "standard", "minmax"], default="none", help="按样本归一化方式")

    # table 模式专用
    ap.add_argument("--intensity_prefix", type=str, default=None, help="强度列前缀（如 i_）")
    ap.add_argument("--intensity_cols", type=str, nargs="*", default=None, help="强度列名显式列表（覆盖 prefix）")
    ap.add_argument("--wavelength_col", type=str, default=None, help="波长列列名（可选）")
    ap.add_argument("--y_col", type=str, default=None, help="回归标签列名（可选）")
    ap.add_argument("--c_col", type=str, default=None, help="类别标签列名（可选）")

    # files 模式专用
    ap.add_argument("--label_map", type=str, default=None, help="标签映射 CSV：列包含 file,y,c（file 为文件名或相对路径）")
    ap.add_argument("--filename_regex", type=str, default=None, help="从文件名提取标签的正则，命名捕获组为 y/c，如 y(?P<y>[0-9.]+)_c(?P<c>\\d+)")

    ap.add_argument("--out_root", type=str, default="data", help="输出根目录（将写入 out_root/<domain>/...）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = set_seed(args.seed)
    out_dir = Path(args.out_root) / args.domain
    ensure_dir(out_dir)

    if args.mode == "table":
        X, y, c = load_table_mode(
            Path(args.input), args.spectral_length, args.wavemin, args.wavemax,
            args.intensity_prefix, args.intensity_cols, args.wavelength_col,
            args.y_col, args.c_col, normalize=args.normalize)
    else:
        label_map = Path(args.label_map) if args.label_map else None
        X, y, c = load_files_mode(
            Path(args.input), args.spectral_length, args.wavemin, args.wavemax,
            label_map, args.filename_regex, normalize=args.normalize)

    # 基础健康检查
    if not np.isfinite(X).all():
        raise ValueError("X 中存在 NaN/Inf，请检查 CSV 或参数配置。")
    if X.shape[1] != args.spectral_length:
        raise ValueError(f"得到的光谱长度为 {X.shape[1]}，与期望 {args.spectral_length} 不一致。")

    np.save(out_dir / "X.npy", X.astype(np.float32))
    print(f"已保存: {out_dir/'X.npy'}  形状={X.shape} dtype=float32")

    if y is not None:
        np.save(out_dir / "y.npy", y.astype(np.float32))
        print(f"已保存: {out_dir/'y.npy'}  形状={y.shape} dtype=float32")
    else:
        print("未提供 y 标签，跳过 y.npy。")

    if c is not None:
        np.save(out_dir / "c.npy", c.astype(np.int64))
        print(f"已保存: {out_dir/'c.npy'}  形状={c.shape} dtype=int64")
    else:
        print("未提供 c 标签，跳过 c.npy。")

    # 简要统计
    def stats(x: np.ndarray):
        return dict(shape=tuple(x.shape), min=float(x.min()), max=float(x.max()), mean=float(x.mean()))
    print("X 统计:", stats(X))


if __name__ == "__main__":
    main()

