#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1/eval.py —— 统一版评估脚本（接入随机种子 + 统一指标输出）

功能：
- 读取 config.yaml 与 ckpt，构建共享编码器 + 分类/回归头的统一模型；
- 对 source/target 域分别评估分类与回归指标；
- 同时输出两种口径的回归：soft（先分类再定量）与 global（全局回归），便于横向比较；
- DataLoader 接入统一随机种子（seed_worker/generator），保证可复现；
- 指标以 JSON 文件写出（默认 artifacts/eval.json）。

用法示例：
  python S1/eval.py --config S1/config.yaml --ckpt artifacts/best.pt --domains source target \
      --out artifacts/eval.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# —— 项目内 ——
from mgtl.data import NpyDataset
from mgtl.models import SpecEncoder, ClassifierHead, PerClassRegressors, GlobalRegressor
from mgtl.utils.metrics import RegressionMetrics, ClassificationMetrics
from mgtl.utils.seed import set_seed, seed_worker, get_generator
from mgtl.utils.logging import get_logger


# ============================= 通用工具 ============================= #

def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(mode: str) -> torch.device:
    if mode == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================= 统一模型定义 ============================= #
class UnifiedModel(nn.Module):
    def __init__(self, d_model: int, n_classes: int, model_cfg: Dict):
        super().__init__()
        self.encoder = SpecEncoder(
            d_model=d_model,
            conv_channels=model_cfg.get("conv_channels", [64, 128]),
            conv_kernels=model_cfg.get("conv_kernels", [5, 3]),
            conv_dilation=model_cfg.get("conv_dilation", [1, 2]),
            conv_stride=model_cfg.get("conv_stride", [1, 1]),
            conv_pool=model_cfg.get("conv_pool", "none"),
            transformer_layers=model_cfg.get("transformer_layers", 2),
            nhead=model_cfg.get("nhead", 4),
            dim_ff=model_cfg.get("dim_ff", 512),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "gelu"),
            pooling=model_cfg.get("pooling", "mean"),
            positional_encoding=model_cfg.get("positional_encoding", "learned"),
        )
        self.classifier = ClassifierHead(d_model, n_classes)
        self.per_class_reg = PerClassRegressors(d_model, n_classes)
        self.global_reg = GlobalRegressor(d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)
        per_cls = self.per_class_reg(z)
        y_g = self.global_reg(z).squeeze(-1)
        return {"z": z, "logits": logits, "per_cls": per_cls, "y_g": y_g}


# ============================= 数据加载 ============================= #

def build_loader(data_dir: Path, L: int, normalize: str, batch_size: int, num_workers: int, pin_memory: bool,
                 seed: int) -> Optional[DataLoader]:
    x = data_dir / "X.npy"
    if not x.exists():
        return None
    ds = NpyDataset(x, data_dir / "y.npy", data_dir / "c.npy", spectral_length=L, normalize=normalize)
    g = get_generator(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,                 # 评估不洗牌
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dl


# ============================= 评估流程 ============================= #
@torch.no_grad()
def eval_one(model: UnifiedModel, dl: DataLoader, device: torch.device, n_classes: int) -> Dict[str, Dict[str, float]]:
    model.eval()
    # 两套回归口径的度量器
    reg_soft = RegressionMetrics(); m_soft = reg_soft.new_meter()
    reg_glob = RegressionMetrics(); m_glob = reg_glob.new_meter()
    cls = ClassificationMetrics(n_classes=n_classes); m_cls = cls.new_meter()

    for batch in dl:
        x = batch["x"].to(device)
        y = batch.get("y"); y = y.to(device) if y is not None else None
        c = batch.get("c"); c = c.to(device) if c is not None else None

        out = model(x)
        logits, y_pc, y_g = out["logits"], out["per_cls"], out["y_g"]
        p = F.softmax(logits, dim=1)
        y_soft = (p * y_pc).sum(dim=1)

        if y is not None:
            reg_soft.update_meter(m_soft, y_soft.detach().cpu(), y.detach().cpu())
            reg_glob.update_meter(m_glob, y_g.detach().cpu(), y.detach().cpu())
        if c is not None:
            cls.update_meter(m_cls, logits.detach().cpu(), c.detach().cpu())

    res = {
        "reg_soft": reg_soft.compute(m_soft, prefix="reg/"),
        "reg_global": reg_glob.compute(m_glob, prefix="reg/"),
        "cls": cls.compute(m_cls, prefix="cls/"),
    }
    return res


# ============================= 主入口 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径（config.yaml）")
    ap.add_argument("--ckpt", type=str, required=True, help="权重文件路径（.pt）")
    ap.add_argument("--domains", nargs="*", default=["source", "target"], choices=["source", "target"],
                    help="评估哪些域：source/target，多选")
    ap.add_argument("--out", type=str, default="artifacts/eval.json", help="输出指标 JSON 路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(level=cfg.get("logging", {}).get("level", "INFO"), logfile=cfg.get("logging", {}).get("logfile"))

    # 设备与随机性
    device = select_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 构建模型并加载权重
    n_classes = int(cfg["n_classes"])
    d_model = int(cfg["model"]["d_model"])
    model = UnifiedModel(d_model=d_model, n_classes=n_classes, model_cfg=cfg["model"]).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state: missing={len(missing)} unexpected={len(unexpected)}（阶段/形态变化常见）")

    # DataLoader 参数
    L = int(cfg["spectral_length"])
    normalize = cfg.get("data", {}).get("normalize", "standard")
    bs = int(cfg["train"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 2))
    pm = bool(cfg["train"].get("pin_memory", True))

    # 域路径
    src_dir = Path(cfg["paths"]["source_dir"]).expanduser()
    tgt_dir = Path(cfg["paths"]["target_dir"]).expanduser()

    results: Dict[str, Dict] = {"profile": cfg.get("profile", "soft"), "domains": {}}

    # 逐域评估
    if "source" in args.domains:
        dl_s = build_loader(src_dir, L, normalize, bs, nw, pm, seed)
        if dl_s is None:
            logger.warning(f"跳过 source：未找到 {src_dir/'X.npy'}")
        else:
            res_s = eval_one(model, dl_s, device, n_classes)
            results["domains"]["source"] = res_s
            logger.info(f"source: {res_s}")

    if "target" in args.domains:
        dl_t = build_loader(tgt_dir, L, normalize, bs, nw, pm, seed)
        if dl_t is None:
            logger.warning(f"跳过 target：未找到 {tgt_dir/'X.npy'}")
        else:
            res_t = eval_one(model, dl_t, device, n_classes)
            results["domains"]["target"] = res_t
            logger.info(f"target: {res_t}")

    # 写出 JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"eval done → {out}")


if __name__ == "__main__":
    main()
