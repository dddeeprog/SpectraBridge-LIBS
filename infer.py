#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1/infer.py —— 统一版推理脚本（含软加权回退、指数平滑、可选指标 JSON）

功能：
- 读取 config.yaml 与权重，构建统一模型（共享编码器 + 分类/回归头）；
- 从 `--input_dir` 读取 NPY/NPZ（X.npy 必需；y.npy/c.npy 可选）进行批量推理；
- 输出逐样本 CSV：包含 y_global / y_soft / y_hat（口径选择+回退+平滑后）、分类 top1 与置信度；
- 若提供 y/c，则计算回归/分类指标并可选写出 JSON（便于快速验收）；
- 新增 `--obs_out` 推理概览（吞吐、置信度/回退率、元数据等）与 `--jsonl_out` 逐样本日志，方便接入监控与审计。

软加权回退与平滑：
- 先计算 soft 回归 y_soft = Σ softmax(logits)_c · y_per_class_c；
- 若 max softmax(logits) < τ，则回退到 y_global；
- 将最终 y_hat 做指数平滑：\tilde{y}_t = α·y_hat_t + (1-α)·\tilde{y}_{t-1}（按样本顺序）。

用法示例：
  python S1/infer.py --config S1/config.yaml --ckpt artifacts/best.pt \
    --input_dir data/target --out artifacts/infer_target.csv \
    --json_out artifacts/infer_target.json --alpha 0.2 --tau 0.6
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# —— 项目内 ——
from mgtl.data import NpyDataset
from mgtl.models import SpecEncoder, ClassifierHead, PerClassRegressors, GlobalRegressor
from mgtl.utils.checkpoint import load_state_strict
from mgtl.utils.config_checks import ConfigValidationError, validate_infer_config
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


def load_class_map(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            # 支持 {"0":"A","1":"B"} 或 {"A":0,...} 两种写法
            if all(k.isdigit() for k in obj.keys()):
                # 索引->名称
                out = []
                for i in range(len(obj)):
                    out.append(str(obj[str(i)]))
                return out
            else:
                # 名称->索引
                inv = {int(v): str(k) for k, v in obj.items()}
                return [inv[i] for i in range(len(inv))]
    except Exception:
        return None


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
                 seed: int) -> DataLoader:
    x = data_dir / "X.npy"
    if not x.exists():
        raise FileNotFoundError(f"未找到输入文件：{x}")
    ds = NpyDataset(x, data_dir / "y.npy", data_dir / "c.npy", spectral_length=L, normalize=normalize)
    g = get_generator(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,                 # 推理必须保持顺序
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dl


# ============================= 推理主流程 ============================= #
@torch.no_grad()
def run_infer(model: UnifiedModel, dl: DataLoader, device: torch.device, *,
              profile: str, tau: float, alpha: float,
              class_names: Optional[List[str]] = None,
              has_y: bool = False, has_c: bool = False,
              logger=None) -> Tuple[List[Dict], Dict[str, Dict[str, float]], Dict[str, float]]:
    model.eval()
    records: List[Dict] = []

    # 指标收集（若有标签）
    reg_soft = RegressionMetrics(); m_soft = reg_soft.new_meter()
    reg_glob = RegressionMetrics(); m_glob = reg_glob.new_meter()
    reg_hat  = RegressionMetrics(); m_hat  = reg_hat.new_meter()
    cls = ClassificationMetrics(n_classes=model.classifier.fc2.out_features); m_cls = cls.new_meter()

    # EMA 平滑缓存
    y_ema = None
    idx_base = 0
    obs_raw = {
        "total_samples": 0,
        "soft_selected": 0,
        "global_fallback": 0,
        "conf_sum": 0.0,
        "conf_sq_sum": 0.0,
        "conf_min": float("inf"),
        "conf_max": float("-inf"),
        "invalid_outputs": 0,
    }

    for batch in dl:
        x = batch["x"].to(device)
        y = batch.get("y"); y = y.to(device) if y is not None else None
        c = batch.get("c"); c = c.to(device) if c is not None else None

        out = model(x)
        logits, y_pc, y_g = out["logits"], out["per_cls"], out["y_g"]
        p = F.softmax(logits, dim=1)
        conf, cls_pred = torch.max(p, dim=1)      # [B]
        y_soft = (p * y_pc).sum(dim=1)            # [B]

        # 回退与口径选择
        use_soft = conf >= tau
        y_hat = torch.where(use_soft, y_soft, y_g)
        if profile == "global_only":
            y_hat = y_g

        # EMA 平滑（按数据原顺序累积）
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float64)
        for i in range(y_hat_np.shape[0]):
            val = y_hat_np[i]
            if y_ema is None:
                y_ema = float(val)
            else:
                y_ema = alpha * float(val) + (1.0 - alpha) * y_ema
            # 写回平滑值
            y_hat_np[i] = y_ema

        invalid_mask = ~np.isfinite(y_hat_np)
        invalid_cnt = int(invalid_mask.sum())
        if invalid_cnt:
            obs_raw["invalid_outputs"] += invalid_cnt
            if logger is not None:
                logger.warning(f"检测到 {invalid_cnt} 个 NaN/Inf 推理结果，已以 0.0 填充")
            y_hat_np[invalid_mask] = 0.0

        # 逐样本记录
        B = x.size(0)
        for i in range(B):
            rec = {
                "idx": idx_base + i,
                "y_global": float(y_g[i].detach().cpu()),
                "y_soft": float(y_soft[i].detach().cpu()),
                "y_hat": float(y_hat_np[i]),
                "cls_pred": int(cls_pred[i].detach().cpu()),
                "cls_conf": float(conf[i].detach().cpu()),
                "used_soft": int(use_soft[i].detach().cpu()),
            }
            if class_names is not None and 0 <= rec["cls_pred"] < len(class_names):
                rec["cls_name"] = class_names[rec["cls_pred"]]
            if has_y and (y is not None):
                yt = float(y[i].detach().cpu())
                rec["y_true"] = yt
                rec["abs_err_hat"] = abs(rec["y_hat"] - yt)
                rec["abs_err_soft"] = abs(float(y_soft[i].detach().cpu()) - yt)
                rec["abs_err_global"] = abs(float(y_g[i].detach().cpu()) - yt)
            if has_c and (c is not None):
                rec["c_true"] = int(c[i].detach().cpu())
                if class_names is not None and 0 <= rec["c_true"] < len(class_names):
                    rec["c_name"] = class_names[rec["c_true"]]
            records.append(rec)
        idx_base += B

        conf_np = conf.detach().cpu().numpy().astype(np.float64)
        use_soft_np = use_soft.detach().cpu().numpy().astype(np.int64)
        obs_raw["total_samples"] += B
        obs_raw["soft_selected"] += int(use_soft_np.sum())
        obs_raw["global_fallback"] += int(B - use_soft_np.sum())
        obs_raw["conf_sum"] += float(conf_np.sum())
        obs_raw["conf_sq_sum"] += float(np.square(conf_np).sum())
        obs_raw["conf_min"] = float(min(obs_raw["conf_min"], float(conf_np.min())))
        obs_raw["conf_max"] = float(max(obs_raw["conf_max"], float(conf_np.max())))

        # 指标累计
        if has_y and (y is not None):
            reg_soft.update_meter(m_soft, y_soft.detach().cpu(), y.detach().cpu())
            reg_glob.update_meter(m_glob, y_g.detach().cpu(), y.detach().cpu())
            reg_hat.update_meter(m_hat, torch.from_numpy(y_hat_np), y.detach().cpu())
        if has_c and (c is not None):
            cls.update_meter(m_cls, logits.detach().cpu(), c.detach().cpu())

    # 汇总指标
    metrics: Dict[str, Dict[str, float]] = {}
    if has_y:
        metrics["reg_hat"] = reg_hat.compute(m_hat, prefix="reg/")
        metrics["reg_soft"] = reg_soft.compute(m_soft, prefix="reg/")
        metrics["reg_global"] = reg_glob.compute(m_glob, prefix="reg/")
    if has_c:
        metrics["cls"] = cls.compute(m_cls, prefix="cls/")

    obs: Dict[str, float] = {}
    total = obs_raw["total_samples"]
    obs["total_samples"] = total
    obs["soft_selected"] = obs_raw["soft_selected"]
    obs["global_fallback"] = obs_raw["global_fallback"]
    obs["fallback_rate"] = (obs_raw["global_fallback"] / total) if total else 0.0
    if total:
        mean = obs_raw["conf_sum"] / total
        var = max(obs_raw["conf_sq_sum"] / total - mean * mean, 0.0)
        obs["cls_conf_mean"] = mean
        obs["cls_conf_std"] = math.sqrt(var)
        obs["cls_conf_min"] = obs_raw["conf_min"]
        obs["cls_conf_max"] = obs_raw["conf_max"]
    else:
        obs["cls_conf_mean"] = 0.0
        obs["cls_conf_std"] = 0.0
        obs["cls_conf_min"] = 0.0
        obs["cls_conf_max"] = 0.0
    obs["invalid_outputs"] = obs_raw["invalid_outputs"]

    return records, metrics, obs


# ============================= 主入口 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径（config.yaml）")
    ap.add_argument("--ckpt", type=str, required=True, help="权重文件路径（.pt）")
    ap.add_argument("--input_dir", type=str, default=None, help="输入目录（含 X.npy/y.npy/c.npy），默认取 config.paths.target_dir")
    ap.add_argument("--out", type=str, default="artifacts/infer.csv", help="输出 CSV 路径")
    ap.add_argument("--json_out", type=str, default=None, help="可选：输出指标 JSON 路径（存在 y/c 时有效）")
    ap.add_argument("--obs_out", type=str, default=None,
                    help="可选：输出推理概览 JSON（包含吞吐、回退率等可观测信息）")
    ap.add_argument("--jsonl_out", type=str, default=None,
                    help="可选：逐样本 JSON Lines 记录，便于实时监控/集成")
    ap.add_argument("--alpha", type=float, default=None, help="EMA 平滑系数（覆盖 config.infer.alpha）")
    ap.add_argument("--tau", type=float, default=None, help="软加权回退阈值（覆盖 config.infer.tau）")
    ap.add_argument("--profile", type=str, default=None, choices=["soft", "global_only"], help="推理口径（覆盖 config.profile）")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--allow-missing-keys", action="store_true",
                    help="加载权重时若存在缺失/多余键则继续（默认严格校验并报错）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(level=cfg.get("logging", {}).get("level", "INFO"), logfile=cfg.get("logging", {}).get("logfile"))

    # 设备/随机
    device = select_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 推理参数
    alpha = float(args.alpha if args.alpha is not None else cfg.get("infer", {}).get("alpha", 0.0))
    tau = float(args.tau if args.tau is not None else cfg.get("infer", {}).get("tau", 0.5))
    profile = str(args.profile if args.profile is not None else cfg.get("profile", "soft"))

    # 模型
    n_classes = int(cfg["n_classes"])
    d_model = int(cfg["model"]["d_model"])
    model = UnifiedModel(d_model=d_model, n_classes=n_classes, model_cfg=cfg["model"]).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    load_state_strict(model, state, allow_missing=args.allow_missing_keys, logger=logger)

    # 数据
    L = int(cfg["spectral_length"])
    normalize = cfg.get("data", {}).get("normalize", "standard")
    input_dir = Path(args.input_dir or cfg["paths"]["target_dir"]).expanduser()
    try:
        validate_infer_config(cfg, input_dir=input_dir, alpha=alpha, tau=tau, profile=profile)
    except ConfigValidationError as err:
        raise SystemExit(str(err)) from err
    dl = build_loader(input_dir, L, normalize, args.batch_size, args.num_workers, args.pin_memory, seed)

    # 类别名称（可选）
    class_map = load_class_map(Path(cfg.get("paths", {}).get("class_map", "")))

    # 推理
    has_y = (input_dir / "y.npy").exists()
    has_c = (input_dir / "c.npy").exists()
    wall_start = time.perf_counter()
    records, metrics, obs_stats = run_infer(
        model, dl, device, profile=profile, tau=tau, alpha=alpha,
        class_names=class_map,
        has_y=has_y,
        has_c=has_c,
        logger=logger,
    )
    runtime = time.perf_counter() - wall_start

    # 写 CSV
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "idx", "y_global", "y_soft", "y_hat", "cls_pred", "cls_conf", "used_soft",
        "cls_name", "y_true", "abs_err_hat", "abs_err_soft", "abs_err_global", "c_true", "c_name"
    ]
    # 仅输出存在的列
    def row_for(rec: Dict):
        return [rec.get(k, "") for k in header]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            w.writerow(row_for(r))
    logger.info(f"infer csv → {out_csv}")

    # 写 JSON 指标（若可用且指定）
    if args.json_out is not None and (len(metrics) > 0):
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"profile": profile, "alpha": alpha, "tau": tau, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        logger.info(f"infer metrics → {out_json}")

    # 可观测性摘要
    obs_stats["runtime_sec"] = runtime
    obs_stats["throughput_sps"] = (obs_stats["total_samples"] / runtime) if runtime > 0 else None
    obs_stats["profile"] = profile
    obs_stats["alpha"] = alpha
    obs_stats["tau"] = tau
    obs_stats["batch_size"] = args.batch_size
    obs_stats["num_workers"] = args.num_workers
    obs_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "input_dir": str(input_dir.resolve()),
        "has_regression_labels": has_y,
        "has_class_labels": has_c,
        "model_params": sum(p.numel() for p in model.parameters()),
        "stats": obs_stats,
        "metrics": metrics,
    }
    if args.obs_out is not None:
        obs_path = Path(args.obs_out)
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(obs_path, "w", encoding="utf-8") as f:
            json.dump(obs_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"infer observability summary → {obs_path}")
    else:
        logger.info(
            "可观测性摘要：samples=%d fallback_rate=%.3f throughput=%.2f samples/s",
            obs_stats["total_samples"],
            obs_stats.get("fallback_rate", 0.0),
            obs_stats.get("throughput_sps") or 0.0,
        )

    # JSONL 逐样本记录（用于实时监控）
    if args.jsonl_out is not None:
        jsonl_path = Path(args.jsonl_out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        logger.info(f"infer jsonl records → {jsonl_path}")

    logger.info("推理完成。")


if __name__ == "__main__":
    main()
