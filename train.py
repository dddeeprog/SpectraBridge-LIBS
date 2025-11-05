#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1/train.py —— 统一版单模态光谱训练脚本（含 UDA/SSDA）

特性概览：
- 阶段：pretrain（仅源域监督）/ uda（无监督目标域 + Deep CORAL）/ ssda（少量目标标注微调）
- 结构：共享编码器 + 分类头 + 每类回归头 + 全局回归头（Soft Routing 可选回退）
- 损失：CE 或 Focal（分类），MSE（回归），Deep CORAL（二阶统计对齐）
- 训练：AMP 可选、早停、最佳模型保存、Cosine 调度（含 warmup）
- 数据：NPY/NPZ 数据集（X.npy / y.npy / c.npy），DataLoader 接入统一随机种子

用法示例：
  python S1/train.py --config S1/config.yaml --save artifacts/best.pt
  python S1/train.py --config S1/config.yaml --resume artifacts/best_pretrain.pt --save artifacts/best_uda.pt
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# —— 项目内模块 ——
from mgtl.data import NpyDataset, NpyDualIter
from mgtl.models import SpecEncoder, ClassifierHead, PerClassRegressors, GlobalRegressor
from mgtl.losses import coral_loss, FocalLoss
from mgtl.utils.metrics import RegressionMetrics, ClassificationMetrics
from mgtl.utils.seed import set_seed, seed_worker, get_generator
from mgtl.utils.logging import get_logger, Progress


# ============================= 配置与模型封装 ============================= #
@dataclass
class TrainCfg:
    stage: str
    profile: str
    epochs: int
    amp: bool
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    # early stopping
    es_metric: str
    es_mode: str
    es_patience: int
    # optimizer
    opt_name: str
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    # scheduler
    sch_name: str
    warmup_epochs: int
    min_lr: float
    # freeze & lr multipliers
    freeze_encoder: bool
    freeze_classifier: bool
    lr_mul_encoder: float
    lr_mul_heads: float
    # loss weights
    w_ce: float
    w_soft: float
    w_pcreg: float
    w_greg: float
    w_coral: float
    mean_align_weight: float
    coral_normalize: str
    # classification imbalance
    use_focal: bool
    focal_gamma: float
    focal_alpha: Optional[np.ndarray]
    # UDA/SSDA
    uda_target_ratio: float
    ssda_epochs: int


class UnifiedModel(nn.Module):
    """共享编码器 + 分类头 + 多头回归 + 全局回归。"""
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


# ============================= 实用函数 ============================= #

def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(mode: str) -> torch.device:
    if mode == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_cfg(cfg: Dict) -> TrainCfg:
    t = cfg["train"]
    lw = t["loss_weights"]
    cla = t.get("classification", {})
    uda = t.get("uda", {})
    ssda = t.get("ssda", {})
    focal_alpha = cla.get("focal_alpha")
    if focal_alpha is not None:
        focal_alpha = np.asarray(focal_alpha, dtype=np.float32)
    return TrainCfg(
        stage=str(t.get("stage", "uda")),
        profile=str(t.get("profile", cfg.get("profile", "soft"))),
        epochs=int(t.get("epochs", 50)),
        amp=bool(t.get("amp", True)),
        batch_size=int(t.get("batch_size", 64)),
        num_workers=int(t.get("num_workers", 2)),
        pin_memory=bool(t.get("pin_memory", True)),
        shuffle=bool(t.get("shuffle", True)),
        es_metric=str(t.get("early_stopping", {}).get("metric", "reg/rmse_macro")),
        es_mode=str(t.get("early_stopping", {}).get("mode", "min")),
        es_patience=int(t.get("early_stopping", {}).get("patience", 8)),
        opt_name=str(t.get("optimizer", {}).get("name", "adamw")).lower(),
        lr=float(t.get("optimizer", {}).get("lr", 3e-4)),
        weight_decay=float(t.get("optimizer", {}).get("weight_decay", 1e-4)),
        betas=tuple(t.get("optimizer", {}).get("betas", [0.9, 0.999])),
        sch_name=str(t.get("scheduler", {}).get("name", "cosine")).lower(),
        warmup_epochs=int(t.get("scheduler", {}).get("warmup_epochs", 5)),
        min_lr=float(t.get("scheduler", {}).get("min_lr", 1e-6)),
        freeze_encoder=bool(t.get("freeze", {}).get("encoder", False)),
        freeze_classifier=bool(t.get("freeze", {}).get("classifier", False)),
        lr_mul_encoder=float(t.get("lr_multipliers", {}).get("encoder", 1.0)),
        lr_mul_heads=float(t.get("lr_multipliers", {}).get("heads", 1.0)),
        w_ce=float(lw.get("ce_weight", 1.0)),
        w_soft=float(lw.get("softmix_weight", 1.0)),
        w_pcreg=float(lw.get("per_class_reg_weight", 0.5)),
        w_greg=float(lw.get("global_reg_weight", 0.5)),
        w_coral=float(lw.get("coral_weight", 0.1)),
        mean_align_weight=float(lw.get("mean_align_weight", 0.0)),
        coral_normalize=str(lw.get("coral_normalize", "d2")),
        use_focal=bool(cla.get("use_focal", False)),
        focal_gamma=float(cla.get("focal_gamma", 2.0)),
        focal_alpha=focal_alpha,
        uda_target_ratio=float(uda.get("target_batch_ratio", 1.0)),
        ssda_epochs=int(ssda.get("epochs", 10)),
    )


def build_dataloaders(cfg: Dict, tc: TrainCfg, logger) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    L = int(cfg["spectral_length"])
    norm = cfg.get("data", {}).get("normalize", "standard")
    mmap = bool(cfg.get("data", {}).get("mmap", False))
    seed = int(cfg.get("seed", 42))

    src_dir = Path(cfg["paths"]["source_dir"]).expanduser()
    tgt_dir = Path(cfg["paths"]["target_dir"]).expanduser()

    g = get_generator(seed)

    src_ds, src_dl = None, None
    if (src_dir / "X.npy").exists():
        src_ds = NpyDataset(src_dir / "X.npy", src_dir / "y.npy", src_dir / "c.npy", L, norm, mmap)
        src_dl = DataLoader(
            src_ds,
            batch_size=tc.batch_size,
            shuffle=tc.shuffle,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        logger.warning(f"未找到源域数据：{src_dir/'X.npy'}，pretrain/uda 可能无法进行。")

    tgt_ds, tgt_dl = None, None
    if (tgt_dir / "X.npy").exists():
        tgt_ds = NpyDataset(tgt_dir / "X.npy", tgt_dir / "y.npy", tgt_dir / "c.npy", L, norm, mmap)
        tgt_dl = DataLoader(
            tgt_ds,
            batch_size=tc.batch_size,
            shuffle=tc.shuffle,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        logger.warning(f"未找到目标域数据：{tgt_dir/'X.npy'}，uda/ssda 将仅做源域训练。")

    return src_dl, tgt_dl


def build_optimizer(model: UnifiedModel, tc: TrainCfg):
    params = []
    # 冻结与分组学习率
    if tc.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    else:
        params.append({"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": tc.lr * tc.lr_mul_encoder})

    if tc.freeze_classifier:
        for p in model.classifier.parameters():
            p.requires_grad_(False)
    else:
        params.append({"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": tc.lr * tc.lr_mul_heads})

    params.append({"params": [p for p in model.per_class_reg.parameters() if p.requires_grad], "lr": tc.lr * tc.lr_mul_heads})
    params.append({"params": [p for p in model.global_reg.parameters() if p.requires_grad], "lr": tc.lr * tc.lr_mul_heads})

    if tc.opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay, betas=tc.betas)
    elif tc.opt_name == "adam":
        opt = torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay, betas=tc.betas)
    else:
        raise ValueError(f"不支持的优化器：{tc.opt_name}")
    return opt


def build_scheduler(opt: torch.optim.Optimizer, tc: TrainCfg, total_epochs: int):
    if tc.sch_name == "cosine":
        # 线性 warmup + 余弦退火到 min_lr
        def lr_lambda(epoch):
            if epoch < tc.warmup_epochs:
                return (epoch + 1) / max(tc.warmup_epochs, 1)
            # 余弦从 1 到 min_lr/lr
            t = (epoch - tc.warmup_epochs) / max(total_epochs - tc.warmup_epochs, 1)
            cos = 0.5 * (1 + np.cos(np.pi * t))
            return max(tc.min_lr / tc.lr, cos)
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1.0)


# ============================= 评估流程 ============================= #
@torch.no_grad()
def evaluate(model: UnifiedModel, dl: DataLoader, device: torch.device, profile: str, n_classes: int) -> Dict[str, float]:
    model.eval()
    reg = RegressionMetrics(); reg_m = reg.new_meter()
    cls = ClassificationMetrics(n_classes=n_classes); cls_m = cls.new_meter()

    for batch in dl:
        x = batch["x"].to(device)
        y = batch.get("y")
        c = batch.get("c")
        if y is not None: y = y.to(device)
        if c is not None: c = c.to(device)

        out = model(x)
        logits, y_pc, y_g = out["logits"], out["per_cls"], out["y_g"]
        p = F.softmax(logits, dim=1)
        y_soft = (p * y_pc).sum(dim=1)
        y_hat = y_soft if profile == "soft" else y_g

        if y is not None:
            reg.update_meter(reg_m, y_hat.detach().cpu(), y.detach().cpu())
        if c is not None:
            cls.update_meter(cls_m, logits.detach().cpu(), c.detach().cpu())

    res = {}
    res.update(reg.compute(reg_m, prefix="reg/"))
    res.update(cls.compute(cls_m, prefix="cls/"))
    return res


def metric_better(a: float, b: float, mode: str) -> bool:
    return (a < b) if mode == "min" else (a > b)


# ============================= 主训练循环 ============================= #

def train_one_epoch(model: UnifiedModel, opt, scaler, dl_s: Optional[DataLoader], dl_t: Optional[NpyDualIter],
                    device: torch.device, tc: TrainCfg, n_classes: int, logger, epoch: int) -> Dict[str, float]:
    model.train()

    # 损失器
    if tc.use_focal:
        if tc.focal_alpha is None:
            ce_crit = FocalLoss(gamma=tc.focal_gamma)
        else:
            ce_crit = FocalLoss(alpha=tc.focal_alpha, gamma=tc.focal_gamma)
    else:
        ce_crit = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # 统计
    total_steps = len(dl_s) if dl_s is not None else 0
    prog = Progress(total=total_steps, prefix=f"Epoch {epoch}")

    # 主循环：以源域为驱动
    agg = {"loss": 0.0, "ce": 0.0, "soft": 0.0, "pcreg": 0.0, "greg": 0.0, "coral": 0.0}
    for step, batch_s in enumerate(dl_s or [], 1):
        x_s = batch_s["x"].to(device)
        y_s = batch_s.get("y")
        c_s = batch_s.get("c")
        if y_s is not None: y_s = y_s.to(device)
        if c_s is not None: c_s = c_s.to(device)

        # 目标域 batch（UDA/SSDA）
        x_t = y_t = c_t = None
        if tc.stage in {"uda", "ssda"} and dl_t is not None:
            bt = next(dl_t)
            x_t = bt["x"].to(device)
            if bt.get("y") is not None: y_t = bt["y"].to(device)
            if bt.get("c") is not None: c_t = bt["c"].to(device)

        with torch.cuda.amp.autocast(enabled=tc.amp):
            # 源域前向
            out_s = model(x_s)
            logits_s, ypc_s, yg_s, z_s = out_s["logits"], out_s["per_cls"], out_s["y_g"], out_s["z"]
            p_s = F.softmax(logits_s, dim=1)
            ysoft_s = (p_s * ypc_s).sum(dim=1)

            loss = torch.zeros((), device=device)
            # 分类损失
            if c_s is not None:
                loss_ce = ce_crit(logits_s, c_s) * tc.w_ce
                loss = loss + loss_ce
            else:
                loss_ce = torch.zeros_like(loss)
            # 回归损失（软加权 + 每类 + 全局）
            if y_s is not None:
                l_soft = mse(ysoft_s, y_s) * tc.w_soft
                l_g = mse(yg_s, y_s) * tc.w_greg
                # 仅当有真实类别 c_s 才能对齐该类回归头
                if c_s is not None:
                    ypc_true = ypc_s.gather(1, c_s.view(-1, 1)).squeeze(1)
                    l_pc = mse(ypc_true, y_s) * tc.w_pcreg
                else:
                    l_pc = torch.zeros_like(loss)
                loss = loss + l_soft + l_g + l_pc
            else:
                l_soft = l_g = l_pc = torch.zeros_like(loss)

            # 目标域：CORAL 对齐 +（SSDA 可选监督）
            l_coral = torch.zeros_like(loss)
            if tc.stage in {"uda", "ssda"} and x_t is not None:
                out_t = model(x_t)
                z_t = out_t["z"]
                l_coral = coral_loss(z_s, z_t, mean_align_weight=tc.mean_align_weight, normalize=tc.coral_normalize) * tc.w_coral
                loss = loss + l_coral
                if tc.stage == "ssda" and (y_t is not None or c_t is not None):
                    # 简单策略：使用与源域相同的监督项，但整体缩放 0.5
                    scale = 0.5
                    if c_t is not None:
                        loss = loss + ce_crit(out_t["logits"], c_t) * tc.w_ce * scale
                    if y_t is not None:
                        p_t = F.softmax(out_t["logits"], dim=1)
                        ysoft_t = (p_t * out_t["per_cls"]).sum(dim=1)
                        loss = loss + mse(ysoft_t, y_t) * tc.w_soft * scale
                        loss = loss + mse(out_t["y_g"], y_t) * tc.w_greg * scale
                        if c_t is not None:
                            ypc_true_t = out_t["per_cls"].gather(1, c_t.view(-1, 1)).squeeze(1)
                            loss = loss + mse(ypc_true_t, y_t) * tc.w_pcreg * scale

        # 反传
        opt.zero_grad(set_to_none=True)
        if tc.amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # 统计
        agg["loss"] += float(loss.detach().cpu())
        agg["ce"] += float(loss_ce.detach().cpu())
        agg["soft"] += float(l_soft.detach().cpu())
        agg["pcreg"] += float(l_pc.detach().cpu())
        agg["greg"] += float(l_g.detach().cpu())
        agg["coral"] += float(l_coral.detach().cpu())
        prog.step()

    # 均值化
    steps = max(total_steps, 1)
    for k in agg:
        agg[k] /= steps
    return agg


# ============================= 主入口 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径（config.yaml）")
    ap.add_argument("--resume", type=str, default=None, help="从权重恢复/热启动训练（可选）")
    ap.add_argument("--save", type=str, default=None, help="最佳权重保存路径（默认取 config.save.best_ckpt）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(level=cfg.get("logging", {}).get("level", "INFO"), logfile=cfg.get("logging", {}).get("logfile"))

    # 设备与随机性
    device = select_device(cfg.get("device", "auto"))
    set_seed(int(cfg.get("seed", 42)))

    # 训练配置
    tc = build_train_cfg(cfg)

    # 构建模型
    n_classes = int(cfg["n_classes"])
    d_model = int(cfg["model"]["d_model"])
    model = UnifiedModel(d_model=d_model, n_classes=n_classes, model_cfg=cfg["model"]).to(device)

    # 恢复权重（若提供）
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(f"load_state: missing={len(missing)} unexpected={len(unexpected)}（阶段/形态变化常见）")

    # 优化器与调度器
    opt = build_optimizer(model, tc)
    scheduler = build_scheduler(opt, tc, total_epochs=tc.epochs)

    # 混合精度 scaler
    scaler = torch.cuda.amp.GradScaler(enabled=tc.amp)

    # 数据加载器（带随机种子）
    dl_s, dl_t_raw = build_dataloaders(cfg, tc, logger)
    dl_t = NpyDualIter(dl_t_raw) if dl_t_raw is not None else None

    # 训练循环与早停
    best_metric = float("inf") if tc.es_mode == "min" else -float("inf")
    best_path = Path(args.save or cfg.get("save", {}).get("best_ckpt", "artifacts/best.pt"))
    best_path.parent.mkdir(parents=True, exist_ok=True)
    patience = tc.es_patience

    for epoch in range(1, tc.epochs + 1):
        # 1) 训练一个 epoch
        stats = train_one_epoch(model, opt, scaler, dl_s, dl_t, device, tc, n_classes, logger, epoch)
        scheduler.step()

        # 2) 评估（uda/ssda 优先在目标域评估，否则源域）
        eval_dl = dl_t_raw if (tc.stage in {"uda", "ssda"} and dl_t_raw is not None) else dl_s
        eval_domain = "target" if eval_dl is dl_t_raw else "source"
        eval_res = evaluate(model, eval_dl, device, profile=tc.profile, n_classes=n_classes)

        # 3) 提取监控指标
        monitor = eval_res.get(tc.es_metric)
        if monitor is None:
            # 回退：默认使用 reg/rmse_macro
            monitor = eval_res.get("reg/rmse_macro", None)
        logger.info(f"Epoch {epoch:03d} | train: {stats} | eval[{eval_domain}]: {eval_res}")

        # 4) 早停/保存
        improved = metric_better(monitor, best_metric, tc.es_mode)
        if improved:
            best_metric = monitor
            patience = tc.es_patience
            # 保存
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metric": monitor,
                "config": cfg,
            }, best_path)
            logger.info(f"[save] best @ epoch {epoch}: {tc.es_metric}={monitor:.6f} → {best_path}")
        else:
            patience -= 1
            if patience <= 0:
                logger.info("[early-stop] patience exhausted, stop training.")
                break

    logger.info("训练结束。")


if __name__ == "__main__":
    main()
