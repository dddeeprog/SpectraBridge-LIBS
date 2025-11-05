#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 ONNX（统一版）

- 读取 config.yaml 与权重 ckpt（train.py 导出的 .pt）
- 固定输入口径：x ∈ R[B, L]，L 来自 config.spectral_length
- 输出 4 个张量：
  1) logits      : [B, C]  分类未归一化分数
  2) y_per_class : [B, C]  每类回归头输出（逐类标量）
  3) y_global    : [B, 1]  全局回归输出
  4) y_soft      : [B, 1]  软加权回归（softmax(logits) 与 y_per_class 按类相乘求和）

注意：
- 仅对“批维”设置动态维度（dynamic_axes=True 时）。序列长度 L 不建议设为动态：
  learned positional encoding / 卷积核尺寸通常绑定固定 L。
- 若 config.export.simplify=True，且环境已安装 onnx & onnxsim，将尝试做图简化。

用法示例：
  python export_onnx.py --config S1/config.yaml --ckpt artifacts/best.pt \
      --out artifacts/model.onnx --opset 17 --dynamic
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 项目内依赖（保持与 train/eval/infer 一致）——
from mgtl.models.spec_encoder import SpecEncoder
from mgtl.models.classifier import ClassifierHead
from mgtl.models.regressors import PerClassRegressors, GlobalRegressor


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


# ============================= 统一模型与导出包装 ============================= #
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

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.classifier(z)
        y_pc = self.per_class_reg(z)
        y_g = self.global_reg(z).unsqueeze(-1)  # [B,1]
        return logits, y_pc, y_g


class ExportWrapper(nn.Module):
    """导出用包装：在原模型输出基础上，补充 y_soft 便于部署选择。"""
    def __init__(self, core: UnifiedModel):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor):
        logits, y_pc, y_g = self.core(x)   # logits:[B,C], y_pc:[B,C], y_g:[B,1]
        p = F.softmax(logits, dim=1)
        y_soft = (p * y_pc).sum(dim=1, keepdim=True)  # [B,1]
        return logits, y_pc, y_g, y_soft


# ============================= 主导出流程 ============================= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径（config.yaml）")
    ap.add_argument("--ckpt", type=str, required=True, help="权重文件（.pt）")
    ap.add_argument("--out", type=str, default=None, help="导出 ONNX 路径（覆盖 config.export.out）")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset（覆盖 config.export.onnx_opset）")
    ap.add_argument("--dynamic", action="store_true", help="仅对批维启用动态轴")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 设备与形态
    device = select_device(cfg.get("device", "auto"))
    n_classes = int(cfg["n_classes"])
    d_model = int(cfg["model"]["d_model"])

    # 构造与加载
    model = UnifiedModel(d_model=d_model, n_classes=n_classes, model_cfg=cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}（阶段/形态差异常见，可忽略）")
    model.eval()

    # 包装 + 虚拟输入
    L = int(cfg["spectral_length"])
    wrapper = ExportWrapper(model).to(device)
    dummy = torch.randn(1, L, dtype=torch.float32, device=device)

    # 导出配置
    exp_cfg = cfg.get("export", {})
    opset = int(args.opset or exp_cfg.get("onnx_opset", 17))
    out_path = Path(args.out or exp_cfg.get("out", "artifacts/model.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic or bool(exp_cfg.get("dynamic_axes", True)):
        # 仅放开 batch 维度；序列长度 L 固定
        dynamic_axes = {
            "x": {0: "batch"},
            "logits": {0: "batch"},
            "y_per_class": {0: "batch"},
            "y_global": {0: "batch"},
            "y_soft": {0: "batch"},
        }

    print(f"[export] opset={opset}, dynamic_axes={dynamic_axes is not None}, out={out_path}")

    # 导出
    torch.onnx.export(
        wrapper,
        dummy,
        f=str(out_path),
        input_names=["x"],
        output_names=["logits", "y_per_class", "y_global", "y_soft"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )
    print("[export] raw onnx done.")

    # 图简化（可选）
    simplify = bool(exp_cfg.get("simplify", True))
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            model_onnx = onnx.load(str(out_path))
            simp_model, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(simp_model, str(out_path))
                print("[export] onnx simplified.")
            else:
                print("[export] onnx simplification reported not-ok, keep raw model.")
        except Exception as e:
            print(f"[export] onnx simplification skipped: {e}")

    print("导出完成。")


if __name__ == "__main__":
    main()
