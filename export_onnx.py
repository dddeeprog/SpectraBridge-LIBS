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

新增：
- --precision 可选择 fp32/fp16，方便快速生成 TensorRT/ORT-friendly 的半精度模型。
- --verify/--bundle-dir/--trt-batch-profile 等参数，可以输出部署元数据、TensorRT profile
  JSON，并在导出后自动做 onnx.checker 与 onnxruntime 推理自检。

注意：
- 仅对“批维”设置动态维度（dynamic_axes=True 时）。序列长度 L 不建议设为动态：
  learned positional encoding / 卷积核尺寸通常绑定固定 L。
- 若 config.export.simplify=True 且环境已安装 onnx & onnxsim，将尝试做图简化。

用法示例：
  python export_onnx.py --config S1/config.yaml --ckpt artifacts/best.pt \
      --out artifacts/model.onnx --opset 17 --dynamic --precision fp16 \
      --verify --bundle-dir artifacts/bundle --trt-batch-profile 1,4,16
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 项目内依赖（保持与 train/eval/infer 一致）——
from mgtl.models.spec_encoder import SpecEncoder
from mgtl.models.classifier import ClassifierHead
from mgtl.models.regressors import PerClassRegressors, GlobalRegressor
from mgtl.utils.checkpoint import load_state_strict
from mgtl.utils.config_checks import ConfigValidationError, validate_export_config
from mgtl.utils.onnx_utils import (
    build_export_metadata,
    build_trt_profile_dict,
    dump_json,
    dump_metadata,
    parse_trt_profile,
)


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
    ap.add_argument("--allow-missing-keys", action="store_true",
                    help="加载权重时若存在缺失/多余键则继续（默认严格校验并报错）")
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default=None,
                    help="导出精度（覆盖 config.export.precision，默认为 fp32")
    ap.add_argument("--verify", action="store_true",
                    help="导出后使用 onnx.checker + onnxruntime 做一致性校验")
    ap.add_argument("--bundle-dir", type=str, default=None,
                    help="若指定则输出 deployment bundle（metadata + profile）")
    ap.add_argument("--trt-batch-profile", type=str, default=None,
                    help="TensorRT 动态 batch 规格，格式：min,opt,max")
    ap.add_argument("--trt-profile-out", type=str, default=None,
                    help="TensorRT profile JSON 输出路径（默认写在 bundle/out 同目录)")
    ap.add_argument("--skip-simplify", action="store_true",
                    help="强制跳过 onnxsim 简化（即使配置里开启）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        validate_export_config(cfg, ckpt_path=Path(args.ckpt))
    except ConfigValidationError as err:
        raise SystemExit(str(err)) from err

    # 设备与形态
    device = select_device(cfg.get("device", "auto"))
    n_classes = int(cfg["n_classes"])
    d_model = int(cfg["model"]["d_model"])

    # 构造与加载
    model = UnifiedModel(d_model=d_model, n_classes=n_classes, model_cfg=cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    load_state_strict(model, state, allow_missing=args.allow_missing_keys)
    model.eval()

    # 包装 + 虚拟输入
    L = int(cfg["spectral_length"])
    wrapper = ExportWrapper(model).to(device)
    exp_cfg = cfg.get("export", {})
    precision = str(args.precision or exp_cfg.get("precision", "fp32")).lower()
    if precision == "fp16":
        wrapper = wrapper.half()
    dummy = torch.randn(1, L, dtype=torch.float16 if precision == "fp16" else torch.float32,
                        device=device)

    # 导出配置
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
    simplify_cfg = bool(exp_cfg.get("simplify", True))
    simplify = simplify_cfg and not args.skip_simplify
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

    verify_flag = bool(args.verify or exp_cfg.get("verify", False))
    if verify_flag:
        try:
            import onnx
            from onnx import checker

            model_onnx = onnx.load(str(out_path))
            checker.check_model(model_onnx)
            print("[export] onnx.checker passed.")
        except Exception as err:
            raise SystemExit(f"[export] onnx.checker failed: {err}") from err

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            ort_dtype = np.float16 if precision == "fp16" else np.float32
            rng = np.random.default_rng(0)
            ort_input = rng.standard_normal(size=(1, L), dtype=ort_dtype)
            sess.run(None, {sess.get_inputs()[0].name: ort_input})
            print("[export] onnxruntime inference passed.")
        except ImportError:
            print("[export] onnxruntime 未安装，跳过推理验证。")
        except Exception as err:
            raise SystemExit(f"[export] onnxruntime 推理失败：{err}") from err

    bundle_dir = args.bundle_dir or exp_cfg.get("bundle_dir")
    trt_profile_spec = args.trt_batch_profile or exp_cfg.get("trt_batch_profile")
    trt_profile = None
    if trt_profile_spec:
        try:
            trt_profile = parse_trt_profile(str(trt_profile_spec))
        except ValueError as err:
            raise SystemExit(f"TensorRT profile 参数非法：{err}") from err

    trt_profile_out = args.trt_profile_out or exp_cfg.get("trt_profile_out")
    bundle_path = Path(bundle_dir).expanduser() if bundle_dir else None
    metadata_path = bundle_path / "deployment_metadata.json" if bundle_path else None
    if trt_profile and not trt_profile_out:
        if bundle_path:
            trt_profile_out = str(bundle_path / "tensorrt_profile.json")
        else:
            trt_profile_out = str(out_path.with_suffix(".trt-profile.json"))

    trt_profile_dict = None
    if trt_profile:
        trt_profile_dict = build_trt_profile_dict(
            batch_min=trt_profile[0],
            batch_opt=trt_profile[1],
            batch_max=trt_profile[2],
            spectral_length=L,
            precision=precision,
        )
        dump_json(Path(trt_profile_out), trt_profile_dict)
        print(f"[export] TensorRT profile written to {trt_profile_out}")

    if metadata_path:
        metadata = build_export_metadata(
            opset=opset,
            precision=precision,
            dynamic_axes=dynamic_axes is not None,
            spectral_length=L,
            n_classes=n_classes,
            ckpt_path=Path(args.ckpt),
            onnx_path=out_path,
            simplify=simplify,
            verify=verify_flag,
            trt_profile=trt_profile_dict,
        )
        dump_metadata(metadata_path, metadata)
        print(f"[export] deployment metadata written to {metadata_path}")

    print("导出完成。")


if __name__ == "__main__":
    main()
