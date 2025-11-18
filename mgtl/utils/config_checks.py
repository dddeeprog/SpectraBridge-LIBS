"""Simple configuration validation helpers used by the CLI scripts.

These checks are intentionally lightweight—they only guard against the most
common configuration or filesystem mistakes so that failures happen fast and
with actionable messages instead of letting the training/eval/infer scripts run
for minutes before crashing deep inside the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mgtl.utils.onnx_utils import parse_trt_profile


@dataclass
class ConfigValidationError(RuntimeError):
    """Raised when one or more blocking configuration issues are found."""

    issues: List[str]

    def __str__(self) -> str:  # pragma: no cover - trivial string join
        joined = "\n - ".join(self.issues)
        return f"配置校验失败：\n - {joined}" if joined else "配置校验失败"


def _ensure(condition: bool, issues: List[str], message: str) -> None:
    if not condition:
        issues.append(message)


def _check_required_paths(paths: Dict, required: Sequence[str], issues: List[str], *,
                          require_labels: Iterable[str] = ()) -> None:
    for key in required:
        val = paths.get(key)
        _ensure(bool(val), issues, f"paths.{key} 未配置")
        if not val:
            continue
        p = Path(val).expanduser()
        _ensure(p.exists(), issues, f"paths.{key}={p} 不存在")
        if not p.exists():
            continue
        x_file = p / "X.npy"
        _ensure(x_file.exists(), issues, f"{x_file} 不存在，无法构建数据集")
        for extra in require_labels:
            if key == "source_dir":
                target = p / f"{extra}.npy"
                _ensure(target.exists(), issues, f"{target} 不存在，源域需要 {extra}.npy")


def validate_training_config(cfg: Dict) -> None:
    """Validate the minimal set of fields required for training."""

    issues: List[str] = []
    paths = cfg.get("paths", {})
    _check_required_paths(paths, ["source_dir"], issues, require_labels=["y", "c"])
    _check_required_paths(paths, ["target_dir"], issues)

    L = int(cfg.get("spectral_length", 0))
    _ensure(L > 0, issues, "spectral_length 必须为正整数")

    n_classes = int(cfg.get("n_classes", 0))
    _ensure(n_classes > 0, issues, "n_classes 必须为正整数")

    train = cfg.get("train", {})
    stage = str(train.get("stage", "")).lower()
    _ensure(stage in {"pretrain", "uda", "ssda"}, issues, "train.stage 仅支持 pretrain/uda/ssda")

    profile = str(train.get("profile", cfg.get("profile", ""))).lower()
    _ensure(profile in {"soft", "global_only"}, issues, "train.profile/profile 必须为 soft 或 global_only")

    batch_size = int(train.get("batch_size", 0))
    _ensure(batch_size > 0, issues, "train.batch_size 必须大于 0")

    num_workers = int(train.get("num_workers", -1))
    _ensure(num_workers >= 0, issues, "train.num_workers 不能为负")

    if issues:
        raise ConfigValidationError(issues)


def validate_eval_config(cfg: Dict, domains: Sequence[str]) -> None:
    issues: List[str] = []
    _ensure(domains, issues, "至少需要指定一个评估域（--domains）")
    paths = cfg.get("paths", {})
    for dom in domains:
        key = f"{dom}_dir"
        if dom not in {"source", "target"}:
            issues.append(f"未知域：{dom}")
            continue
        _check_required_paths(paths, [key], issues, require_labels=("y", "c") if dom == "source" else ())
    if issues:
        raise ConfigValidationError(issues)


def validate_infer_config(cfg: Dict, *, input_dir: Path, alpha: float, tau: float, profile: str) -> None:
    issues: List[str] = []
    _ensure(0.0 <= alpha <= 1.0, issues, "infer.alpha/--alpha 必须位于 [0,1]")
    _ensure(0.0 <= tau <= 1.0, issues, "infer.tau/--tau 必须位于 [0,1]")
    _ensure(profile in {"soft", "global_only"}, issues, "profile 仅支持 soft/global_only")
    _ensure(input_dir.exists(), issues, f"输入目录 {input_dir} 不存在")
    _ensure((input_dir / "X.npy").exists(), issues, f"{input_dir/'X.npy'} 不存在")
    if issues:
        raise ConfigValidationError(issues)


def validate_export_config(cfg: Dict, *, ckpt_path: Path) -> None:
    issues: List[str] = []
    _ensure(Path(ckpt_path).expanduser().exists(), issues, f"权重文件 {ckpt_path} 不存在")
    L = int(cfg.get("spectral_length", 0))
    _ensure(L > 0, issues, "spectral_length 必须为正整数")
    n_classes = int(cfg.get("n_classes", 0))
    _ensure(n_classes > 0, issues, "n_classes 必须为正整数")
    export_cfg = cfg.get("export", {})
    precision = export_cfg.get("precision")
    if precision is not None:
        val = str(precision).lower()
        _ensure(val in {"fp32", "fp16"}, issues, "export.precision 仅支持 fp32/fp16")
    profile = export_cfg.get("trt_batch_profile")
    if profile:
        try:
            parse_trt_profile(str(profile))
        except ValueError as err:
            issues.append(str(err))
    if issues:
        raise ConfigValidationError(issues)
