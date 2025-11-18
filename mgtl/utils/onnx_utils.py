"""Helpers for packaging ONNX exports for downstream deployment stacks."""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

try:  # torch may be unavailable when only doing config validation
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:  # onnx is optional as well
    import onnx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    onnx = None  # type: ignore


@dataclass
class ExportMetadata:
    """Structured metadata describing an ONNX artifact."""

    python_version: str
    platform: str
    torch_version: str | None
    onnx_version: str | None
    opset: int
    precision: str
    dynamic_axes: bool
    input_shape: Tuple[int, int]
    outputs: Dict[str, Tuple[int, int]]
    extra: Dict[str, Any]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding, creating parents when required."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def dump_metadata(path: Path, metadata: ExportMetadata) -> None:
    dump_json(Path(path), asdict(metadata))


def build_export_metadata(*, opset: int, precision: str, dynamic_axes: bool,
                          spectral_length: int, n_classes: int,
                          ckpt_path: Path, onnx_path: Path,
                          simplify: bool, verify: bool,
                          trt_profile: Dict[str, Any] | None) -> ExportMetadata:
    """Collect environment information for downstream auditability."""

    python_version = sys.version.split()[0]
    platform_str = platform.platform()
    torch_version = getattr(torch, "__version__", None) if torch else None
    onnx_version = getattr(onnx, "__version__", None) if onnx else None
    outputs = {
        "logits": (1, n_classes),
        "y_per_class": (1, n_classes),
        "y_global": (1, 1),
        "y_soft": (1, 1),
    }
    extra = {
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "onnx_path": str(Path(onnx_path).resolve()),
        "simplify": simplify,
        "verify": verify,
    }
    if trt_profile:
        extra["tensorrt_profile"] = trt_profile
    return ExportMetadata(
        python_version=python_version,
        platform=platform_str,
        torch_version=torch_version,
        onnx_version=onnx_version,
        opset=opset,
        precision=precision,
        dynamic_axes=dynamic_axes,
        input_shape=(1, spectral_length),
        outputs=outputs,
        extra=extra,
    )


def parse_trt_profile(spec: str) -> Tuple[int, int, int]:
    """Parse a "min,opt,max" batch specification string for TensorRT."""

    if not spec:
        raise ValueError("空的 TensorRT profile 描述")
    normalized = spec.replace(":", ",")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"TensorRT profile 必须包含 min/opt/max 三个值：{spec}")
    try:
        values = tuple(int(p) for p in parts)  # type: ignore[assignment]
    except ValueError as exc:  # pragma: no cover - trivial conversion check
        raise ValueError(f"TensorRT profile 仅接受整数：{spec}") from exc
    min_b, opt_b, max_b = values
    if not (min_b > 0 and opt_b > 0 and max_b > 0):
        raise ValueError("TensorRT profile 的 batch 大小必须大于 0")
    if not (min_b <= opt_b <= max_b):
        raise ValueError("需满足 min <= opt <= max")
    return values


def build_trt_profile_dict(batch_min: int, batch_opt: int, batch_max: int,
                            spectral_length: int, precision: str) -> Dict[str, Any]:
    """Return a small JSON-serialisable dict that TensorRT tooling can ingest."""

    return {
        "input": {
            "name": "x",
            "dtype": "float16" if precision == "fp16" else "float32",
            "profiles": [
                {
                    "min": [batch_min, spectral_length],
                    "opt": [batch_opt, spectral_length],
                    "max": [batch_max, spectral_length],
                }
            ],
        }
    }
