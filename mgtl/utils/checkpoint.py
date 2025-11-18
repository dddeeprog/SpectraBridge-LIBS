"""Utilities for safely loading checkpoints and surfacing incomplete restores."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def load_state_strict(model, state: Dict[str, Any], *, allow_missing: bool = False,
                      logger=None, name: str = "model") -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load a state dict and optionally fail fast when keys do not match.

    Returns the (missing, unexpected) tuples so callers can surface additional
    context if desired.
    """

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        msg = (f"{name} state_dict 不匹配：missing={len(missing)} unexpected={len(unexpected)}\n"
               f"missing keys: {missing}\nunexpected keys: {unexpected}")
        if allow_missing:
            if logger is not None:
                logger.warning(msg)
        else:
            raise RuntimeError(msg)
    else:
        if logger is not None:
            logger.info(f"{name} state_dict 加载完成（无缺失键）")
    return missing, unexpected
