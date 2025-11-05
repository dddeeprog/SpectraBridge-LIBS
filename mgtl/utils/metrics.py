#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py：统一的回归/分类指标实现（无外部依赖版本）

提供两个度量器：
- RegressionMetrics：MAPE / RMSE / MAE / R²（支持逐批累计）；
- ClassificationMetrics：Accuracy / Precision / Recall / F1（macro / weighted），带混淆矩阵累计。

使用方式（与 train.py / eval.py / infer.py 一致）：

    reg = RegressionMetrics()
    meter = reg.new_meter()
    reg.update_meter(meter, y_pred_cpu_tensor, y_true_cpu_tensor)  # 可多次调用
    res = reg.compute(meter, prefix="reg/")

    cls = ClassificationMetrics(n_classes=6)
    meter2 = cls.new_meter()
    cls.update_meter(meter2, logits_cpu_tensor, y_true_cpu_tensor)
    res2 = cls.compute(meter2, prefix="cls/")

说明：
- 所有输入张量均为 **CPU**（外层已 .cpu()），内部自动转 numpy；
- RegressionMetrics 的 macro/weighted 在未提供类别时等价（== overall）；
- 分类的 weighted 权重为各类别支持度（样本数）。
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch


# ============================= 回归指标 ============================= #
class RegressionMetrics:
    """回归任务指标累计与计算。
    累计信息：样本数 n、SSE、|err| 累计、MAPE 分子/分母、y 的一阶/二阶矩用于 R²。
    """

    def new_meter(self) -> Dict[str, float]:
        return {
            "n": 0.0,               # 样本总数
            "sse": 0.0,            # ∑(y - ŷ)^2
            "sae": 0.0,            # ∑|y - ŷ|
            "mape_num": 0.0,       # ∑|y - ŷ|
            "mape_den": 0.0,       # ∑max(|y|, eps)
            "sum_y": 0.0,          # ∑y
            "sum_y2": 0.0,         # ∑y^2
        }

    @staticmethod
    def _to_np(a: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def update_meter(self, meter: Dict[str, float], y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray,
                     eps: float = 1e-8) -> None:
        """逐批更新。
        参数：
        - y_pred: [B]  预测
        - y_true: [B]  真值
        - eps   : MAPE 稳定项
        """
        yp = self._to_np(y_pred).astype(np.float64).reshape(-1)
        yt = self._to_np(y_true).astype(np.float64).reshape(-1)
        assert yp.shape == yt.shape

        n = float(yp.shape[0])
        err = yp - yt
        sae = np.abs(err).sum()
        sse = (err ** 2).sum()
        mape_num = np.abs(err).sum()
        mape_den = np.maximum(np.abs(yt), eps).sum()
        sum_y = yt.sum()
        sum_y2 = (yt ** 2).sum()

        meter["n"] += n
        meter["sae"] += float(sae)
        meter["sse"] += float(sse)
        meter["mape_num"] += float(mape_num)
        meter["mape_den"] += float(mape_den)
        meter["sum_y"] += float(sum_y)
        meter["sum_y2"] += float(sum_y2)

    def compute(self, meter: Dict[str, float], prefix: str = "") -> Dict[str, float]:
        n = max(meter["n"], 1.0)
        rmse = (meter["sse"] / n) ** 0.5
        mae = meter["sae"] / n
        mape = (meter["mape_num"] / max(meter["mape_den"], 1e-8)) * 100.0  # 百分比形式
        # R² = 1 - SSE / SST，SST = ∑(y - ȳ)^2 = ∑y^2 - n*ȳ^2
        y_mean = meter["sum_y"] / n
        sst = meter["sum_y2"] - n * (y_mean ** 2)
        r2 = 1.0 - (meter["sse"] / max(sst, 1e-12)) if sst > 1e-12 else 0.0

        return {
            f"{prefix}rmse_macro": float(rmse),
            f"{prefix}mae_macro": float(mae),
            f"{prefix}mape_macro": float(mape),
            f"{prefix}r2": float(r2),
        }


# ============================= 分类指标 ============================= #
class ClassificationMetrics:
    """多类分类指标累计与计算（macro/weighted）。"""
    def __init__(self, n_classes: int):
        self.C = int(n_classes)

    def new_meter(self) -> Dict[str, np.ndarray]:
        return {
            "cm": np.zeros((self.C, self.C), dtype=np.int64),  # 混淆矩阵：行=true，列=pred
        }

    @staticmethod
    def _to_np(a: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def update_meter(self, meter: Dict[str, np.ndarray], logits: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        """逐批更新混淆矩阵。
        - logits: [B, C]
        - target: [B]
        """
        if isinstance(logits, torch.Tensor):
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        else:
            pred = np.argmax(np.asarray(logits), axis=1)
        y = self._to_np(target).astype(np.int64).reshape(-1)
        assert pred.shape == y.shape

        # 累计到混淆矩阵
        for t, p in zip(y, pred):
            if 0 <= t < self.C and 0 <= p < self.C:
                meter["cm"][t, p] += 1

    @staticmethod
    def _prf_from_cm(cm: np.ndarray) -> Dict[str, float]:
        """从混淆矩阵计算 acc、precision/recall/f1 的 macro / weighted。"""
        C = cm.shape[0]
        support = cm.sum(axis=1)  # 每类样本数
        pred_sum = cm.sum(axis=0) # 每类预测为该类的数量
        tp = np.diag(cm).astype(np.float64)
        total = cm.sum()

        acc = float(tp.sum() / max(total, 1))

        # 按类计算 P/R/F1，避免除零
        precision_c = np.divide(tp, np.maximum(pred_sum, 1), where=pred_sum>0)
        recall_c    = np.divide(tp, np.maximum(support, 1), where=support>0)
        f1_c = np.zeros_like(precision_c)
        denom = precision_c + recall_c
        nonzero = denom > 0
        f1_c[nonzero] = 2 * precision_c[nonzero] * recall_c[nonzero] / denom[nonzero]

        # macro：各类简单平均
        macro = lambda v: float(np.mean(v)) if C > 0 else 0.0
        # weighted：按支持度加权
        w = support.astype(np.float64)
        w = w / max(w.sum(), 1.0)
        weighted = lambda v: float((v * w).sum()) if C > 0 else 0.0

        return {
            "acc": acc,
            "precision_macro": macro(precision_c),
            "recall_macro": macro(recall_c),
            "f1_macro": macro(f1_c),
            "precision_weighted": weighted(precision_c),
            "recall_weighted": weighted(recall_c),
            "f1_weighted": weighted(f1_c),
        }

    def compute(self, meter: Dict[str, np.ndarray], prefix: str = "") -> Dict[str, float]:
        cm = meter["cm"].astype(np.int64)
        prf = self._prf_from_cm(cm)
        return {
            f"{prefix}acc": prf["acc"],
            f"{prefix}precision_macro": prf["precision_macro"],
            f"{prefix}recall_macro": prf["recall_macro"],
            f"{prefix}f1_macro": prf["f1_macro"],
            f"{prefix}precision_weighted": prf["precision_weighted"],
            f"{prefix}recall_weighted": prf["recall_weighted"],
            f"{prefix}f1_weighted": prf["f1_weighted"],
        }
