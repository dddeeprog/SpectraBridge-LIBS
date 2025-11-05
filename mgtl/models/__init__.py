#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mgtl.models：模型子包导出

统一导出本子包下的核心模块，便于：

    from mgtl.models import (
        SpecEncoder,
        ClassifierHead,
        PerClassRegressors,
        GlobalRegressor,
    )
"""
from __future__ import annotations

__all__ = [
    "SpecEncoder",
    "ClassifierHead",
    "PerClassRegressors",
    "GlobalRegressor",
]

from .spec_encoder import SpecEncoder
from .classifier import ClassifierHead
from .regressors import PerClassRegressors, GlobalRegressor
