#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logging.py：轻量日志与进度工具

- get_logger(name, level='INFO', logfile=None)：返回标准 logging.Logger；
- Progress(total=None)：简单进度器（ETA/速度），适合 epoch 内手工打印；
- 便捷函数：human_time, human_num。

说明：
- 不额外依赖第三方库，默认仅输出到控制台；若提供 logfile，将同时写入文件。
- 与 train.py 的直接耦合很弱，可按需在训练循环中插入打印。
"""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
from typing import Optional


# ============================= Logger ============================= #

def get_logger(name: str = "mgtl",
               level: str | int = "INFO",
               logfile: Optional[str | Path] = None) -> logging.Logger:
    """获取（或创建）一个带控制台/文件输出的 Logger。"""
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        # 控制台
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
        # 文件
        if logfile is not None:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(logfile), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)
        logger.propagate = False
    return logger


# ============================= Progress ============================= #
class Progress:
    """简单进度器：打印步速/ETA。示例：

        p = Progress(total=len(loader))
        for i, batch in enumerate(loader, 1):
            ...
            p.step()

    """
    def __init__(self, total: Optional[int] = None, tick_every: float = 1.0, prefix: str = ""):
        self.total = total
        self.tick_every = float(tick_every)
        self.prefix = prefix
        self.start = time.time()
        self.last = self.start
        self.n = 0

    def step(self, k: int = 1) -> None:
        self.n += int(k)
        now = time.time()
        if now - self.last >= self.tick_every:
            rate = self.n / max(now - self.start, 1e-6)
            if self.total:
                remain = max(self.total - self.n, 0)
                eta = remain / max(rate, 1e-6)
                msg = f"{self.prefix} {self.n}/{self.total} | {rate:,.1f}/s | ETA {human_time(eta)}"
            else:
                msg = f"{self.prefix} {self.n} | {rate:,.1f}/s"
            print(msg)
            self.last = now


# ============================= Utils ============================= #

def human_time(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def human_num(x: float) -> str:
    x = float(x)
    for unit in ["", "K", "M", "G"]:
        if abs(x) < 1000.0:
            return f"{x:.1f}{unit}"
        x /= 1000.0
    return f"{x:.1f}T"
