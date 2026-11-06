"""
内存分析工具。

提供类似 notebook 中 ``whos`` 的变量内存占用查看能力。
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


def _get_shape_text(value: Any) -> str:
    """返回对象 shape 的可读文本。"""
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    try:
        if isinstance(shape, tuple):
            return str(shape)
        return str(tuple(shape))
    except Exception:
        return "-"


def _get_memory_mb(value: Any) -> float:
    """
    估算对象内存占用（MB）。

    对 pandas / numpy 使用更准确统计，其余对象回退为 sys.getsizeof。
    """
    try:
        if isinstance(value, pd.DataFrame):
            bytes_size = int(value.memory_usage(index=True, deep=True).sum())
        elif isinstance(value, pd.Series):
            bytes_size = int(value.memory_usage(index=True, deep=True))
        elif isinstance(value, np.ndarray):
            bytes_size = int(value.nbytes)
        else:
            bytes_size = int(sys.getsizeof(value))
    except Exception:
        bytes_size = 0
    return bytes_size / (1024 ** 2)


def _iter_user_variables(scope: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """过滤出用户变量，排除内置与模块对象。"""
    for name, value in scope.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        # 避免把模块本身列进结果，降低噪音
        if getattr(value, "__class__", None).__name__ == "module":
            continue
        yield name, value


def whos(scope: Mapping[str, Any], top: int = 10) -> pd.DataFrame:
    """
    查看当前变量的内存占用并打印 TopN 结果。

    Parameters
    ----------
    scope : Mapping[str, Any]
        变量作用域，通常传 ``globals()`` 或 ``locals()``。
    top : int, default=10
        展示占用内存最大的前 N 个变量。

    Returns
    -------
    pd.DataFrame
        包含变量名、类型、shape、内存占用（MB）的明细表。
    """
    if top <= 0:
        top = 10

    rows: List[Dict[str, Any]] = []
    for name, value in _iter_user_variables(scope):
        rows.append(
            {
                "Variable": name,
                "Type": type(value).__name__,
                "Shape": _get_shape_text(value),
                "Memory(MB)": _get_memory_mb(value),
            }
        )

    if not rows:
        result = pd.DataFrame(columns=["Variable", "Type", "Shape", "Memory(MB)"])
        print("没有可展示的变量。")
        return result

    result = (
        pd.DataFrame(rows)
        .sort_values("Memory(MB)", ascending=False)
        .head(int(top))
        .reset_index(drop=True)
    )
    result["Memory(MB)"] = result["Memory(MB)"].round(2)

    print(result.to_string(index=True))
    return result


__all__: List[str] = ["whos"]
