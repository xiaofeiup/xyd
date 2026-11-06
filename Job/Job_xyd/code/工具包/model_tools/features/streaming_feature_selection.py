"""Stable import path for the legacy CSV-by-feature streaming selector.

The original file name contains a dash and a local-copy suffix, so Python
cannot import it reliably.  This compatibility module intentionally exposes
the supported public API from that implementation without changing its
checkpoint CSV format.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LEGACY_PATH = Path(__file__).with_name("streaming_feature_selection-1_副本2.py")
_SPEC = importlib.util.spec_from_file_location("model_tools.features._streaming_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - installation failure
    raise ImportError(f"无法加载流式特征筛选实现: {_LEGACY_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

StreamingFeatureSelector = _MODULE.StreamingFeatureSelector
build_feature_index = _MODULE.build_feature_index
evaluate_single_feature = _MODULE.evaluate_single_feature
load_target = _MODULE.load_target

__all__ = ["StreamingFeatureSelector", "build_feature_index", "evaluate_single_feature", "load_target"]
