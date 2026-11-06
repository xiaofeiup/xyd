"""Bounded-memory CSV feature selection and modeling.

The selector reads one *column batch* at a time, rather than materializing
every feature CSV.  Each batch is evaluated by :class:`FeatureQualitySelector`
so the quality rules (missing ratio, single-value ratio, IV and time-block
PSI) stay identical to the in-memory implementation.

Example
-------
>>> from sklearn.linear_model import LogisticRegression
>>> from model_tools.features.csv_streaming import CSVFeatureModelingPipeline
>>> pipeline = CSVFeatureModelingPipeline(
...     feature_paths=["part_1.csv", "part_2.csv"], label_path="labels.csv",
...     key_col="test_id", target_col="flag", time_col="apply_time",
...     time_blocks=4,
... )
>>> pipeline.fit(LogisticRegression(max_iter=1000), n_jobs=4, batch_size=64,
...              top_k=200)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .selection import FeatureQualitySelector


_WORKER_LABEL_DATA: Optional[pd.DataFrame] = None
_WORKER_CONFIG: Optional[Dict[str, Any]] = None
_CHECKPOINT_SCHEMA_VERSION = 1


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield ordered, non-empty batches of ``size`` feature names."""
    for start in range(0, len(values), size):
        yield list(values[start:start + size])


def _validate_unique_keys(data: pd.DataFrame, key_col: str, source: str) -> None:
    if key_col not in data.columns:
        raise ValueError("{} 缺少主键列 {!r}".format(source, key_col))
    duplicate_count = int(data[key_col].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            "{} 的主键列 {!r} 存在 {} 条重复记录，无法安全按主键对齐".format(
                source, key_col, duplicate_count
            )
        )


def build_feature_index(
    feature_paths: Sequence[str],
    key_col: str,
    exclude_cols: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """Return the shared feature schema for row-sharded feature CSV files.

    Only CSV headers are read.  Every input CSV must expose exactly the same
    candidate feature columns and each feature maps to every source file.  Row
    uniqueness is checked when a requested column batch is read.
    """
    if not feature_paths:
        raise ValueError("feature_paths 不能为空")

    excluded = set(exclude_cols or [])
    excluded.add(key_col)
    feature_to_file: Dict[str, List[str]] = OrderedDict()
    expected_features: Optional[List[str]] = None
    for raw_path in feature_paths:
        path = str(raw_path)
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if key_col not in columns:
            raise ValueError("特征 CSV {} 缺少主键列 {!r}".format(path, key_col))
        features = [feature for feature in columns if feature not in excluded]
        if expected_features is None:
            expected_features = features
            feature_to_file = OrderedDict(
                (feature, [str(item) for item in feature_paths])
                for feature in expected_features
            )
        elif features != expected_features:
            missing = [feature for feature in expected_features if feature not in features]
            extra = [feature for feature in features if feature not in expected_features]
            raise ValueError(
                "特征 CSV {} 与第一个文件的特征表头不一致；缺少: {}，额外: {}".format(
                    path, missing, extra
                )
            )

    if not feature_to_file:
        raise ValueError("特征 CSV 中没有可筛选的特征列")
    return feature_to_file


def _load_label_data(
    label_path: str,
    key_col: str,
    target_col: str,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    columns = [key_col, target_col]
    if time_col is not None and time_col not in columns:
        columns.append(time_col)
    try:
        labels = pd.read_csv(label_path, usecols=columns)
    except ValueError as exc:
        raise ValueError(
            "标签 CSV {} 必须包含 {}".format(label_path, columns)
        ) from exc
    _validate_unique_keys(labels, key_col, "标签 CSV {}".format(label_path))
    missing_target_count = int(labels[target_col].isna().sum())
    if missing_target_count:
        raise ValueError(
            "标签列 {!r} 存在缺失值（{} 条），请在建模前处理".format(
                target_col, missing_target_count
            )
        )
    if labels[target_col].dropna().nunique() != 2:
        raise ValueError("标签列 {!r} 必须包含两个非空类别".format(target_col))
    return labels


def _file_fingerprint(path: str) -> Dict[str, Any]:
    """Return a cheap, stable-enough identity for a local CSV input file."""
    resolved_path = Path(path).resolve()
    stat = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _make_quality_selector(config: Dict[str, Any]) -> FeatureQualitySelector:
    return FeatureQualitySelector(
        missing_threshold=config["missing_threshold"],
        single_value_threshold=config["single_value_threshold"],
        iv_threshold=config["iv_threshold"],
        iv_bins=config["iv_bins"],
        time_col=config["time_col"],
        time_blocks=config["time_blocks"],
        psi_threshold=config["psi_threshold"],
        psi_bins=config["psi_bins"],
        psi_min_sample=config["psi_min_sample"],
        positive_label=config["positive_label"],
        exclude_cols=[config["key_col"], config["target_col"]],
        invalid_psi_policy=config["invalid_psi_policy"],
        calculate_metrics_for_rejected=config["calculate_metrics_for_rejected"],
    )


def _evaluate_feature_batch(
    feature_paths: Sequence[str],
    features: Sequence[str],
    label_data: pd.DataFrame,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Read one column batch from every row shard and evaluate it."""
    key_col = config["key_col"]
    shards = [
        pd.read_csv(csv_path, usecols=[key_col] + list(features))
        for csv_path in feature_paths
    ]
    feature_data = pd.concat(shards, ignore_index=True, copy=False)
    _validate_unique_keys(feature_data, key_col, "所有特征 CSV")
    merged = label_data.merge(
        feature_data, on=key_col, how="left", sort=False, validate="one_to_one"
    )
    selector = _make_quality_selector(config)
    selector.fit(merged, merged[config["target_col"]])
    return selector.get_selection_log().to_dict("records")


def _initialize_selection_worker(label_data: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Copy label data once into each process instead of once per task."""
    global _WORKER_LABEL_DATA, _WORKER_CONFIG
    _WORKER_LABEL_DATA = label_data
    _WORKER_CONFIG = config


def _evaluate_feature_batch_in_worker(task: List[str]) -> List[Dict[str, Any]]:
    if _WORKER_LABEL_DATA is None or _WORKER_CONFIG is None:  # pragma: no cover
        raise RuntimeError("特征筛选 worker 未初始化")
    return _evaluate_feature_batch(
        _WORKER_CONFIG["feature_paths"], task, _WORKER_LABEL_DATA, _WORKER_CONFIG
    )


def _keep_mask(log: pd.DataFrame) -> pd.Series:
    """Normalize checkpoint booleans that pandas may load as strings."""
    return log["keep"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )


class CSVFeatureQualitySelector:
    """Select wide CSV features in bounded column batches.

    ``time_col`` must be present in ``label_path``.  The label data is small
    enough to stay in memory and gives every feature batch the same target and
    PSI time blocks.
    """

    def __init__(
        self,
        feature_paths: Sequence[str],
        label_path: str,
        key_col: str = "test_id",
        target_col: str = "flag",
        time_col: Optional[str] = None,
        time_blocks: Optional[int] = None,
        exclude_cols: Optional[Sequence[str]] = None,
        missing_threshold: float = 0.95,
        single_value_threshold: float = 0.95,
        iv_threshold: float = 0.02,
        iv_bins: int = 10,
        psi_threshold: float = 0.25,
        psi_bins: int = 10,
        psi_min_sample: int = 10,
        positive_label: Any = 1,
        invalid_psi_policy: str = "drop",
        calculate_metrics_for_rejected: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if not time_col or time_blocks is None:
            raise ValueError("流式 PSI 筛选必须同时指定 time_col 和 time_blocks")
        self.feature_paths = [str(path) for path in feature_paths]
        self.label_path = str(label_path)
        self.key_col = key_col
        self.target_col = target_col
        self.time_col = time_col
        self.exclude_cols = list(exclude_cols or [])
        self.checkpoint_path = checkpoint_path
        self._quality_config = {
            "key_col": key_col,
            "target_col": target_col,
            "time_col": time_col,
            "time_blocks": time_blocks,
            "missing_threshold": missing_threshold,
            "single_value_threshold": single_value_threshold,
            "iv_threshold": iv_threshold,
            "iv_bins": iv_bins,
            "psi_threshold": psi_threshold,
            "psi_bins": psi_bins,
            "psi_min_sample": psi_min_sample,
            "positive_label": positive_label,
            "invalid_psi_policy": invalid_psi_policy,
            "calculate_metrics_for_rejected": calculate_metrics_for_rejected,
        }
        # Validate all numerical thresholds before potentially scanning a large file set.
        _make_quality_selector(self._quality_config)
        self.feature_to_file: Dict[str, List[str]] = OrderedDict()
        self.results_: Optional[pd.DataFrame] = None

    @property
    def _checkpoint_metadata_path(self) -> Optional[Path]:
        if not self.checkpoint_path:
            return None
        return Path(str(self.checkpoint_path) + ".meta.json")

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "quality_config": self._quality_config,
            "exclude_cols": sorted(self.exclude_cols),
            "feature_to_file": [
                [feature, [str(Path(path).resolve()) for path in csv_paths]]
                for feature, csv_paths in self.feature_to_file.items()
            ],
            "feature_files": [_file_fingerprint(path) for path in self.feature_paths],
            "label_file": _file_fingerprint(self.label_path),
        }

    def _write_checkpoint_metadata(self, metadata: Dict[str, Any]) -> None:
        metadata_path = self._checkpoint_metadata_path
        if metadata_path is None:
            return
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary_path.replace(metadata_path)

    def _load_checkpoint(
        self, expected_metadata: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], set]:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return [], set()
        metadata_path = self._checkpoint_metadata_path
        if metadata_path is None or not metadata_path.exists():
            raise ValueError(
                "检查点缺少元数据，无法确认其是否适用于当前数据或筛选参数；"
                "请删除检查点后重新运行"
            )
        try:
            actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("检查点元数据无法读取，请删除检查点后重新运行") from exc
        if actual_metadata != expected_metadata:
            raise ValueError(
                "检查点与当前数据或筛选参数不一致；请使用新的 checkpoint_path 或删除旧检查点"
            )
        checkpoint = pd.read_csv(self.checkpoint_path)
        if "feature" not in checkpoint.columns:
            raise ValueError("检查点 {} 缺少 feature 列".format(self.checkpoint_path))
        checkpoint = checkpoint[checkpoint["feature"].isin(self.feature_to_file)]
        return checkpoint.to_dict("records"), set(checkpoint["feature"].dropna())

    def _append_checkpoint(self, rows: List[Dict[str, Any]]) -> None:
        if not self.checkpoint_path or not rows:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )

    def run(self, n_jobs: int = 1, batch_size: int = 128) -> pd.DataFrame:
        """Evaluate all uncheckpointed feature batches and return their log."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 必须是大于等于 1 的整数")
        if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs 必须是正整数或 -1")
        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs

        excluded = set(self.exclude_cols + [self.target_col, self.time_col])
        self.feature_to_file = build_feature_index(
            self.feature_paths, self.key_col, exclude_cols=excluded
        )
        label_data = _load_label_data(
            self.label_path, self.key_col, self.target_col, self.time_col
        )
        metadata = self._checkpoint_metadata()
        existing_rows, completed = self._load_checkpoint(metadata)
        if self.checkpoint_path and not existing_rows:
            self._write_checkpoint_metadata(metadata)
        tasks = list(_chunked(
            [feature for feature in self.feature_to_file if feature not in completed],
            batch_size,
        ))

        results = list(existing_rows)
        if workers == 1:
            for features in tasks:
                rows = _evaluate_feature_batch(
                    self.feature_paths, features, label_data, self._quality_config
                )
                results.extend(rows)
                self._append_checkpoint(rows)
        elif tasks:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_selection_worker,
                initargs=(label_data, dict(self._quality_config, feature_paths=self.feature_paths)),
            ) as executor:
                futures = {
                    executor.submit(_evaluate_feature_batch_in_worker, task): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    rows = future.result()
                    results.extend(rows)
                    self._append_checkpoint(rows)

        self.results_ = pd.DataFrame(results)
        if self.results_.empty:
            self.results_ = pd.DataFrame(columns=["feature", "iv", "keep"])
        else:
            self.results_ = (
                self.results_
                .drop_duplicates(subset=["feature"], keep="last")
                .sort_values("iv", ascending=False, na_position="last")
                .reset_index(drop=True)
            )
        return self.results_.copy()

    @property
    def selected_features(self) -> List[str]:
        """Features kept by the latest run, in descending IV order."""
        if self.results_ is None:
            raise RuntimeError("请先调用 run()")
        return self.results_.loc[_keep_mask(self.results_), "feature"].tolist()


def load_selected_feature_matrix(
    feature_paths: Sequence[str],
    label_path: str,
    selected_features: Sequence[str],
    key_col: str = "test_id",
    target_col: str = "flag",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load only selected feature columns and align them to label CSV order."""
    selected = list(selected_features)
    if not selected:
        raise ValueError("selected_features 不能为空")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_features 中存在重复特征名")

    feature_to_file = build_feature_index(
        feature_paths, key_col, exclude_cols=[target_col]
    )
    missing = [feature for feature in selected if feature not in feature_to_file]
    if missing:
        raise ValueError("入选特征不在特征 CSV 中: {}".format(missing))
    labels = _load_label_data(label_path, key_col, target_col)
    matrix = labels[[key_col]].copy()
    shards = [
        pd.read_csv(csv_path, usecols=[key_col] + selected)
        for csv_path in feature_paths
    ]
    feature_data = pd.concat(shards, ignore_index=True, copy=False)
    _validate_unique_keys(feature_data, key_col, "所有特征 CSV")
    matrix = matrix.merge(
        feature_data, on=key_col, how="left", sort=False, validate="one_to_one"
    )
    return matrix.loc[:, selected].copy(), labels[target_col].copy()


class CSVFeatureModelingPipeline:
    """Select features from wide CSV files, then fit a sklearn-compatible estimator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.selector_ = CSVFeatureQualitySelector(*args, **kwargs)
        self.model_: Optional[Any] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(
        self,
        estimator: Any,
        n_jobs: int = 1,
        batch_size: int = 128,
        top_k: Optional[int] = None,
    ) -> "CSVFeatureModelingPipeline":
        """Run selection, load only selected columns, and fit ``estimator``."""
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k 必须是大于等于 1 的整数或 None")
        self.selector_.run(n_jobs=n_jobs, batch_size=batch_size)
        selected = self.selector_.selected_features
        if top_k is not None:
            selected = selected[:top_k]
        if not selected:
            raise ValueError("特征筛选后没有可用于建模的特征")
        X, y = load_selected_feature_matrix(
            self.selector_.feature_paths,
            self.selector_.label_path,
            selected,
            key_col=self.selector_.key_col,
            target_col=self.selector_.target_col,
        )
        estimator.fit(X, y)
        self.model_ = estimator
        self.selected_features_ = selected
        return self


__all__ = [
    "CSVFeatureQualitySelector",
    "CSVFeatureModelingPipeline",
    "build_feature_index",
    "load_selected_feature_matrix",
]
