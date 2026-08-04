"""Small, dependency-light integration tests for the public auto-modeling API."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from model_tools.auto.pipeline import AutoModelingConfig, AutoModelingPipeline
from model_tools.features.streaming_feature_selection import evaluate_single_feature


def _make_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 120
    feature_a = rng.normal(size=rows)
    feature_b = rng.normal(size=rows)
    target = (feature_a + feature_b * 0.4 + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    return pd.DataFrame({
        "id": np.arange(rows), "sample": ["train"] * 80 + ["oos"] * 40,
        "target": target, "feature_a": feature_a, "feature_b": feature_b,
        "category_only": ["x"] * rows,
    })


def _config(data_path: str, output_dir: str) -> AutoModelingConfig:
    return AutoModelingConfig.from_dict({
        "data": {"path": data_path, "key_col": "id", "target_col": "target",
                 "sample_type_col": "sample", "train_label": "train", "oos_label": "oos"},
        "model": {"type": "logistic_regression"},
        "feature_selection": {"mode": "in_memory", "iv_threshold": 0.0,
                              "single_value_threshold": 1.0, "missing_threshold": 1.0},
        "tuning": {"enabled": False, "n_trials": 1, "cv_folds": 2},
        "output": {"output_dir": output_dir},
    })


def test_pipeline_writes_reproducible_artifacts_and_combined_html(tmp_path):
    data_path = tmp_path / "model.csv"
    output_dir = tmp_path / "output"
    _make_data().to_csv(data_path, index=False)

    result = AutoModelingPipeline(_config(str(data_path), str(output_dir))).run()

    assert result.metrics["oos"]["auc"] >= 0.0
    assert result.selected_features == ["feature_a", "feature_b"]
    assert result.html_report_path.exists()
    assert result.excel_report_path.exists()
    assert "数据质量" in result.html_report_path.read_text(encoding="utf-8")
    assert "模型评估" in result.html_report_path.read_text(encoding="utf-8")
    assert json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))["train"]["auc"] >= 0.0
    for name in ("config.json", "selected_features.json", "feature_selection.csv", "model.joblib", "preprocessor.joblib", "pipeline_summary.json"):
        assert (result.run_dir / name).exists()


def test_pipeline_rejects_missing_required_partition(tmp_path):
    data = _make_data().query("sample == 'train'")
    path = tmp_path / "train_only.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="OOS"):
        AutoModelingPipeline(_config(str(path), str(tmp_path / "output"))).run()


def test_pipeline_supports_feature_and_label_tables(tmp_path):
    data = _make_data()
    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    data[["id", "feature_a", "feature_b", "category_only"]].to_csv(feature_path, index=False)
    data[["id", "sample", "target"]].to_csv(label_path, index=False)
    config = AutoModelingConfig.from_dict({
        "data": {"feature_path": str(feature_path), "label_path": str(label_path), "key_col": "id",
                 "target_col": "target", "sample_type_col": "sample", "train_label": "train", "oos_label": "oos"},
        "model": {"type": "logistic_regression"},
        "feature_selection": {"iv_threshold": 0.0, "single_value_threshold": 1.0, "missing_threshold": 1.0},
        "tuning": {"enabled": False}, "output": {"output_dir": str(tmp_path / "output")},
    })

    assert AutoModelingPipeline(config).run().selected_features == ["feature_a", "feature_b"]


def test_streaming_selector_uses_same_inclusive_single_value_cutoff():
    result = evaluate_single_feature(
        "feature", pd.Series(["a"] * 95 + ["b"] * 5), pd.Series([0, 1] * 50),
        single_value_threshold=0.95,
    )

    assert result["keep"] is False
    assert "单一值占比" in result["reason"]
