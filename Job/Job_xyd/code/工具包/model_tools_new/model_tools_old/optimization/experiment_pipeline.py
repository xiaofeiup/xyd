"""
参数-评估一体化实验流水线

将参数组合、模型训练、OOT评估、交付报告和台账记录打通。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ..evaluation.delivery_report import ModelDeliveryReport


def _safe_float(value: Any, default: float = 0.0) -> float:
    """将值安全转换为float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_ks(y_true: pd.Series, y_score: pd.Series) -> float:
    """计算KS值。"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float((tpr - fpr).max())


def _normalize_candidate(candidate: Dict[str, Any], fallback_trial: int) -> Dict[str, Any]:
    """
    归一化参数候选记录。

    支持两种输入：
    1. 纯参数字典（如 {'max_depth': 6, ...}）
    2. performance_history风格（包含 params / trial / val_score 等）
    """
    if "params" in candidate and isinstance(candidate["params"], dict):
        params = candidate["params"]
        trial = candidate.get("trial", fallback_trial)
        train_metric = candidate.get("train_score")
        val_metric = candidate.get("val_score")
    else:
        params = candidate
        trial = candidate.get("trial", fallback_trial)
        train_metric = candidate.get("train_score")
        val_metric = candidate.get("val_score")

    return {
        "trial": int(trial),
        "params": params,
        "train_metric": train_metric,
        "val_metric": val_metric,
    }


def _serialize_params(params: Dict[str, Any]) -> str:
    """稳定序列化参数，用于哈希与落盘。"""
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _build_experiment_id(
    study_name: str,
    trial: int,
    params: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    生成唯一实验ID。

    格式：{study_name}-t{trial}-{timestamp}-{params_hash8}
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    params_json = _serialize_params(params)
    params_hash = hashlib.md5(params_json.encode("utf-8")).hexdigest()
    experiment_id = f"{study_name}-t{trial}-{ts}-{params_hash[:8]}"
    return experiment_id, params_hash, params_json


def _append_experiment_ledger(ledger_path: Union[str, Path], row: Dict[str, Any]) -> None:
    """将单条实验记录追加到CSV台账。"""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])

    if ledger_path.exists():
        old_df = pd.read_csv(ledger_path)
        pd.concat([old_df, new_df], ignore_index=True).to_csv(ledger_path, index=False)
    else:
        new_df.to_csv(ledger_path, index=False)


def run_experiment_batch(
    model_class: Any,
    train_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    candidate_params: List[Dict[str, Any]],
    report_dir: Union[str, Path],
    ledger_path: Union[str, Path],
    study_name: str = "anti_overfitting",
    score_col: str = "score",
    sample_type_col: str = "sample_type",
    date_col: str = "date",
    report_features: Optional[List[str]] = None,
    report_feature_names: Optional[Dict[str, str]] = None,
    bin_method: str = "quantile",
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    批量运行参数实验并自动产出Excel+CSV映射。

    Returns:
        实验台账DataFrame（与CSV写入字段一致）
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(feature_cols)
    report_features = report_features if report_features is not None else feature_cols
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows: List[Dict[str, Any]] = []
    for i, raw_candidate in enumerate(candidate_params):
        normalized = _normalize_candidate(raw_candidate, fallback_trial=i)
        trial = normalized["trial"]
        params = normalized["params"]
        train_metric = normalized["train_metric"]
        val_metric = normalized["val_metric"]

        experiment_id, params_hash, params_json = _build_experiment_id(
            study_name=study_name,
            trial=trial,
            params=params,
        )
        report_path = report_dir / f"{experiment_id}.xlsx"

        model = model_class(**params)
        model.fit(train_data[feature_cols], train_data[target_col])

        train_pred = model.predict_proba(train_data[feature_cols])[:, 1]
        oot_pred = model.predict_proba(oot_data[feature_cols])[:, 1]

        train_eval = train_data[[target_col]].copy()
        train_eval[score_col] = train_pred
        if sample_type_col in train_data.columns:
            train_eval[sample_type_col] = "train_" + train_data[sample_type_col].astype(str)
        else:
            train_eval[sample_type_col] = "train"
        if date_col in train_data.columns:
            train_eval[date_col] = train_data[date_col].values
        else:
            train_eval[date_col] = pd.Timestamp.today().normalize()

        oot_eval = oot_data[[target_col]].copy()
        oot_eval[score_col] = oot_pred
        if sample_type_col in oot_data.columns:
            oot_eval[sample_type_col] = "oot_" + oot_data[sample_type_col].astype(str)
        else:
            oot_eval[sample_type_col] = "oot"
        if date_col in oot_data.columns:
            oot_eval[date_col] = oot_data[date_col].values
        else:
            oot_eval[date_col] = pd.Timestamp.today().normalize()

        eval_data = pd.concat([train_eval, oot_eval], ignore_index=True)

        oot_auc = float(roc_auc_score(oot_data[target_col], oot_pred))
        oot_ks = _calculate_ks(oot_data[target_col], pd.Series(oot_pred))

        experiment_meta = {
            "experiment_id": experiment_id,
            "study_name": study_name,
            "trial": trial,
            "params_hash": params_hash,
            "params_json": params_json,
            "created_at": created_at,
            "report_path": str(report_path),
        }

        reporter = ModelDeliveryReport(
            data=eval_data,
            target_col=target_col,
            score_col=score_col,
            date_col=date_col,
            sample_type_col=sample_type_col,
        )
        reporter.generate_full_report(
            features=report_features,
            feature_names=report_feature_names,
            save_path=str(report_path),
            bin_method=bin_method,
            model=model,
            importance_type=importance_type,
            experiment_meta=experiment_meta,
        )

        row = {
            "experiment_id": experiment_id,
            "trial": trial,
            "params_json": params_json,
            "params_hash": params_hash,
            "train_metric": _safe_float(train_metric),
            "val_metric": _safe_float(val_metric),
            "oot_auc": oot_auc,
            "oot_ks": oot_ks,
            "report_path": str(report_path),
            "created_at": created_at,
        }
        _append_experiment_ledger(ledger_path, row)
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "run_experiment_batch",
]
