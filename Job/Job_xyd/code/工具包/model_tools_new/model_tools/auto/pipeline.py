"""Configuration-driven binary credit-risk modeling pipeline.

The module deliberately keeps all state for one run below a timestamped
directory, so a rendered report can always be traced to its data contract,
features, preprocessing, model, predictions, and metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.delivery_report import ModelDeliveryReport
from ..features.data_analysis_optimized import DataAnalyzer
from ..features.selection import FeatureSelector


@dataclass
class DataConfig:
    path: Optional[str] = None
    feature_path: Optional[str] = None
    feature_paths: List[str] = field(default_factory=list)
    label_path: Optional[str] = None
    key_col: Optional[str] = None
    target_col: str = "target"
    sample_type_col: str = "sample_type"
    train_label: str = "train"
    oos_label: str = "oos"
    date_col: Optional[str] = None
    exclude_cols: List[str] = field(default_factory=list)


@dataclass
class FeatureSelectionConfig:
    mode: str = "in_memory"
    missing_threshold: float = 0.95
    single_value_threshold: float = 0.95
    iv_threshold: float = 0.02
    iv_bins: int = 10


@dataclass
class ModelConfig:
    type: str = "lightgbm"


@dataclass
class TuningConfig:
    enabled: bool = True
    n_trials: int = 20
    cv_folds: int = 3
    max_auc_gap: float = 0.05
    max_ks_gap: float = 0.03
    n_jobs: int = 1


@dataclass
class OutputConfig:
    output_dir: str = "./output"
    run_name: Optional[str] = None


@dataclass
class AutoModelingConfig:
    data: DataConfig
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AutoModelingConfig":
        """Build a typed configuration from the documented nested YAML shape."""
        return cls(
            data=DataConfig(**raw.get("data", {})),
            feature_selection=FeatureSelectionConfig(**raw.get("feature_selection", {})),
            model=ModelConfig(**raw.get("model", {})),
            tuning=TuningConfig(**raw.get("tuning", {})),
            output=OutputConfig(**raw.get("output", {})),
        )


@dataclass
class FeatureSelectionResult:
    decision_log: pd.DataFrame
    selected_features: List[str]
    summary: Dict[str, int]


@dataclass
class PipelineRunResult:
    run_dir: Path
    selected_features: List[str]
    metrics: Dict[str, Dict[str, float]]
    html_report_path: Path
    excel_report_path: Path
    warnings: List[str]
    model_path: Path
    preprocessor_path: Path


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"只支持 CSV 或 Parquet，收到: {path}")


def _ks(y_true: pd.Series, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def _svg_curve(y_true: pd.Series, score: np.ndarray, title: str) -> str:
    """Render a small self-contained ROC/KS chart for offline HTML."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    points = []
    for x, value in zip(fpr[:: max(1, len(fpr) // 60)], tpr[:: max(1, len(tpr) // 60)]):
        points.append(f"{40 + x * 300:.1f},{220 - value * 170:.1f}")
    polyline = " ".join(points) or "40,220 340,50"
    return f'<div class="chart"><strong>{title}</strong><svg viewBox="0 0 380 250" role="img" aria-label="{title}"><path d="M40 220H340M40 220V40" stroke="#9bb5cc" fill="none"/><polyline points="{polyline}" fill="none" stroke="#1677c8" stroke-width="3"/><text x="42" y="242">0</text><text x="330" y="242">1</text><text x="8" y="48">1</text></svg></div>'


class AutoModelingPipeline:
    """Run a strictly validated binary-modeling workflow from one config."""

    def __init__(self, config: AutoModelingConfig):
        self.config = config
        self.warnings: List[str] = []

    def _make_run_dir(self) -> Path:
        run_name = self.config.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir = Path(self.config.output.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _load_and_validate_data(self) -> pd.DataFrame:
        data_cfg = self.config.data
        has_split_input = bool(data_cfg.feature_path or data_cfg.feature_paths or data_cfg.label_path)
        if bool(data_cfg.path) == has_split_input:
            raise ValueError("数据配置必须提供单表 path，或同时提供 feature_path 与 label_path")
        if data_cfg.path:
            data = _read_table(data_cfg.path)
        else:
            feature_paths = data_cfg.feature_paths or ([data_cfg.feature_path] if data_cfg.feature_path else [])
            if not feature_paths or not data_cfg.label_path or not data_cfg.key_col:
                raise ValueError("双表输入需要 feature_path、label_path 和 key_col")
            labels = _read_table(data_cfg.label_path)
            if labels[data_cfg.key_col].duplicated().any():
                raise ValueError("标签表主键不能重复，避免多对多合并")
            data = labels
            for feature_path in feature_paths:
                features = _read_table(feature_path)
                if features[data_cfg.key_col].duplicated().any():
                    raise ValueError(f"特征表主键不能重复: {feature_path}")
                data = data.merge(features, on=data_cfg.key_col, how="inner", suffixes=("", "_label"))

        required = [data_cfg.target_col, data_cfg.sample_type_col]
        if data_cfg.key_col:
            required.append(data_cfg.key_col)
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"输入数据缺少关键列: {missing}")
        target_values = set(pd.Series(data[data_cfg.target_col]).dropna().unique())
        if not target_values.issubset({0, 1}) or len(target_values) != 2:
            raise ValueError("目标列必须且只能包含二分类标签 0/1")
        train = data[data[data_cfg.sample_type_col] == data_cfg.train_label]
        oos = data[data[data_cfg.sample_type_col] == data_cfg.oos_label]
        if train.empty:
            raise ValueError("Train 分区为空")
        if oos.empty:
            raise ValueError("OOS 分区为空")
        for name, subset in (("Train", train), ("OOS", oos)):
            if subset[data_cfg.target_col].nunique() != 2:
                raise ValueError(f"{name} 分区必须同时包含标签 0 和 1")
        return data.reset_index(drop=True)

    def _numeric_features(self, data: pd.DataFrame) -> List[str]:
        cfg = self.config.data
        excluded = set(cfg.exclude_cols + [cfg.target_col, cfg.sample_type_col])
        if cfg.key_col:
            excluded.add(cfg.key_col)
        if cfg.date_col:
            excluded.add(cfg.date_col)
        features = [column for column in data.select_dtypes(include=[np.number]).columns if column not in excluded]
        if not features:
            raise ValueError("没有可用于建模的数值特征")
        return features

    def _select_features(self, train: pd.DataFrame, features: List[str], run_dir: Path) -> FeatureSelectionResult:
        cfg = self.config.feature_selection
        if cfg.mode == "streaming":
            data_cfg = self.config.data
            csv_paths = data_cfg.feature_paths or ([data_cfg.feature_path] if data_cfg.feature_path else [])
            if not csv_paths or not data_cfg.label_path or not data_cfg.key_col:
                raise ValueError("streaming 筛选需要双表 CSV 输入、feature_paths/feature_path、label_path 和 key_col")
            if any(Path(path).suffix.lower() != ".csv" for path in csv_paths + [data_cfg.label_path]):
                raise ValueError("streaming 筛选目前仅支持 CSV 输入")
            from ..features.streaming_feature_selection import StreamingFeatureSelector
            selector = StreamingFeatureSelector(
                csv_paths=csv_paths, target_path=data_cfg.label_path, key_col=data_cfg.key_col,
                target_col=data_cfg.target_col, exclude_cols=data_cfg.exclude_cols + [data_cfg.sample_type_col],
                single_value_threshold=cfg.single_value_threshold, missing_threshold=cfg.missing_threshold,
                iv_threshold=cfg.iv_threshold, iv_bins=cfg.iv_bins,
                log_path=str(run_dir / "streaming_feature_selection_checkpoint.csv"),
            )
            raw_log = selector.run(n_jobs=1)
            selected = [feature for feature in selector.selected_features if feature in features]
            if not selected:
                raise ValueError("流式特征筛选后无可用数值特征")
            log = raw_log.rename(columns={"iv": "iv_value", "keep": "keep_feature", "reason": "filter_reason"}).copy()
            log["selected"] = log["feature"].isin(selected)
            return FeatureSelectionResult(log, selected, {"candidates": len(features), "after_missing": int((log["missing_ratio"] < cfg.missing_threshold).sum()), "selected": len(selected)})
        if cfg.mode != "in_memory":
            raise ValueError("feature_selection.mode 仅支持 in_memory 或 streaming")
        missing = train[features].isna().mean()
        viable = missing[missing < cfg.missing_threshold].index.tolist()
        if not viable:
            raise ValueError("缺失率过滤后无可用特征")
        selector = FeatureSelector(method="iv", single_value_threshold=cfg.single_value_threshold, iv_threshold=cfg.iv_threshold)
        selector.fit(train[viable], train[self.config.data.target_col])
        selected = selector.selected_features_ or []
        if not selected:
            raise ValueError("特征筛选后无可用特征")
        log = selector.get_iv_log().copy()
        log["missing_ratio"] = log["feature"].map(missing)
        log["selected"] = log["feature"].isin(selected)
        return FeatureSelectionResult(log, selected, {"candidates": len(features), "after_missing": len(viable), "selected": len(selected)})

    def _model_class(self) -> Tuple[Any, str]:
        model_type = self.config.model.type
        if model_type == "logistic_regression":
            return LogisticRegression, "lr"
        if model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("lightgbm 模型需要安装 model-tools[ml]") from exc
            return LGBMClassifier, "lgb"
        if model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("xgboost 模型需要安装 model-tools[ml]") from exc
            return XGBClassifier, "xgb"
        raise ValueError("model.type 仅支持 lightgbm、xgboost、logistic_regression")

    def _preprocessor(self) -> Pipeline:
        steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.config.model.type == "logistic_regression":
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    def _best_params(self, model_class: Any, model_type: str, X: np.ndarray, y: pd.Series) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        if not self.config.tuning.enabled:
            return ({"max_iter": 1000, "random_state": 42} if model_type == "lr" else {"random_state": 42, "n_jobs": self.config.tuning.n_jobs}), None
        from ..optimization.anti_overfitting_tuner import AntiOverfittingTuner
        tuner = AntiOverfittingTuner(model_class=model_class, model_type=model_type, direction="maximize", max_auc_gap=self.config.tuning.max_auc_gap, max_ks_gap=self.config.tuning.max_ks_gap)
        result = tuner.optimize_anti_overfitting(X, y, n_trials=self.config.tuning.n_trials, cv_folds=self.config.tuning.cv_folds, n_jobs=1, show_progress_bar=False)
        return result["recommended_params"], pd.DataFrame(result["performance_history"])

    def _feature_importance(self, model: Any, features: List[str]) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_)[0])
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({"feature": features, "importance": values}).sort_values("importance", ascending=False)

    def _write_html(self, path: Path, data: pd.DataFrame, selection: FeatureSelectionResult, metrics: Dict[str, Dict[str, float]], importance: pd.DataFrame, excel_name: str) -> None:
        analyzer = DataAnalyzer(data)
        missing = analyzer.missing_summary(selection.selected_features).head(20).to_html(index=False, classes="table")
        distribution = analyzer.distribution_summary(selection.selected_features).head(20).to_html(index=False, classes="table")
        feature_table = selection.decision_log.head(50).to_html(index=False, classes="table")
        importance_html = importance.head(20).to_html(index=False, classes="table")
        metric_rows = "".join(f"<tr><td>{name.upper()}</td><td>{value['auc']:.4f}</td><td>{value['ks']:.4f}</td><td>{value['samples']}</td></tr>" for name, value in metrics.items())
        train_data = data[data[cfg.sample_type_col] == cfg.train_label] if (cfg := self.config.data) else data.iloc[0:0]
        oos_data = data[data[cfg.sample_type_col] == cfg.oos_label]
        train_score = self._last_scores["train"]
        oos_score = self._last_scores["oos"]
        path.write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>自动化模型报告</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;margin:0;background:#f4f7fb;color:#152238}}main{{max-width:1200px;margin:auto;padding:36px}}section{{background:white;border-radius:12px;padding:24px;margin:18px 0;box-shadow:0 2px 10px #dbe3ef}}h1{{color:#0a4d8c}}h2{{border-left:4px solid #1677c8;padding-left:10px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#eaf4ff;border-radius:8px;padding:16px;min-width:180px}}.value{{font-size:28px;font-weight:700}}.table{{border-collapse:collapse;width:100%;font-size:13px}}.table th{{background:#0a4d8c;color:white}}.table th,.table td{{padding:8px;border:1px solid #dce5ef;text-align:left}}.table tr:nth-child(even){{background:#f8fbff}}a{{color:#0a68bf}}.charts{{display:flex;gap:20px;flex-wrap:wrap}}.chart{{border:1px solid #dce5ef;border-radius:8px;padding:12px;flex:1;min-width:320px}}</style></head><body><main><h1>自动化风控建模报告</h1><p>离线交付 · 仅数值特征 · Train 内筛选/调参，OOS 最终评估</p><section><h2>运行摘要</h2><div class="cards"><div class="card">候选特征<div class="value">{selection.summary['candidates']}</div></div><div class="card">最终特征<div class="value">{selection.summary['selected']}</div></div><div class="card">样本数<div class="value">{len(data):,}</div></div></div><p><a href="{excel_name}">下载 Excel 交付明细</a></p></section><section><h2>模型评估</h2><table class="table"><tr><th>分区</th><th>AUC</th><th>KS</th><th>样本数</th></tr>{metric_rows}</table><div class="charts">{_svg_curve(train_data[cfg.target_col], train_score, 'Train ROC')}{_svg_curve(oos_data[cfg.target_col], oos_score, 'OOS ROC')}</div></section><section><h2>数据质量</h2>{missing}<h3>数值分布</h3>{distribution}</section><section><h2>特征筛选</h2>{feature_table}</section><section><h2>特征重要性</h2>{importance_html}</section></main></body></html>''', encoding="utf-8")

    def run(self) -> PipelineRunResult:
        run_dir = self._make_run_dir()
        (run_dir / "config.json").write_text(json.dumps(asdict(self.config), ensure_ascii=False, indent=2), encoding="utf-8")
        data = self._load_and_validate_data()
        cfg = self.config.data
        train = data[data[cfg.sample_type_col] == cfg.train_label].copy()
        oos = data[data[cfg.sample_type_col] == cfg.oos_label].copy()
        selection = self._select_features(train, self._numeric_features(data), run_dir)
        selection.decision_log.to_csv(run_dir / "feature_selection.csv", index=False)
        (run_dir / "selected_features.json").write_text(json.dumps(selection.selected_features, ensure_ascii=False, indent=2), encoding="utf-8")
        preprocessor = self._preprocessor()
        X_train = preprocessor.fit_transform(train[selection.selected_features])
        X_oos = preprocessor.transform(oos[selection.selected_features])
        model_class, model_type = self._model_class()
        params, history = self._best_params(model_class, model_type, X_train, train[cfg.target_col])
        (run_dir / "best_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if history is not None:
            history.drop(columns=["params"], errors="ignore").to_csv(run_dir / "tuning_history.csv", index=False)
        model = model_class(**params)
        model.fit(X_train, train[cfg.target_col])
        train_score = model.predict_proba(X_train)[:, 1]
        oos_score = model.predict_proba(X_oos)[:, 1]
        self._last_scores = {"train": train_score, "oos": oos_score}
        metrics = {"train": {"auc": float(roc_auc_score(train[cfg.target_col], train_score)), "ks": _ks(train[cfg.target_col], train_score), "samples": int(len(train))}, "oos": {"auc": float(roc_auc_score(oos[cfg.target_col], oos_score)), "ks": _ks(oos[cfg.target_col], oos_score), "samples": int(len(oos))}}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(model, run_dir / "model.joblib")
        joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
        predictions = pd.concat([train[[cfg.target_col]].assign(prediction=train_score, partition="train"), oos[[cfg.target_col]].assign(prediction=oos_score, partition="oos")], ignore_index=True)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        importance = self._feature_importance(model, selection.selected_features)
        report_data = pd.concat([train.assign(prediction=train_score, report_partition="train"), oos.assign(prediction=oos_score, report_partition="oos")], ignore_index=True)
        excel_path = run_dir / "model_delivery_report.xlsx"
        report_date_col = cfg.date_col or "report_date"
        if report_date_col not in report_data.columns:
            report_data[report_date_col] = pd.Timestamp.now().normalize()
            self.warnings.append("输入未提供日期列，Excel 报告按运行日期生成时间维度")
        reporter = ModelDeliveryReport(
            report_data,
            target_col=cfg.target_col,
            score_col="prediction",
            date_col=report_date_col,
            sample_type_col="report_partition",
        )
        reporter.generate_full_report(features=selection.selected_features, model=model, save_path=str(excel_path))
        html_path = run_dir / "model_report.html"
        self._write_html(html_path, data, selection, metrics, importance, excel_path.name)
        summary = {"metrics": metrics, "selected_features": selection.selected_features, "warnings": self.warnings, "html_report": str(html_path), "excel_report": str(excel_path)}
        (run_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return PipelineRunResult(run_dir, selection.selected_features, metrics, html_path, excel_path, self.warnings, run_dir / "model.joblib", run_dir / "preprocessor.joblib")


def run_auto_modeling(config_path: str) -> PipelineRunResult:
    """Load YAML configuration and run the public pipeline."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as file:
        return AutoModelingPipeline(AutoModelingConfig.from_dict(yaml.safe_load(file) or {})).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="自动化二分类风控建模")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    result = run_auto_modeling(parser.parse_args().config)
    print(f"HTML 报告: {result.html_report_path}")


if __name__ == "__main__":
    main()
