"""示例：使用 ModelDeliveryReportGenerator 生成模型交付报告。

该脚本构造多份合成数据，覆盖交付报告的所有章节，并调用
``ModelDeliveryReportGenerator`` 产生最终结构化结果，方便人工检查或导出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from model_tools.evaluation import ModelDeliveryReportGenerator


def build_sample_data(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range("2023-01-01", periods=6, freq="MS")
    rows = []
    for month in months:
        good = rng.integers(800, 1300)
        bad = rng.integers(80, 180)
        rows.append({"month": month, "label": 0, "count": good})
        rows.append({"month": month, "label": 1, "count": bad})
    return pd.DataFrame(rows)


def build_performance_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_type": "all", "good": 6200, "bad": 720, "total": 6920, "ks": 0.46, "auc": 0.79, "psi": 0.0},
            {"sample_type": "train", "good": 3500, "bad": 390, "total": 3890, "ks": 0.48, "auc": 0.81, "psi": 0.01},
            {"sample_type": "test", "good": 1700, "bad": 210, "total": 1910, "ks": 0.44, "auc": 0.77, "psi": 0.02},
            {"sample_type": "oot", "good": 1000, "bad": 120, "total": 1120, "ks": 0.41, "auc": 0.75, "psi": 0.03},
        ]
    )


def build_binning_table(rng: np.random.Generator, sample_type: str) -> pd.DataFrame:
    bins = [f"Bin{i}" for i in range(1, 6)]
    good = rng.integers(200, 900, size=len(bins))
    bad = rng.integers(20, 120, size=len(bins))
    table = pd.DataFrame({
        "样本类型": sample_type,
        "分箱": bins,
        "好用户": good,
        "坏用户": bad,
    })
    return table


def build_binning_data(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    sample_types = ["all", "train", "test", "oot"]
    return {sample_type: build_binning_table(rng, sample_type) for sample_type in sample_types}


def build_top_effectiveness(rng: np.random.Generator) -> pd.DataFrame:
    features = [f"var_{i}" for i in range(1, 11)]
    data = []
    for feature in features:
        bins = ["low", "mid", "high"]
        for score_bin in bins:
            data.append({
                "feature_en": feature,
                "feature_cn": f"特征{feature.split('_')[-1]}",
                "data_type": "category",
                "bin": score_bin,
                "n_train": rng.integers(200, 500),
                "train_ratio": rng.uniform(0.05, 0.2),
                "train_bad": rng.integers(10, 60),
                "train_bad_rate": rng.uniform(0.02, 0.15),
                "n_test": rng.integers(80, 200),
                "test_ratio": rng.uniform(0.05, 0.2),
                "test_bad": rng.integers(5, 30),
                "test_bad_rate": rng.uniform(0.02, 0.15),
                "n_oot": rng.integers(60, 150),
                "oot_ratio": rng.uniform(0.05, 0.2),
                "oot_bad": rng.integers(3, 25),
                "oot_bad_rate": rng.uniform(0.02, 0.15),
                "total_gain": rng.uniform(0.01, 0.05),
            })
    return pd.DataFrame(data)


def build_test_monthly_breakdown(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range("2023-01-01", periods=6, freq="MS")
    records = []
    for feature in [f"var_{i}" for i in range(1, 11)]:
        for month in months:
            records.append({
                "feature_en": feature,
                "month": month,
                "N_test": rng.integers(20, 60),
                "分布占比_test": rng.uniform(0.01, 0.08),
                "坏样本数量_test": rng.integers(1, 8),
                "逾期率_test": rng.uniform(0.02, 0.18),
            })
    return pd.DataFrame(records)


def build_feature_stability(rng: np.random.Generator) -> pd.DataFrame:
    features = [f"var_{i}" for i in range(1, 11)]
    data = []
    for feature in features:
        data.append({
            "feature_en": feature,
            "feature_cn": f"特征{feature.split('_')[-1]}",
            "data_type": "category",
            "n_train": rng.integers(800, 1200),
            "iv_train": rng.uniform(0.05, 0.25),
            "psi_train": 0.0,
            "n_test": rng.integers(400, 800),
            "iv_test": rng.uniform(0.04, 0.22),
            "psi_test": rng.uniform(0.01, 0.05),
            "n_oot": rng.integers(300, 600),
            "iv_oot": rng.uniform(0.03, 0.2),
            "psi_oot": rng.uniform(0.02, 0.08),
            "total_gain": rng.uniform(0.05, 0.2),
        })
    return pd.DataFrame(data)


def build_time_distribution(
    rng: np.random.Generator, feature_name: str, data_type: str
) -> pd.DataFrame:
    months = pd.date_range("2023-01-01", periods=6, freq="MS")
    bins = ["low", "mid", "high"]
    rows = []
    for score_bin in bins:
        for month in months:
            num = rng.integers(30, 120)
            rows.append({
                "score_bin": score_bin,
                "data_type": data_type,
                "month": month,
                "num": num,
                "ratio": rng.uniform(0.01, 0.1),
                "bad_ratio": rng.uniform(0.02, 0.18),
                "bad_num": rng.integers(1, max(2, num // 5)),
                "lift": rng.uniform(0.8, 1.4),
            })
    return pd.DataFrame(rows)


def build_feature_time_distributions(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    features = {
        "var_1": "category",
        "var_2": "category",
        "var_3": "numeric",
        "var_4": "numeric",
        "var_5": "category",
        "var_6": "category",
        "var_7": "numeric",
        "var_8": "numeric",
        "var_9": "category",
        "var_10": "numeric",
    }
    return {
        feature: build_time_distribution(rng, feature, data_type)
        for feature, data_type in features.items()
    }


def build_score_time_distribution(rng: np.random.Generator) -> pd.DataFrame:
    return build_time_distribution(rng, "score", "numeric")


def main(output_path: Path | None = None) -> None:
    rng = np.random.default_rng(2023)

    sample_data = build_sample_data(rng)
    performance_data = build_performance_data().to_dict(orient="records")
    binning_data = build_binning_data(rng)
    top_effectiveness = build_top_effectiveness(rng)
    test_monthly_breakdown = build_test_monthly_breakdown(rng)
    feature_stability = build_feature_stability(rng)
    feature_time_distributions = build_feature_time_distributions(rng)
    score_time_distribution = build_score_time_distribution(rng)

    generator = ModelDeliveryReportGenerator()
    report = generator.generate(
        sample_data=sample_data,
        performance_data=performance_data,
        binning_data={k: v.drop(columns="样本类型", errors="ignore") for k, v in binning_data.items()},
        top_variable_effectiveness=top_effectiveness,
        feature_stability=feature_stability,
        feature_time_distributions=feature_time_distributions,
        score_time_distribution=score_time_distribution,
        test_monthly_breakdown=test_monthly_breakdown,
    )

    for section, content in report.items():
        print(f"\n=== {section} ===")
        if isinstance(content, pd.DataFrame):
            print(content.head())
        elif isinstance(content, dict):
            for key, value in content.items():
                print(f"-- {key} --")
                if isinstance(value, pd.DataFrame):
                    print(value.head())
                else:
                    print(value)
        else:
            print(content)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path) as writer:
            report["样本情况"].to_excel(writer, sheet_name="sample_overview", index=False)
            report["模型效果"].to_excel(writer, sheet_name="performance", index=False)
            for sample_type, df in report["模型效果分箱"].items():
                df.to_excel(writer, sheet_name=f"binning_{sample_type}", index=False)
            report["top10变量有效性"].to_excel(writer, sheet_name="top_effectiveness", index=False)
            report["top10变量稳定性"].to_excel(writer, sheet_name="top_stability", index=False)
            for feature, df in report["单特征时间分布top10"].items():
                df.to_excel(writer, sheet_name=f"feature_{feature}", index=False)
            report["模型分表现"].to_excel(writer, sheet_name="score_distribution", index=False)
        print(f"\n导出示例写入: {output_path}")


if __name__ == "__main__":
    main()
