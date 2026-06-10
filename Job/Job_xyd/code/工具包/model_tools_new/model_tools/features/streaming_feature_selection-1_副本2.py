"""
风控建模 - LGB特征筛选 (逐特征流式读取版本)

设计目标：
    特征文件分散在多个CSV中，单文件可能列很多（宽表）。
    一次性读取所有CSV内存放不下，因此每次只读取一个特征列：
        每个特征 -> (打开它所在的csv，只读这一列 + key列) -> 与target拼接 -> 计算IV/单一值占比/缺失率
    这样内存占用 ~ 一个特征列的大小，跟特征总数无关。

筛选维度：
    1. 单一值占比（含NaN）>= single_value_threshold  -> 剔除
    2. 缺失率 >= missing_threshold                    -> 剔除
    3. IV < iv_threshold                              -> 剔除

支持：
    - 断点续跑（根据已写入的日志文件跳过已计算的特征）
    - 多进程并行（每个进程处理一个特征，内存独立可控）
    - 输出详细筛选日志 CSV

参考: /Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new/model_tools/features/selection.py
"""

from __future__ import annotations

import os
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. 索引：构建 "特征名 -> 所在CSV文件" 的映射
# ============================================================

def build_feature_index(
    csv_paths: List[str],
    key_col: str,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    扫描每个CSV的header（只读第一行），构建 特征名 -> csv路径 的映射。

    Parameters
    ----------
    csv_paths : list[str]
        所有特征CSV的路径
    key_col : str
        样本主键列名（每个CSV都应包含此列，用于和target对齐）
    exclude_cols : list[str], optional
        需要排除的列名（如key、target、日期等不参与筛选）

    Returns
    -------
    feature_to_file : dict[str, str]
        特征名 -> 该特征所在的CSV路径
    """
    exclude = set(exclude_cols or [])
    exclude.add(key_col)

    feature_to_file: Dict[str, str] = {}
    for path in csv_paths:
        # 只读header，nrows=0 极快
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        if key_col not in cols:
            raise ValueError(f"CSV {path} 缺少主键列 {key_col}")

        for col in cols:
            if col in exclude:
                continue
            if col in feature_to_file:
                # 多个CSV出现同名列 - 报警告，使用先扫描到的
                warnings.warn(
                    f"特征 {col} 在多个CSV中出现: "
                    f"{feature_to_file[col]} 和 {path}，只保留前者"
                )
                continue
            feature_to_file[col] = path

    print(f"扫描完成: {len(csv_paths)} 个CSV, 共 {len(feature_to_file)} 个候选特征")
    return feature_to_file


# ============================================================
# 2. 加载target（只读一次，常驻内存，体积一般很小）
# ============================================================

def load_target(
    target_path: str,
    key_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    加载target表（key + label），后续每次读特征列后按key左连接。
    """
    df = pd.read_csv(target_path, usecols=[key_col, target_col])
    print(f"target 加载完成: {len(df)} 行, 正样本率 {df[target_col].mean():.4%}")
    return df


# ============================================================
# 3. 单特征指标计算
# ============================================================

def calculate_iv(
    feature: pd.Series,
    target: pd.Series,
    bins: int = 10,
) -> float:
    """计算单特征IV，逻辑参考 model_tools/features/selection.py:91"""
    try:
        df = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if len(df) == 0:
            return 0.0

        # 数值型分箱
        if pd.api.types.is_numeric_dtype(df["feature"]):
            df["bin"] = pd.qcut(
                df["feature"], q=bins, duplicates="drop", precision=3
            )
        else:
            # 类别型按取值直接分箱
            df["bin"] = df["feature"].astype(str)

        grouped = df.groupby("bin")["target"].agg(["count", "sum"])
        grouped["good"] = grouped["count"] - grouped["sum"]
        grouped["bad"] = grouped["sum"]

        total_good = grouped["good"].sum()
        total_bad = grouped["bad"].sum()
        if total_good == 0 or total_bad == 0:
            return 0.0

        # 拉普拉斯平滑，避免log(0)
        eps = 0.5
        grouped["good_rate"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))
        grouped["bad_rate"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))

        grouped["woe"] = np.log(grouped["bad_rate"] / grouped["good_rate"])
        grouped["iv"] = (grouped["bad_rate"] - grouped["good_rate"]) * grouped["woe"]
        grouped = grouped.replace([np.inf, -np.inf], 0)

        return float(grouped["iv"].sum())
    except Exception as e:
        warnings.warn(f"计算IV失败: {e}")
        return 0.0


def evaluate_single_feature(
    feature_name: str,
    feature_series: pd.Series,
    target: pd.Series,
    single_value_threshold: float = 0.95,
    missing_threshold: float = 0.95,
    iv_threshold: float = 0.02,
    iv_bins: int = 10,
) -> Dict:
    """
    对一个特征做完整筛选评估，返回一行记录。
    """
    n = len(feature_series)
    missing_cnt = int(feature_series.isna().sum())
    missing_ratio = missing_cnt / n if n > 0 else 1.0

    value_counts = feature_series.value_counts(dropna=False)
    if len(value_counts) == 0:
        most_freq_val = None
        max_ratio = 1.0
        unique_cnt = 0
    else:
        most_freq_val = value_counts.index[0]
        max_ratio = float(value_counts.iloc[0] / n)
        unique_cnt = int(feature_series.nunique(dropna=False))

    # 决策逻辑（短路）
    keep = True
    reasons: List[str] = []

    if missing_ratio >= missing_threshold:
        keep = False
        reasons.append(f"缺失率{missing_ratio:.2%}>={missing_threshold:.2%}")

    if max_ratio >= single_value_threshold:
        keep = False
        reasons.append(f"单一值占比{max_ratio:.2%}>={single_value_threshold:.2%}")

    # 只在前两步通过的情况下计算IV，省时间
    if keep:
        iv = calculate_iv(feature_series, target, bins=iv_bins)
        if iv < iv_threshold:
            keep = False
            reasons.append(f"IV={iv:.4f}<{iv_threshold}")
    else:
        iv = np.nan  # 不计算

    return {
        "feature": feature_name,
        "n_samples": n,
        "missing_ratio": missing_ratio,
        "unique_count": unique_cnt,
        "most_frequent_value": most_freq_val,
        "max_value_ratio": max_ratio,
        "iv": iv,
        "keep": keep,
        "reason": "; ".join(reasons) if reasons else "通过",
    }


# ============================================================
# 4. 按"特征"为单位读取 + 评估（worker函数，可并行）
# ============================================================

def _process_one_feature(
    args: Tuple[str, str, str, pd.DataFrame, str, Dict],
) -> Dict:
    """
    并行worker：读一个特征列 -> 与target join -> 评估
    """
    feature_name, csv_path, key_col, target_df, target_col, params = args

    # 只读 key + 这一个特征列
    df = pd.read_csv(csv_path, usecols=[key_col, feature_name])

    # 与target对齐（左连接）
    merged = target_df.merge(df, on=key_col, how="left")

    result = evaluate_single_feature(
        feature_name=feature_name,
        feature_series=merged[feature_name],
        target=merged[target_col],
        **params,
    )

    # 立即释放
    del df, merged
    gc.collect()
    return result


# ============================================================
# 5. 主流程
# ============================================================

class StreamingFeatureSelector:
    """
    逐特征流式读取的特征筛选器。

    用法
    ----
    >>> selector = StreamingFeatureSelector(
    ...     csv_paths=["feat_part1.csv", "feat_part2.csv", "feat_part3.csv"],
    ...     target_path="label.csv",
    ...     key_col="user_id",
    ...     target_col="y",
    ...     exclude_cols=["apply_date"],
    ...     log_path="./feature_selection_log.csv",
    ... )
    >>> selector.run(n_jobs=4)
    >>> print(selector.selected_features)
    """

    def __init__(
        self,
        csv_paths: List[str],
        target_path: str,
        key_col: str,
        target_col: str,
        exclude_cols: Optional[List[str]] = None,
        single_value_threshold: float = 0.95,
        missing_threshold: float = 0.95,
        iv_threshold: float = 0.02,
        iv_bins: int = 10,
        log_path: str = "./feature_selection_log.csv",
    ):
        self.csv_paths = csv_paths
        self.target_path = target_path
        self.key_col = key_col
        self.target_col = target_col
        self.exclude_cols = exclude_cols or []
        self.params = dict(
            single_value_threshold=single_value_threshold,
            missing_threshold=missing_threshold,
            iv_threshold=iv_threshold,
            iv_bins=iv_bins,
        )
        self.log_path = log_path

        self.feature_to_file: Dict[str, str] = {}
        self.target_df: Optional[pd.DataFrame] = None
        self.results_: Optional[pd.DataFrame] = None

    # ---------- 断点续跑：读取已有日志 ----------
    def _load_existing_log(self) -> Tuple[List[Dict], set]:
        if not os.path.exists(self.log_path):
            return [], set()
        try:
            df = pd.read_csv(self.log_path)
            done = set(df["feature"].tolist())
            print(f"检测到已有日志 {self.log_path}，已完成 {len(done)} 个特征，将跳过")
            return df.to_dict("records"), done
        except Exception as e:
            warnings.warn(f"读取已有日志失败 {e}, 重新开始")
            return [], set()

    def _append_log(self, row: Dict):
        """每完成一个特征就追加写入，防止中途崩溃丢数据"""
        df = pd.DataFrame([row])
        write_header = not os.path.exists(self.log_path)
        df.to_csv(self.log_path, mode="a", header=write_header, index=False)

    # ---------- 主入口 ----------
    def run(self, n_jobs: int = 1, verbose_every: int = 50):
        # 1. 构建索引
        self.feature_to_file = build_feature_index(
            self.csv_paths,
            key_col=self.key_col,
            exclude_cols=self.exclude_cols + [self.target_col],
        )

        # 2. 加载target
        self.target_df = load_target(
            self.target_path,
            key_col=self.key_col,
            target_col=self.target_col,
        )

        # 3. 处理断点续跑
        existing_results, done_set = self._load_existing_log()
        todo = [(f, p) for f, p in self.feature_to_file.items() if f not in done_set]
        print(f"待处理特征数: {len(todo)} (已完成 {len(done_set)})")

        results: List[Dict] = list(existing_results)

        # 4. 逐特征处理
        if n_jobs <= 1:
            for i, (feat, path) in enumerate(todo, 1):
                row = _process_one_feature(
                    (feat, path, self.key_col, self.target_df, self.target_col, self.params)
                )
                results.append(row)
                self._append_log(row)
                if i % verbose_every == 0:
                    print(f"[{i}/{len(todo)}] {feat} -> keep={row['keep']}, iv={row['iv']}")
        else:
            # 多进程：注意 target_df 会被pickle到子进程，体积较小可以接受
            tasks = [
                (feat, path, self.key_col, self.target_df, self.target_col, self.params)
                for feat, path in todo
            ]
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = {ex.submit(_process_one_feature, t): t[0] for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    feat = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as e:
                        warnings.warn(f"特征 {feat} 处理失败: {e}")
                        row = {
                            "feature": feat, "n_samples": np.nan,
                            "missing_ratio": np.nan, "unique_count": np.nan,
                            "most_frequent_value": None, "max_value_ratio": np.nan,
                            "iv": np.nan, "keep": False, "reason": f"异常: {e}",
                        }
                    results.append(row)
                    self._append_log(row)
                    if i % verbose_every == 0:
                        print(f"[{i}/{len(todo)}] {feat} -> keep={row['keep']}, iv={row.get('iv')}")

        # 5. 汇总
        self.results_ = (
            pd.DataFrame(results)
            .drop_duplicates(subset=["feature"], keep="last")
            .sort_values("iv", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

        n_keep = int(self.results_["keep"].sum())
        print(f"\n筛选完成: 总特征 {len(self.results_)}, 保留 {n_keep}, "
              f"剔除 {len(self.results_) - n_keep}")
        return self.results_

    @property
    def selected_features(self) -> List[str]:
        if self.results_ is None:
            raise RuntimeError("请先调用 run()")
        return self.results_.loc[self.results_["keep"], "feature"].tolist()

    def summary(self) -> pd.DataFrame:
        """按剔除原因统计"""
        if self.results_ is None:
            raise RuntimeError("请先调用 run()")
        df = self.results_.copy()
        df["bucket"] = df["reason"].str.split(";").str[0].str.split("=").str[0].str.split("<").str[0]
        return df.groupby("bucket").size().reset_index(name="count")


# ============================================================
# 6. 使用示例
# ============================================================

if __name__ == "__main__":
    # ----------- 配置 -----------
    CSV_PATHS = [
        "/path/to/feature_part1.csv",
        "/path/to/feature_part2.csv",
        "/path/to/feature_part3.csv",
    ]
    TARGET_PATH = "/path/to/label.csv"   # 含 user_id 和 y
    KEY_COL = "user_id"
    TARGET_COL = "y"
    EXCLUDE_COLS = ["apply_date", "channel"]  # 不参与筛选的非特征列

    selector = StreamingFeatureSelector(
        csv_paths=CSV_PATHS,
        target_path=TARGET_PATH,
        key_col=KEY_COL,
        target_col=TARGET_COL,
        exclude_cols=EXCLUDE_COLS,
        single_value_threshold=0.95,
        missing_threshold=0.95,
        iv_threshold=0.02,
        iv_bins=10,
        log_path="./feature_selection_log.csv",
    )

    # 单进程跑（最稳）
    results = selector.run(n_jobs=1, verbose_every=20)

    # 或多进程加速（推荐 n_jobs <= CPU核数 - 1）
    # results = selector.run(n_jobs=4, verbose_every=20)

    print("\n保留特征数:", len(selector.selected_features))
    print("Top 20 by IV:")
    print(results.head(20))

    # 保存最终保留的特征清单
    pd.Series(selector.selected_features, name="feature").to_csv(
        "./selected_features.csv", index=False
    )
