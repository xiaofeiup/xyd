"""
特征选择模块

提供多种特征选择方法，包括基于统计的、基于模型的特征选择
"""

import pandas as pd
import numpy as np
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def filter_features_by_single_value_ratio(
    data: pd.DataFrame,
    threshold: float = 0.95,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    基于单一值占比筛选特征（优化版本：使用向量化操作）

    如果某个特征的最频繁值（包含空值）占总数据量的比例超过阈值，则去掉该特征

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    threshold : float, default=0.95
        单一值占比阈值
    exclude_cols : list, optional
        排除检查的列名列表（如目标变量）

    Returns:
    --------
    selected_features : list
        保留的特征列表
    filter_log : pd.DataFrame
        筛选日志记录
    """
    exclude_cols = exclude_cols or []
    check_cols = [col for col in data.columns if col not in exclude_cols]
    
    total_count = len(data)
    
    # 向量化计算：一次性计算所有特征的统计
    filter_results = []
    
    for col in check_cols:
        # 计算每个值的频次（包括NaN）
        value_counts = data[col].value_counts(dropna=False)
        
        if len(value_counts) == 0:
            # 空列
            max_ratio = 1.0
            most_frequent_value = None
            reason = "空列"
            keep_feature = False
        else:
            # 最频繁值及其占比
            most_frequent_value = value_counts.index[0]
            max_count = value_counts.iloc[0]
            max_ratio = max_count / total_count
            
            # Treat the public percentage threshold as a two-decimal value.
            # This avoids a surprising pass for values such as 949/1000 when
            # users configure a 95% cutoff and matches report display logic.
            if round(max_ratio, 2) >= threshold:
                reason = f"最频繁值占比{max_ratio:.2%}超过阈值{threshold:.2%}"
                keep_feature = False
            else:
                reason = f"最频繁值占比{max_ratio:.2%}，通过筛选"
                keep_feature = True
        
        filter_results.append({
            'feature': col,
            'keep_feature': keep_feature,
            'most_frequent_value': most_frequent_value,
            'max_value_ratio': max_ratio,
            'unique_count': data[col].nunique(dropna=False),
            'missing_ratio': data[col].isna().sum() / total_count,
            'filter_reason': reason
        })
    
    selected_features = [item['feature'] for item in filter_results if item['keep_feature']]
    filter_log = pd.DataFrame(filter_results)
    
    return selected_features, filter_log


def calculate_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
    """
    计算Information Value (IV)

    Parameters:
    -----------
    feature : pd.Series
        特征变量
    target : pd.Series
        目标变量 (0/1)
    bins : int, default=10
        分箱数量

    Returns:
    --------
    iv : float
        IV值
    """
    try:
        # 处理缺失值
        df = pd.DataFrame({'feature': feature, 'target': target}).dropna()

        if len(df) == 0:
            return 0.0

        # 分箱：数值型用等频分箱，类别型按取值直接分箱
        if pd.api.types.is_numeric_dtype(df['feature']):
            df['bin'] = pd.qcut(df['feature'], q=bins, duplicates='drop', precision=3)
        else:
            df['bin'] = df['feature'].astype(str)

        # 计算每个分箱的统计量
        grouped = df.groupby('bin')['target'].agg(['count', 'sum'])
        grouped['good'] = grouped['count'] - grouped['sum']  # 好客户数
        grouped['bad'] = grouped['sum']  # 坏客户数

        # 计算总的好坏客户数
        total_good = grouped['good'].sum()
        total_bad = grouped['bad'].sum()

        if total_good == 0 or total_bad == 0:
            return 0.0

        # 拉普拉斯平滑，避免 log(0)
        eps = 0.5
        grouped['good_rate'] = (grouped['good'] + eps) / (total_good + eps * len(grouped))
        grouped['bad_rate'] = (grouped['bad'] + eps) / (total_bad + eps * len(grouped))

        # 计算WOE和IV
        grouped['woe'] = np.log(grouped['bad_rate'] / grouped['good_rate'])
        grouped['iv_component'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']

        # 处理无穷大值
        grouped = grouped.replace([np.inf, -np.inf], 0)

        return float(grouped['iv_component'].sum())

    except Exception as e:
        warnings.warn(f"计算IV时出错: {str(e)}")
        return 0.0


def _validate_quality_parameters(
    missing_threshold: float,
    single_value_threshold: float,
    iv_threshold: float,
    iv_bins: int,
    psi_threshold: float,
    psi_bins: int,
    psi_min_sample: int,
    invalid_psi_policy: str,
) -> None:
    """校验质量筛选器参数。"""
    for name, value in (
        ("missing_threshold", missing_threshold),
        ("single_value_threshold", single_value_threshold),
    ):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} 必须是 [0, 1] 范围内的数值")

    if not isinstance(iv_threshold, (int, float)) or iv_threshold < 0:
        raise ValueError("iv_threshold 必须是大于等于 0 的数值")
    if not isinstance(psi_threshold, (int, float)) or psi_threshold < 0:
        raise ValueError("psi_threshold 必须是大于等于 0 的数值")
    if not isinstance(iv_bins, int) or iv_bins < 2:
        raise ValueError("iv_bins 必须是大于等于 2 的整数")
    if not isinstance(psi_bins, int) or psi_bins < 2:
        raise ValueError("psi_bins 必须是大于等于 2 的整数")
    if not isinstance(psi_min_sample, int) or psi_min_sample < 1:
        raise ValueError("psi_min_sample 必须是大于等于 1 的整数")
    if invalid_psi_policy not in {"drop", "keep", "raise"}:
        raise ValueError("invalid_psi_policy 只能是 'drop'、'keep' 或 'raise'")


def _prepare_binary_target(
    target: Union[pd.Series, List, np.ndarray],
    index: pd.Index,
    positive_label: Union[str, int, float],
) -> pd.Series:
    """校验二分类目标变量，并按正样本标签转换为 0/1。"""
    if isinstance(target, pd.Series):
        if len(target) != len(index):
            raise ValueError("target 与特征数据的行数不一致")
        if not target.index.equals(index):
            raise ValueError("target 与特征数据的索引不一致，请先完成样本对齐")
        target_series = target.copy()
    else:
        target_array = np.asarray(target)
        if target_array.ndim != 1:
            raise ValueError("target 必须是一维数据")
        if len(target_array) != len(index):
            raise ValueError("target 与特征数据的行数不一致")
        target_series = pd.Series(target_array, index=index)

    valid_target = target_series.dropna()
    labels = pd.unique(valid_target)
    if len(labels) != 2:
        raise ValueError(f"IV 筛选要求二分类目标变量，当前有效类别数为 {len(labels)}")
    if not valid_target.eq(positive_label).fillna(False).any():
        raise ValueError(f"positive_label={positive_label!r} 不在目标变量中")

    binary_target = pd.Series(np.nan, index=index, dtype=float)
    valid_mask = target_series.notna()
    binary_target.loc[valid_mask] = (
        target_series.loc[valid_mask].eq(positive_label).astype(int)
    )
    return binary_target


def _build_time_blocks(
    data: pd.DataFrame,
    time_col: Optional[str],
    time_blocks: Optional[int],
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    """按时间排序后切分为等样本量的连续时间块。"""
    if time_blocks is None:
        if time_col is not None:
            raise ValueError("指定 time_col 时必须同时指定 time_blocks")
        return [], []
    if not isinstance(time_blocks, int) or time_blocks < 2:
        raise ValueError("time_blocks 必须是大于等于 2 的整数")
    if not time_col:
        raise ValueError("启用 PSI 时间分块时必须指定 time_col")
    if time_col not in data.columns:
        raise ValueError(f"时间列 {time_col!r} 不在 data 中")

    raw_time = data[time_col]
    if pd.api.types.is_datetime64_any_dtype(raw_time):
        parsed_time = pd.to_datetime(raw_time, errors="coerce")
    elif pd.api.types.is_numeric_dtype(raw_time):
        # 业务日期通常以 20240101 形式存储，优先按 YYYYMMDD 解析。
        compact_time = pd.to_datetime(
            raw_time.astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        generic_time = pd.to_datetime(raw_time, errors="coerce")
        parsed_time = (
            compact_time
            if compact_time.notna().sum() >= generic_time.notna().sum()
            else generic_time
        )
    else:
        parsed_time = pd.to_datetime(raw_time, errors="coerce")

    valid_positions = np.flatnonzero(parsed_time.notna().to_numpy())
    invalid_count = len(data) - len(valid_positions)
    if invalid_count:
        warnings.warn(
            f"时间列 {time_col} 有 {invalid_count} 条空值或无法解析的值，"
            "这些样本不参与 PSI 计算"
        )
    if len(valid_positions) < time_blocks:
        raise ValueError(
            f"有效时间样本数 {len(valid_positions)} 小于 time_blocks={time_blocks}"
        )

    time_values = parsed_time.iloc[valid_positions].to_numpy()
    sorted_positions = valid_positions[np.argsort(time_values, kind="stable")]
    block_positions = [
        np.asarray(block, dtype=int)
        for block in np.array_split(sorted_positions, time_blocks)
    ]
    block_summary = []
    for block_number, positions in enumerate(block_positions, 1):
        block_times = parsed_time.iloc[positions]
        block_summary.append({
            "block": block_number,
            "n_samples": len(positions),
            "start_time": block_times.min(),
            "end_time": block_times.max(),
        })
    return block_positions, block_summary


def _evaluate_feature_quality_binary(
    feature_name: str,
    feature_series: pd.Series,
    binary_target: pd.Series,
    psi_base_series: Optional[pd.Series],
    psi_comparison_series: Optional[List[Tuple[str, pd.Series]]],
    missing_threshold: float,
    single_value_threshold: float,
    iv_threshold: float,
    iv_bins: int,
    psi_threshold: float,
    psi_bins: int,
    psi_min_sample: int,
    invalid_psi_policy: str,
    calculate_metrics_for_rejected: bool,
) -> Dict[str, Union[str, int, float, bool, object]]:
    """使用已校验的参数评估单个特征。"""
    sample_count = len(feature_series)
    missing_count = int(feature_series.isna().sum())
    missing_ratio = missing_count / sample_count if sample_count else 1.0

    value_counts = feature_series.value_counts(dropna=False)
    if value_counts.empty:
        most_frequent_value = None
        max_value_ratio = 1.0
        unique_count = 0
    else:
        most_frequent_value = value_counts.index[0]
        max_value_ratio = float(value_counts.iloc[0] / sample_count)
        unique_count = int(feature_series.nunique(dropna=False))

    reasons: List[str] = []
    if missing_ratio >= missing_threshold:
        reasons.append(f"缺失率{missing_ratio:.2%}>={missing_threshold:.2%}")
    if max_value_ratio >= single_value_threshold:
        reasons.append(
            f"单一值占比{max_value_ratio:.2%}>={single_value_threshold:.2%}"
        )

    should_calculate = not reasons or calculate_metrics_for_rejected
    iv_value = (
        calculate_iv(feature_series, binary_target, bins=iv_bins)
        if should_calculate
        else np.nan
    )
    if should_calculate and iv_value < iv_threshold:
        reasons.append(f"IV={iv_value:.4f}<{iv_threshold}")

    psi_values: Dict[str, float] = {}
    if (
        psi_base_series is not None
        and psi_comparison_series
        and (not reasons or calculate_metrics_for_rejected)
    ):
        # 局部导入避免包初始化时产生循环依赖。
        from ..evaluation.metrics import calculate_psi

        for comparison_name, comparison_series in psi_comparison_series:
            psi_value, _ = calculate_psi(
                psi_base_series,
                comparison_series,
                bins=psi_bins,
                min_sample=psi_min_sample,
                feature_name=feature_name,
            )
            psi_values[comparison_name] = float(psi_value)
            if not np.isfinite(psi_value):
                message = f"PSI({comparison_name})无法计算"
                if invalid_psi_policy == "raise":
                    raise ValueError(f"特征 {feature_name} {message}")
                if invalid_psi_policy == "drop":
                    reasons.append(message)
            elif psi_value >= psi_threshold:
                reasons.append(
                    f"PSI({comparison_name})={psi_value:.4f}>={psi_threshold}"
                )

    finite_psi_values = [value for value in psi_values.values() if np.isfinite(value)]
    max_psi = max(finite_psi_values) if finite_psi_values else np.nan
    return {
        "feature": feature_name,
        "n_samples": sample_count,
        "missing_ratio": missing_ratio,
        "unique_count": unique_count,
        "most_frequent_value": most_frequent_value,
        "max_value_ratio": max_value_ratio,
        "iv": iv_value,
        "psi": max_psi,
        "psi_values": psi_values,
        "keep": not reasons,
        "reason": "; ".join(reasons) if reasons else "通过",
    }


def evaluate_feature_quality(
    feature_name: str,
    feature_series: pd.Series,
    target: Union[pd.Series, List, np.ndarray],
    missing_threshold: float = 0.95,
    single_value_threshold: float = 0.95,
    iv_threshold: float = 0.02,
    iv_bins: int = 10,
    positive_label: Union[str, int, float] = 1,
) -> Dict[str, Union[str, int, float, bool, object]]:
    """按缺失率、单一值占比和 IV 评估单个特征。

    PSI 依赖完整数据的时间列和时间分块，应使用
    :func:`filter_features_by_quality` 或 :class:`FeatureQualitySelector`。
    """
    if not isinstance(feature_series, pd.Series):
        feature_series = pd.Series(feature_series)
    _validate_quality_parameters(
        missing_threshold, single_value_threshold, iv_threshold, iv_bins,
        0.25, 10, 10, "drop",
    )
    binary_target = _prepare_binary_target(
        target, feature_series.index, positive_label
    )
    return _evaluate_feature_quality_binary(
        feature_name=feature_name,
        feature_series=feature_series,
        binary_target=binary_target,
        psi_base_series=None,
        psi_comparison_series=None,
        missing_threshold=missing_threshold,
        single_value_threshold=single_value_threshold,
        iv_threshold=iv_threshold,
        iv_bins=iv_bins,
        psi_threshold=0.25,
        psi_bins=10,
        psi_min_sample=10,
        invalid_psi_policy="drop",
        calculate_metrics_for_rejected=False,
    )


def filter_features_by_quality(
    data: pd.DataFrame,
    target: Union[pd.Series, List, np.ndarray],
    time_col: Optional[str] = None,
    time_blocks: Optional[int] = None,
    features: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    missing_threshold: float = 0.95,
    single_value_threshold: float = 0.95,
    iv_threshold: float = 0.02,
    iv_bins: int = 10,
    psi_threshold: float = 0.25,
    psi_bins: int = 10,
    psi_min_sample: int = 10,
    positive_label: Union[str, int, float] = 1,
    invalid_psi_policy: str = "drop",
    calculate_metrics_for_rejected: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    """按缺失率、单一值占比、IV 和时间分块 PSI 筛选特征。

    数据按 ``time_col`` 升序切成 ``time_blocks`` 个等样本量时间块。
    第 1 块为基准，依次与第 2 至第 N 块计算 PSI；任一次 PSI
    达到 ``psi_threshold`` 都会剔除该特征。未设置时间参数时跳过 PSI。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 pandas.DataFrame")
    if not data.columns.is_unique:
        raise ValueError("data 中存在重复列名")

    _validate_quality_parameters(
        missing_threshold, single_value_threshold, iv_threshold, iv_bins,
        psi_threshold, psi_bins, psi_min_sample, invalid_psi_policy,
    )
    binary_target = _prepare_binary_target(target, data.index, positive_label)
    block_positions, block_summary = _build_time_blocks(
        data, time_col, time_blocks
    )

    excluded = set(exclude_cols or [])
    if time_col is not None:
        excluded.add(time_col)
    candidate_features = (
        list(data.columns) if features is None else list(features)
    )
    candidate_features = [f for f in candidate_features if f not in excluded]
    missing_features = [f for f in candidate_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"以下特征不在 data 中: {missing_features}")

    results = []
    for feature in candidate_features:
        if block_positions:
            psi_base_series = data[feature].iloc[block_positions[0]]
            psi_comparisons = [
                (
                    f"block_1_vs_block_{block_number}",
                    data[feature].iloc[positions],
                )
                for block_number, positions in enumerate(block_positions[1:], 2)
            ]
        else:
            psi_base_series = None
            psi_comparisons = None

        results.append(_evaluate_feature_quality_binary(
            feature_name=feature,
            feature_series=data[feature],
            binary_target=binary_target,
            psi_base_series=psi_base_series,
            psi_comparison_series=psi_comparisons,
            missing_threshold=missing_threshold,
            single_value_threshold=single_value_threshold,
            iv_threshold=iv_threshold,
            iv_bins=iv_bins,
            psi_threshold=psi_threshold,
            psi_bins=psi_bins,
            psi_min_sample=psi_min_sample,
            invalid_psi_policy=invalid_psi_policy,
            calculate_metrics_for_rejected=calculate_metrics_for_rejected,
        ))

    columns = [
        "feature", "n_samples", "missing_ratio", "unique_count",
        "most_frequent_value", "max_value_ratio", "iv", "psi",
        "psi_values", "keep", "reason",
    ]
    selection_log = pd.DataFrame(results, columns=columns)
    selection_log.attrs["time_block_summary"] = block_summary
    selected_features = (
        selection_log.loc[selection_log["keep"], "feature"].tolist()
        if not selection_log.empty else []
    )
    return selected_features, selection_log


class FeatureQualitySelector:
    """基于缺失率、单一值占比、IV 和时间分块 PSI 的特征筛选器。"""

    def __init__(
        self,
        missing_threshold: float = 0.95,
        single_value_threshold: float = 0.95,
        iv_threshold: float = 0.02,
        iv_bins: int = 10,
        time_col: Optional[str] = None,
        time_blocks: Optional[int] = None,
        psi_threshold: float = 0.25,
        psi_bins: int = 10,
        psi_min_sample: int = 10,
        positive_label: Union[str, int, float] = 1,
        exclude_cols: Optional[List[str]] = None,
        invalid_psi_policy: str = "drop",
        calculate_metrics_for_rejected: bool = False,
    ):
        _validate_quality_parameters(
            missing_threshold, single_value_threshold, iv_threshold, iv_bins,
            psi_threshold, psi_bins, psi_min_sample, invalid_psi_policy,
        )
        if (time_col is None) != (time_blocks is None):
            raise ValueError("time_col 和 time_blocks 必须同时指定或同时为空")
        if time_blocks is not None and (
            not isinstance(time_blocks, int) or time_blocks < 2
        ):
            raise ValueError("time_blocks 必须是大于等于 2 的整数")

        self.missing_threshold = missing_threshold
        self.single_value_threshold = single_value_threshold
        self.iv_threshold = iv_threshold
        self.iv_bins = iv_bins
        self.time_col = time_col
        self.time_blocks = time_blocks
        self.psi_threshold = psi_threshold
        self.psi_bins = psi_bins
        self.psi_min_sample = psi_min_sample
        self.positive_label = positive_label
        self.exclude_cols = exclude_cols or []
        self.invalid_psi_policy = invalid_psi_policy
        self.calculate_metrics_for_rejected = calculate_metrics_for_rejected
        self.selected_features_: Optional[List[str]] = None
        self.selection_log_: Optional[pd.DataFrame] = None
        self.psi_log_: Optional[pd.DataFrame] = None
        self.time_block_summary_: Optional[pd.DataFrame] = None
        self.feature_scores_: Optional[Dict[str, float]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, List, np.ndarray],
    ) -> "FeatureQualitySelector":
        """计算筛选指标，PSI 由 X 中的时间列自动切块计算。"""
        self.selected_features_, self.selection_log_ = filter_features_by_quality(
            data=X,
            target=y,
            time_col=self.time_col,
            time_blocks=self.time_blocks,
            exclude_cols=self.exclude_cols,
            missing_threshold=self.missing_threshold,
            single_value_threshold=self.single_value_threshold,
            iv_threshold=self.iv_threshold,
            iv_bins=self.iv_bins,
            psi_threshold=self.psi_threshold,
            psi_bins=self.psi_bins,
            psi_min_sample=self.psi_min_sample,
            positive_label=self.positive_label,
            invalid_psi_policy=self.invalid_psi_policy,
            calculate_metrics_for_rejected=self.calculate_metrics_for_rejected,
        )
        self.feature_scores_ = dict(
            zip(self.selection_log_["feature"], self.selection_log_["iv"])
        )
        self.time_block_summary_ = pd.DataFrame(
            self.selection_log_.attrs.get("time_block_summary", [])
        )
        psi_records = []
        for row in self.selection_log_.itertuples(index=False):
            for comparison, psi_value in row.psi_values.items():
                psi_records.append({
                    "feature": row.feature,
                    "comparison": comparison,
                    "psi": psi_value,
                    "psi_threshold": self.psi_threshold,
                    "pass": bool(
                        np.isfinite(psi_value) and psi_value < self.psi_threshold
                    ),
                })
        self.psi_log_ = pd.DataFrame(
            psi_records,
            columns=["feature", "comparison", "psi", "psi_threshold", "pass"],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """仅返回拟合时通过筛选的特征。"""
        if self.selected_features_ is None:
            raise ValueError("请先调用 fit 方法")
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            raise ValueError(f"待转换数据缺少特征: {missing_features}")
        return X.loc[:, self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, List, np.ndarray],
    ) -> pd.DataFrame:
        """拟合筛选器并返回筛选后的数据。"""
        return self.fit(X, y).transform(X)

    def get_selection_log(self) -> pd.DataFrame:
        """返回每个特征的四维度指标及筛选原因。"""
        if self.selection_log_ is None:
            raise ValueError("请先调用 fit 方法")
        return self.selection_log_.copy()

    def get_psi_log(self) -> pd.DataFrame:
        """返回每个特征与各后续时间块之间的 PSI 明细。"""
        if self.psi_log_ is None:
            raise ValueError("请先调用 fit 方法")
        return self.psi_log_.copy()

    def get_time_block_summary(self) -> pd.DataFrame:
        """返回各时间块的样本数和起止时间。"""
        if self.time_block_summary_ is None:
            raise ValueError("请先调用 fit 方法")
        return self.time_block_summary_.copy()

    def get_selection_summary(self) -> Dict[str, Union[int, float, List[str]]]:
        """返回筛选结果摘要。"""
        if self.selection_log_ is None or self.selected_features_ is None:
            raise ValueError("请先调用 fit 方法")
        return {
            "original_features": len(self.selection_log_),
            "selected_features_count": len(self.selected_features_),
            "removed_features_count": (
                len(self.selection_log_) - len(self.selected_features_)
            ),
            "missing_threshold": self.missing_threshold,
            "single_value_threshold": self.single_value_threshold,
            "iv_threshold": self.iv_threshold,
            "psi_threshold": self.psi_threshold,
            "time_blocks": self.time_blocks or 0,
            "selected_features": list(self.selected_features_),
        }


class FeatureSelector:
    """
    特征选择器

    集成多种特征选择方法，包括基于单一值占比的预筛选
    """

    def __init__(self,
                 method: str = 'iv',
                 k_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 single_value_threshold: float = 0.95,
                 iv_threshold: float = 0.1):
        """
        初始化特征选择器

        Parameters:
        -----------
        method : str, default='iv'
            选择方法 ('iv', 'chi2', 'f_classif', 'mutual_info', 'rfe', 'lasso')
        k_features : int, optional
            选择的特征数量
        threshold : float, optional
            选择阈值
        single_value_threshold : float, default=0.95
            单一值占比阈值，超过此比例的特征将被预先过滤
        iv_threshold : float, default=0.1
            IV阈值，小于此值的特征将被过滤
        """
        self.method = method
        self.k_features = k_features
        self.threshold = threshold
        self.single_value_threshold = single_value_threshold
        self.iv_threshold = iv_threshold
        self.selected_features_ = None
        self.feature_scores_ = None
        self.filter_log_ = None
        self.pre_filtered_features_ = None
        self.iv_log_ = None

    def _fit_iv_selection(self, X: pd.DataFrame, y: pd.Series):
        """
        基于IV的特征选择（优化版本：移除不必要的并行逻辑）

        添加IV阈值筛选：小于阈值的特征会被去掉
        """
        def _calculate_feature_iv(feature: str) -> Dict[str, Union[str, float, bool]]:
            try:
                iv_value = calculate_iv(X[feature], y)

                # 判断是否通过IV阈值
                if iv_value < self.iv_threshold:
                    reason = f"IV值{iv_value:.4f}小于阈值{self.iv_threshold}"
                    keep_feature = False
                else:
                    reason = f"IV值{iv_value:.4f}通过阈值筛选"
                    keep_feature = True

                # IV值解释
                if iv_value < 0.02:
                    interpretation = "无预测能力"
                elif iv_value < 0.1:
                    interpretation = "弱预测能力"
                elif iv_value < 0.3:
                    interpretation = "中等预测能力"
                elif iv_value < 0.5:
                    interpretation = "强预测能力"
                else:
                    interpretation = "过强预测能力(可能过拟合)"

                return {
                    'feature': feature,
                    'iv_value': iv_value,
                    'interpretation': interpretation,
                    'keep_feature': keep_feature,
                    'filter_reason': reason
                }

            except Exception as e:
                warnings.warn(f"特征 {feature} IV计算失败: {str(e)}")
                return {
                    'feature': feature,
                    'iv_value': 0.0,
                    'interpretation': "计算失败",
                    'keep_feature': False,
                    'filter_reason': f"IV计算失败: {str(e)}"
                }

        # 计算所有特征的IV值（单线程，避免GIL瓶颈）
        features = list(X.columns)
        iv_results = [_calculate_feature_iv(feature) for feature in features]

        # 保存IV筛选日志
        self.iv_log_ = pd.DataFrame(iv_results).sort_values('iv_value', ascending=False)

        # 通过IV阈值的特征
        iv_passed_features = self.iv_log_[self.iv_log_['keep_feature']]['feature'].tolist()

        # 设置特征得分
        self.feature_scores_ = dict(zip(self.iv_log_['feature'], self.iv_log_['iv_value']))

        # 根据参数进一步筛选
        if self.k_features:
            # 选择IV最高的k个特征（在通过阈值的特征中）
            top_features = self.iv_log_[self.iv_log_['keep_feature']].head(self.k_features)
            self.selected_features_ = top_features['feature'].tolist()
        elif self.threshold:
            # 使用自定义阈值（这里threshold指代其他阈值，不是iv_threshold）
            self.selected_features_ = self.iv_log_[
                (self.iv_log_['keep_feature']) &
                (self.iv_log_['iv_value'] >= self.threshold)
            ]['feature'].tolist()
        else:
            # 默认选择所有通过IV阈值的特征
            self.selected_features_ = iv_passed_features

        print(f"IV筛选完成：通过IV阈值({self.iv_threshold})的特征数量: {len(iv_passed_features)}")
        print(f"最终选择特征数量: {len(self.selected_features_)}")

    def fit(self, X: pd.DataFrame, y: pd.Series, target_col: Optional[str] = None) -> 'FeatureSelector':
        """
        拟合特征选择器

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        target_col : str, optional
            目标变量列名，用于排除预筛选

        Returns:
        --------
        self : FeatureSelector
            拟合后的选择器
        """
        # 第一步：基于单一值占比预筛选
        exclude_cols = [target_col] if target_col else []
        self.pre_filtered_features_, self.filter_log_ = filter_features_by_single_value_ratio(
            X,
            threshold=self.single_value_threshold,
            exclude_cols=exclude_cols
        )

        print(f"预筛选完成：{len(X.columns)} -> {len(self.pre_filtered_features_)} 个特征")

        # 使用预筛选后的特征进行后续选择
        X_filtered = X[self.pre_filtered_features_]

        if len(self.pre_filtered_features_) == 0:
            print("警告：预筛选后没有剩余特征")
            self.selected_features_ = []
            self.feature_scores_ = {}
            return self

        # 第二步：应用指定的特征选择方法
        if self.method == 'iv':
            self._fit_iv_selection(X_filtered, y)
        elif self.method == 'rfe':
            self._fit_rfe_selection(X_filtered, y)
        elif self.method == 'lasso':
            self._fit_lasso_selection(X_filtered, y)
        elif self.method == 'chi2':
            self._fit_statistical_selection(X_filtered, y, chi2)
        elif self.method == 'f_classif':
            self._fit_statistical_selection(X_filtered, y, f_classif)
        elif self.method == 'mutual_info':
            self._fit_mutual_info_selection(X_filtered, y)
        else:
            raise ValueError(f"不支持的选择方法: {self.method}")

        return self

    def get_iv_log(self) -> pd.DataFrame:
        """
        获取IV筛选日志

        Returns:
        --------
        iv_log : pd.DataFrame
            IV计算和筛选的详细记录
        """
        if self.iv_log_ is None:
            raise ValueError("请先调用fit方法")

        return self.iv_log_

    def get_filter_log(self) -> pd.DataFrame:
        """
        获取预筛选日志

        Returns:
        --------
        filter_log : pd.DataFrame
            详细的预筛选记录
        """
        if self.filter_log_ is None:
            raise ValueError("请先调用fit方法")

        return self.filter_log_

    def get_selection_summary(self) -> Dict:
        """
        获取特征选择总结

        Returns:
        --------
        summary : dict
            选择总结信息
        """
        if self.filter_log_ is None:
            raise ValueError("请先调用fit方法")

        pre_filter_removed = len(self.filter_log_[~self.filter_log_['keep_feature']])
        final_selected = len(self.selected_features_) if self.selected_features_ else 0

        # iv_log_ 仅在 method='iv' 时存在，其他方法跳过 IV 相关统计
        if self.iv_log_ is not None:
            iv_filter_removed = len(self.iv_log_[~self.iv_log_['keep_feature']])
            iv_filter_remaining = len(self.iv_log_[self.iv_log_['keep_feature']])
        else:
            iv_filter_removed = 0
            iv_filter_remaining = final_selected

        return {
            'original_features': len(self.filter_log_),
            'pre_filter_removed': pre_filter_removed,
            'pre_filter_remaining': len(self.pre_filtered_features_),
            'iv_filter_removed': iv_filter_removed,
            'iv_filter_remaining': iv_filter_remaining,
            'final_selected': final_selected,
            'single_value_threshold': self.single_value_threshold,
            'iv_threshold': self.iv_threshold,
            'selected_features': self.selected_features_ or [],
            'filter_steps': [
                {
                    'step': 'single_value_filter',
                    'removed': pre_filter_removed,
                    'remaining': len(self.pre_filtered_features_)
                },
                {
                    'step': f'{self.method}_filter',
                    'removed': iv_filter_removed,
                    'remaining': iv_filter_remaining
                }
            ]
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，只保留选中的特征

        Parameters:
        -----------
        X : pd.DataFrame
            输入数据

        Returns:
        --------
        X_selected : pd.DataFrame
            选中特征的数据
        """
        if self.selected_features_ is None:
            raise ValueError("请先调用fit方法")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        拟合并转换数据

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        target_col : str, optional
            目标变量列名

        Returns:
        --------
        X_selected : pd.DataFrame
            选中特征的数据
        """
        return self.fit(X, y, target_col).transform(X)

    def _fit_rfe_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于递归特征消除的选择"""
        estimator = LogisticRegression(random_state=42, max_iter=1000)

        n_features = self.k_features or max(1, len(X.columns) // 2)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)

        self.selected_features_ = X.columns[rfe.support_].tolist()
        self.feature_scores_ = dict(zip(X.columns, rfe.ranking_))

    def _fit_lasso_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于Lasso的特征选择（优化版本：移除 n_jobs 参数）"""
        from sklearn.linear_model import LassoCV

        # 使用交叉验证选择最优alpha（单线程，避免GIL瓶颈）
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X, y)

        # 选择系数非零的特征
        selected_mask = lasso.coef_ != 0
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.feature_scores_ = dict(zip(X.columns, np.abs(lasso.coef_)))

    def _fit_statistical_selection(self, X: pd.DataFrame, y: pd.Series, score_func):
        """基于统计检验的特征选择"""
        # 处理负值（chi2要求非负值）
        if score_func == chi2:
            X_processed = X.copy()
            # 将负值设为0
            X_processed[X_processed < 0] = 0
        else:
            X_processed = X

        if self.k_features:
            selector = SelectKBest(score_func=score_func, k=self.k_features)
        else:
            selector = SelectKBest(score_func=score_func, k='all')

        selector.fit(X_processed, y)

        self.feature_scores_ = dict(zip(X.columns, selector.scores_))

        if self.k_features:
            self.selected_features_ = X.columns[selector.get_support()].tolist()
        elif self.threshold:
            # 使用分数阈值
            scores = selector.scores_
            selected_mask = scores >= self.threshold
            self.selected_features_ = X.columns[selected_mask].tolist()
        else:
            # 选择前50%的特征
            k = max(1, len(X.columns) // 2)
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X_processed, y)
            self.selected_features_ = X.columns[selector.get_support()].tolist()

    def _fit_mutual_info_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于互信息的特征选择"""
        scores = mutual_info_classif(X, y, random_state=42)
        self.feature_scores_ = dict(zip(X.columns, scores))

        if self.k_features:
            # 选择得分最高的k个特征
            sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
            self.selected_features_ = [f[0] for f in sorted_features[:self.k_features]]
        elif self.threshold:
            self.selected_features_ = [f for f, score in self.feature_scores_.items() if score >= self.threshold]
        else:
            # 选择得分大于中位数的特征
            median_score = np.median(scores)
            self.selected_features_ = [f for f, score in self.feature_scores_.items() if score >= median_score]
