"""
数据处理工具模块

提供常用的数据预处理和数据质量检查功能
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


def check_data_quality(data: pd.DataFrame,
                      target_col: Optional[str] = None) -> Dict:
    """
    检查数据质量

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    target_col : str, optional
        目标变量列名

    Returns:
    --------
    quality_report : dict
        数据质量报告
    """
    report = {
        'basic_info': {},
        'missing_analysis': {},
        'data_types': {},
        'duplicates': {}
    }

    # 基础信息
    report['basic_info'] = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
    }

    # 缺失值分析
    missing_stats = []
    for col in data.columns:
        missing_count = data[col].isna().sum()
        missing_ratio = missing_count / len(data)

        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_ratio': missing_ratio,
            'data_type': str(data[col].dtype)
        })

    report['missing_analysis'] = {
        'columns_with_missing': sum(1 for stat in missing_stats if stat['missing_count'] > 0),
        'total_missing_values': sum(stat['missing_count'] for stat in missing_stats),
        'details': missing_stats
    }

    # 数据类型分析
    type_counts = data.dtypes.value_counts().to_dict()
    report['data_types'] = {str(k): int(v) for k, v in type_counts.items()}

    # 重复值分析
    report['duplicates'] = {
        'duplicate_rows': data.duplicated().sum(),
        'unique_rows': len(data.drop_duplicates())
    }

    # 目标变量分析（如果提供）
    if target_col and target_col in data.columns:
        target_series = data[target_col]
        value_counts = target_series.value_counts()

        report['target_analysis'] = {
            'target_column': target_col,
            'unique_values': target_series.nunique(),
            'missing_values': target_series.isna().sum(),
            'value_distribution': value_counts.to_dict(),
            'is_binary': len(value_counts) == 2
        }

        if len(value_counts) == 2:
            # 二分类目标变量的不平衡分析
            positive_class = value_counts.index[0]
            positive_count = value_counts.iloc[0]
            negative_count = value_counts.iloc[1]

            report['target_analysis']['class_balance'] = {
                'positive_class': positive_class,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_ratio': positive_count / (positive_count + negative_count),
                'imbalance_ratio': max(positive_count, negative_count) / min(positive_count, negative_count)
            }

    return report


def clean_column_names(data: pd.DataFrame,
                      remove_special_chars: bool = True,
                      to_lowercase: bool = True) -> pd.DataFrame:
    """
    清理列名

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    remove_special_chars : bool, default=True
        是否移除特殊字符
    to_lowercase : bool, default=True
        是否转换为小写

    Returns:
    --------
    data_cleaned : pd.DataFrame
        清理后的数据
    """
    data_cleaned = data.copy()

    new_columns = []
    for idx, col in enumerate(data.columns):
        new_col = str(col)

        if remove_special_chars:
            # 保留字母、数字和下划线
            import re
            new_col = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', new_col)

        if to_lowercase:
            new_col = new_col.lower()

        new_col = new_col.strip('_')

        if not new_col:
            new_col = f"column_{idx}"

        # 确保列名不以数字开头
        if new_col[0].isdigit():
            new_col = f"col_{new_col}"

        new_columns.append(new_col)

    data_cleaned.columns = new_columns

    return data_cleaned


def handle_missing_values(data: pd.DataFrame,
                         strategy: Dict[str, Union[str, float]] = None,
                         default_strategy: str = 'median') -> pd.DataFrame:
    """
    处理缺失值

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    strategy : dict, optional
        各列的处理策略，如 {'col1': 'mean', 'col2': 0, 'col3': 'mode'}
    default_strategy : str, default='median'
        默认策略 ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')

    Returns:
    --------
    data_filled : pd.DataFrame
        处理后的数据
    """
    data_filled = data.copy()
    strategy = strategy or {}

    for col in data.columns:
        if data[col].isna().sum() == 0:
            continue  # 没有缺失值，跳过

        # 确定处理策略
        col_strategy = strategy.get(col, default_strategy)

        if isinstance(col_strategy, (int, float)):
            # 用指定值填充
            data_filled[col] = data_filled[col].fillna(col_strategy)

        elif col_strategy == 'mean':
            if pd.api.types.is_numeric_dtype(data[col]):
                data_filled[col] = data_filled[col].fillna(data[col].mean())
            else:
                warnings.warn(f"列 {col} 不是数值类型，无法使用均值填充，改用众数")
                data_filled[col] = data_filled[col].fillna(data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'unknown')

        elif col_strategy == 'median':
            if pd.api.types.is_numeric_dtype(data[col]):
                data_filled[col] = data_filled[col].fillna(data[col].median())
            else:
                warnings.warn(f"列 {col} 不是数值类型，无法使用中位数填充，改用众数")
                data_filled[col] = data_filled[col].fillna(data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'unknown')

        elif col_strategy == 'mode':
            mode_value = data[col].mode()
            if len(mode_value) > 0:
                data_filled[col] = data_filled[col].fillna(mode_value.iloc[0])
            else:
                data_filled[col] = data_filled[col].fillna('unknown')

        elif col_strategy == 'forward_fill':
            data_filled[col] = data_filled[col].fillna(method='ffill')

        elif col_strategy == 'backward_fill':
            data_filled[col] = data_filled[col].fillna(method='bfill')

        elif col_strategy == 'drop':
            # 这个策略在整体处理后执行
            continue

        else:
            warnings.warn(f"未知的处理策略: {col_strategy}，使用默认策略")
            if pd.api.types.is_numeric_dtype(data[col]):
                data_filled[col] = data_filled[col].fillna(data[col].median())
            else:
                data_filled[col] = data_filled[col].fillna(data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'unknown')

    # 处理drop策略
    drop_cols = [col for col, strat in strategy.items() if strat == 'drop']
    if drop_cols:
        data_filled = data_filled.dropna(subset=drop_cols)

    return data_filled


def detect_outliers(data: pd.DataFrame,
                   method: str = 'iqr',
                   threshold: float = 1.5,
                   columns: Optional[List[str]] = None) -> Dict:
    """
    检测异常值

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    method : str, default='iqr'
        检测方法 ('iqr', 'zscore', 'isolation_forest')
    threshold : float, default=1.5
        阈值参数
    columns : list, optional
        要检测的列，如果为None则检测所有数值列

    Returns:
    --------
    outlier_report : dict
        异常值检测报告
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    outlier_report = {
        'method': method,
        'threshold': threshold,
        'results': {}
    }

    for col in columns:
        if col not in data.columns:
            continue

        col_data = data[col].dropna()

        if len(col_data) == 0:
            continue

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

            outlier_report['results'][col] = {
                'outlier_count': len(outliers),
                'outlier_ratio': len(outliers) / len(data),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(col_data))
            outliers = data[np.abs(stats.zscore(data[col].fillna(col_data.mean()))) > threshold]

            outlier_report['results'][col] = {
                'outlier_count': len(outliers),
                'outlier_ratio': len(outliers) / len(data),
                'threshold_zscore': threshold,
                'outlier_indices': outliers.index.tolist()
            }

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest

                iso_forest = IsolationForest(contamination=threshold, random_state=42)
                outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))

                outlier_indices = col_data.index[outlier_labels == -1]
                outliers = data.loc[outlier_indices]

                outlier_report['results'][col] = {
                    'outlier_count': len(outliers),
                    'outlier_ratio': len(outliers) / len(data),
                    'contamination': threshold,
                    'outlier_indices': outlier_indices.tolist()
                }

            except ImportError:
                warnings.warn("Isolation Forest需要安装scikit-learn")
                continue

    return outlier_report


def remove_outliers(data: pd.DataFrame,
                   outlier_report: Dict,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    移除异常值

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    outlier_report : dict
        异常值检测报告
    columns : list, optional
        要处理的列

    Returns:
    --------
    data_clean : pd.DataFrame
        移除异常值后的数据
    """
    if columns is None:
        columns = list(outlier_report['results'].keys())

    outlier_indices = set()

    for col in columns:
        if col in outlier_report['results']:
            col_outliers = outlier_report['results'][col]['outlier_indices']
            outlier_indices.update(col_outliers)

    # 移除异常值
    data_clean = data.drop(index=list(outlier_indices))

    print(f"移除了 {len(outlier_indices)} 个异常值，剩余 {len(data_clean)} 行数据")

    return data_clean


def split_features_target(data: pd.DataFrame,
                         target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    分离特征和目标变量

    Parameters:
    -----------
    data : pd.DataFrame
        完整数据
    target_col : str
        目标变量列名

    Returns:
    --------
    X : pd.DataFrame
        特征数据
    y : pd.Series
        目标变量
    """
    if target_col not in data.columns:
        raise ValueError(f"目标变量 {target_col} 不存在于数据中")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    return X, y


def sample_data(data: pd.DataFrame,
               method: str = 'random',
               n_samples: Optional[int] = None,
               frac: Optional[float] = None,
               stratify_col: Optional[str] = None,
               random_state: int = 42) -> pd.DataFrame:
    """
    数据采样

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    method : str, default='random'
        采样方法 ('random', 'stratified')
    n_samples : int, optional
        采样数量
    frac : float, optional
        采样比例
    stratify_col : str, optional
        分层采样的列名
    random_state : int, default=42
        随机种子

    Returns:
    --------
    sampled_data : pd.DataFrame
        采样后的数据
    """
    if n_samples is None and frac is None:
        raise ValueError("必须指定n_samples或frac参数")

    if method == 'random':
        return data.sample(n=n_samples, frac=frac, random_state=random_state)

    elif method == 'stratified':
        if stratify_col is None:
            raise ValueError("分层采样需要指定stratify_col参数")

        if stratify_col not in data.columns:
            raise ValueError(f"分层列 {stratify_col} 不存在于数据中")

        sampled_groups = []

        for group_value in data[stratify_col].unique():
            group_data = data[data[stratify_col] == group_value]

            if frac is not None:
                group_sample = group_data.sample(frac=frac, random_state=random_state)
            else:
                # 按比例计算每组的采样数量
                group_size = int(n_samples * len(group_data) / len(data))
                group_size = max(1, min(group_size, len(group_data)))  # 至少采样1个，不超过组大小
                group_sample = group_data.sample(n=group_size, random_state=random_state)

            sampled_groups.append(group_sample)

        return pd.concat(sampled_groups, ignore_index=True)

    else:
        raise ValueError(f"不支持的采样方法: {method}")


def save_data_report(data: pd.DataFrame,
                    filepath: str,
                    target_col: Optional[str] = None) -> None:
    """
    保存数据报告

    Parameters:
    -----------
    data : pd.DataFrame
        数据
    filepath : str
        保存路径
    target_col : str, optional
        目标变量列名
    """
    # 生成数据质量报告
    quality_report = check_data_quality(data, target_col)

    # 生成描述性统计
    numeric_desc = data.select_dtypes(include=[np.number]).describe()
    categorical_desc = data.select_dtypes(exclude=[np.number]).describe()

    # 保存到Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 数据质量报告
        quality_df = pd.DataFrame([quality_report['basic_info']])
        quality_df.to_excel(writer, sheet_name='basic_info', index=False)

        # 缺失值详情
        missing_df = pd.DataFrame(quality_report['missing_analysis']['details'])
        missing_df.to_excel(writer, sheet_name='missing_analysis', index=False)

        # 描述性统计
        if not numeric_desc.empty:
            numeric_desc.to_excel(writer, sheet_name='numeric_statistics')

        if not categorical_desc.empty:
            categorical_desc.to_excel(writer, sheet_name='categorical_statistics')

        # 目标变量分析
        if quality_report['target_analysis']:
            target_df = pd.DataFrame([quality_report['target_analysis']])
            target_df.to_excel(writer, sheet_name='target_analysis', index=False)

    print(f"数据报告已保存到: {filepath}")
