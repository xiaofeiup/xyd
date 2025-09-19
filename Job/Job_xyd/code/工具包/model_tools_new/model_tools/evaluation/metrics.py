"""
核心指标计算模块

提供AUC、KS、PSI、Lift等关键模型评估指标的计算
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算AUC值

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测概率

    Returns:
    --------
    auc : float
        AUC值
    """
    return roc_auc_score(y_true, y_pred)


def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame, int]:
    """
    计算KS值和相关统计量

    Parameters:
    -----------
    y_true : array-like
        真实标签 (0/1)
    y_prob : array-like
        预测概率

    Returns:
    --------
    ks_value : float
        KS值
    df_ks : pd.DataFrame
        KS统计表
    ks_index : int
        最大KS值对应的索引
    """
    # 创建数据框
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })

    # 按预测概率降序排列
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

    # 计算累积统计
    df['bad'] = df['y_true']
    df['good'] = 1 - df['y_true']

    # 总的good和bad数量
    total_good = df['good'].sum()
    total_bad = df['bad'].sum()

    # 累积计算
    df['cum_good'] = df['good'].cumsum()
    df['cum_bad'] = df['bad'].cumsum()

    # 计算累积率
    df['cum_good_rate'] = df['cum_good'] / total_good  # TPR
    df['cum_bad_rate'] = df['cum_bad'] / total_bad    # FPR

    # 计算KS值
    df['ks'] = df['cum_bad_rate'] - df['cum_good_rate']

    # 找到最大KS值
    ks_value = df['ks'].max()
    ks_index = df['ks'].idxmax()

    return ks_value, df, ks_index


def calculate_lift(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    计算Lift值和相关统计量

    Parameters:
    -----------
    y_true : array-like
        真实标签 (0/1)
    y_prob : array-like
        预测概率
    n_bins : int, default=10
        分箱数量

    Returns:
    --------
    df_lift_summary : pd.DataFrame
        Lift统计表
    df_detail : pd.DataFrame
        详细数据
    baseline_rate : float
        基准正样本率
    """
    # 创建数据框
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })

    # 按预测概率降序排列
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

    # 总体统计
    total_samples = len(df)
    total_positive = df['y_true'].sum()
    baseline_rate = total_positive / total_samples  # 基准正样本率

    # 计算累积统计
    df['cum_positive'] = df['y_true'].cumsum()
    df['cum_samples'] = np.arange(1, len(df) + 1)

    # 计算累积精确率和提升度
    df['cum_precision'] = df['cum_positive'] / df['cum_samples']
    df['cum_lift'] = df['cum_precision'] / baseline_rate

    # 计算召回率
    df['cum_recall'] = df['cum_positive'] / total_positive

    # 按分位数分箱
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop') + 1

    # 计算分箱统计
    lift_summary = []
    for decile in sorted(df['decile'].unique()):
        bin_data = df[df['decile'] == decile]

        bin_stats = {
            '分箱': decile,
            '样本数': len(bin_data),
            '正样本数': bin_data['y_true'].sum(),
            '负样本数': len(bin_data) - bin_data['y_true'].sum(),
            '正样本率': bin_data['y_true'].mean(),
            'Lift': bin_data['y_true'].mean() / baseline_rate if baseline_rate > 0 else 0,
            '概率范围': f"{bin_data['y_prob'].min():.3f}-{bin_data['y_prob'].max():.3f}"
        }
        lift_summary.append(bin_stats)

    df_lift_summary = pd.DataFrame(lift_summary)

    # 计算累积Lift统计
    df_lift_summary['累积样本数'] = df_lift_summary['样本数'].cumsum()
    df_lift_summary['累积正样本数'] = df_lift_summary['正样本数'].cumsum()
    df_lift_summary['累积正样本率'] = df_lift_summary['累积正样本数'] / df_lift_summary['累积样本数']
    df_lift_summary['累积Lift'] = df_lift_summary['累积正样本率'] / baseline_rate
    df_lift_summary['累积召回率'] = df_lift_summary['累积正样本数'] / total_positive

    return df_lift_summary, df, baseline_rate


def calculate_psi(base_data: Union[np.ndarray, pd.Series],
                 test_data: Union[np.ndarray, pd.Series],
                 bins: Union[int, List] = 10,
                 min_sample: int = 10,
                 feature_name: Optional[str] = None) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    计算Population Stability Index (PSI)

    Parameters:
    -----------
    base_data : array-like
        基准数据分布 (基准期)
    test_data : array-like
        测试数据分布 (测试期)
    bins : int or list, default=10
        分箱数量或分箱边界
    min_sample : int, default=10
        每个分箱的最小样本数
    feature_name : str, optional
        特征名称，用于报告

    Returns:
    --------
    psi_value : float
        PSI值
    psi_detail : pd.DataFrame
        PSI计算详情表
    """
    try:
        # 数据预处理
        base_data = pd.Series(base_data).dropna()
        test_data = pd.Series(test_data).dropna()

        if len(base_data) == 0 or len(test_data) == 0:
            warnings.warn("输入数据包含空值或全为空")
            return np.nan, None

        # 确定分箱边界
        if isinstance(bins, int):
            # 基于基准数据确定分位数分箱
            bin_num = min(bins, len(base_data) // min_sample)
            if bin_num < 2:
                warnings.warn(f"样本数量不足，无法进行有效分箱 (需要至少{min_sample*2}个样本)")
                return np.nan, None

            # 计算分位数作为分箱边界
            quantiles = [i / bin_num for i in range(1, bin_num)]
            bin_edges = [-np.inf] + base_data.quantile(quantiles).tolist() + [np.inf]
            bin_edges = sorted(list(set(bin_edges)))  # 去重排序
        else:
            bin_edges = bins

        # 计算各分箱的样本数
        base_counts = pd.cut(base_data, bins=bin_edges, include_lowest=True).value_counts().sort_index()
        test_counts = pd.cut(test_data, bins=bin_edges, include_lowest=True).value_counts().sort_index()

        # 创建详情表
        psi_detail = pd.DataFrame({
            'bin': [str(x) for x in base_counts.index],
            'base_count': base_counts.values,
            'test_count': test_counts.reindex(base_counts.index, fill_value=0).values
        })

        # 计算分布比例
        psi_detail['base_dist'] = psi_detail['base_count'] / len(base_data)
        psi_detail['test_dist'] = psi_detail['test_count'] / len(test_data)

        # PSI计算
        def calculate_sub_psi(row):
            base_dist = row['base_dist']
            test_dist = row['test_dist']

            # 处理0值情况
            if base_dist == 0 and test_dist == 0:
                return 0
            elif base_dist == 0:
                base_dist = 1 / len(base_data)  # 使用最小值平滑
            elif test_dist == 0:
                test_dist = 1 / len(test_data)  # 使用最小值平滑

            return (test_dist - base_dist) * np.log(test_dist / base_dist)

        psi_detail['psi_component'] = psi_detail.apply(calculate_sub_psi, axis=1)
        psi_value = psi_detail['psi_component'].sum()

        # 添加特征名称
        if feature_name:
            psi_detail['feature'] = feature_name

        return psi_value, psi_detail

    except Exception as e:
        warnings.warn(f"PSI计算出错: {str(e)}")
        return np.nan, None


def calculate_multi_feature_psi(base_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               features: Optional[List[str]] = None,
                               bins: int = 10,
                               min_sample: int = 10) -> pd.DataFrame:
    """
    批量计算多个特征的PSI

    Parameters:
    -----------
    base_df : pd.DataFrame
        基准数据
    test_df : pd.DataFrame
        测试数据
    features : list, optional
        要计算PSI的特征列表，如果为None则计算所有数值特征
    bins : int, default=10
        分箱数量
    min_sample : int, default=10
        每个分箱的最小样本数

    Returns:
    --------
    psi_summary : pd.DataFrame
        PSI汇总结果
    """
    if features is None:
        features = base_df.select_dtypes(include=[np.number]).columns.tolist()

    psi_results = []

    for feature in features:
        if feature not in base_df.columns or feature not in test_df.columns:
            warnings.warn(f"特征 {feature} 不存在于数据中")
            continue

        psi_value, _ = calculate_psi(
            base_df[feature],
            test_df[feature],
            bins=bins,
            min_sample=min_sample,
            feature_name=feature
        )

        psi_results.append({
            'feature': feature,
            'psi_value': psi_value,
            'interpretation': interpret_psi(psi_value),
            'stability_level': _get_stability_level(psi_value)
        })

    return pd.DataFrame(psi_results)


def interpret_psi(psi_value: float) -> str:
    """
    解释PSI值的含义

    Parameters:
    -----------
    psi_value : float
        PSI值

    Returns:
    --------
    interpretation : str
        PSI值解释
    """
    if pd.isna(psi_value):
        return "无效值"
    elif psi_value < 0.1:
        return "稳定 (变化很小)"
    elif psi_value < 0.2:
        return "轻微变化"
    elif psi_value < 0.25:
        return "中等变化 (需要关注)"
    else:
        return "显著变化 (需要重新建模)"


def _get_stability_level(psi_value: float) -> int:
    """获取稳定性等级"""
    if pd.isna(psi_value):
        return 0
    elif psi_value < 0.1:
        return 1  # 非常稳定
    elif psi_value < 0.2:
        return 2  # 稳定
    elif psi_value < 0.25:
        return 3  # 需要关注
    else:
        return 4  # 需要重建模型


def calculate_gain(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.Series:
    """
    计算gain

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    n_bins : int, default=10
        分箱数

    Returns:
    --------
    gain : pd.Series
        gain值
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False)
    grouped = df.groupby('bin')
    total_positives = df['y_true'].sum()
    gain = grouped['y_true'].cumsum() / total_positives
    return gain


class ModelEvaluator:
    """
    模型评估器

    提供全面的模型评估功能
    """

    def __init__(self, model_name: str = "model"):
        """
        初始化评估器

        Parameters:
        -----------
        model_name : str, default="model"
            模型名称
        """
        self.model_name = model_name

    def evaluate_binary_classification(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     threshold: float = 0.5) -> Dict:
        """
        二分类模型评估

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        threshold : float, default=0.5
            分类阈值

        Returns:
        --------
        evaluation_result : dict
            评估结果
        """
        # 基础指标
        auc = calculate_auc(y_true, y_pred)
        ks, ks_detail, ks_index = calculate_ks(y_true, y_pred)

        # Lift分析
        lift_summary, lift_detail, baseline_rate = calculate_lift(y_true, y_pred)

        # 混淆矩阵相关指标
        y_pred_binary = (y_pred >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'model_name': self.model_name,
            'basic_metrics': {
                'auc': auc,
                'ks': ks,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'lift_metrics': {
                'baseline_rate': baseline_rate,
                'top_10_lift': lift_summary.iloc[0]['Lift'] if len(lift_summary) > 0 else None,
                'top_20_cumulative_lift': lift_summary.iloc[1]['累积Lift'] if len(lift_summary) > 1 else None
            },
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            },
            'sample_info': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(len(y_true) - np.sum(y_true)),
                'positive_rate': float(np.mean(y_true))
            },
            'detailed_results': {
                'ks_table': ks_detail,
                'lift_table': lift_summary
            }
        }

    def compare_models(self,
                      model_results: List[Dict]) -> pd.DataFrame:
        """
        比较多个模型

        Parameters:
        -----------
        model_results : list
            模型评估结果列表

        Returns:
        --------
        comparison_df : pd.DataFrame
            模型比较结果
        """
        comparison_data = []

        for result in model_results:
            basic_metrics = result['basic_metrics']
            lift_metrics = result['lift_metrics']

            comparison_data.append({
                'model_name': result['model_name'],
                'auc': basic_metrics['auc'],
                'ks': basic_metrics['ks'],
                'precision': basic_metrics['precision'],
                'recall': basic_metrics['recall'],
                'f1_score': basic_metrics['f1_score'],
                'top_10_lift': lift_metrics['top_10_lift'],
                'baseline_rate': lift_metrics['baseline_rate']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('auc', ascending=False)

        return comparison_df