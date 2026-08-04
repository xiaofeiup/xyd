"""
核心指标计算模块

提供AUC、KS、PSI、Lift等关键模型评估指标的计算
"""

import pandas as pd
import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, median_absolute_error,
    max_error, mean_absolute_percentage_error
)


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
        if not 0 <= threshold <= 1:
            raise ValueError("threshold 必须在 [0, 1] 范围内")

        if len(y_true) == 0:
            raise ValueError("y_true 不能为空")

        if len(y_true) != len(y_pred):
            raise ValueError("y_true 与 y_pred 的长度必须一致")

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
                'lift_table': lift_summary,
                'lift_detail': lift_detail
            },
            'threshold': threshold,
            'ks_index': int(ks_index)
        }

    def calculate_lift_analysis(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               n_bins: int = 10) -> Dict[str, Any]:
        """计算 Lift 分析结果。"""
        lift_summary, lift_detail, baseline_rate = calculate_lift(
            y_true, y_pred, n_bins=n_bins
        )

        top_decile_lift = lift_summary.iloc[0]['Lift'] if len(lift_summary) > 0 else None
        cumulative_lift = (
            lift_summary.iloc[-1]['累积Lift']
            if len(lift_summary) > 0 and '累积Lift' in lift_summary.columns
            else None
        )

        return {
            'lift_table': lift_summary,
            'detail': lift_detail,
            'baseline_rate': baseline_rate,
            'top_decile_lift': top_decile_lift,
            'cumulative_lift': cumulative_lift,
        }

    def calculate_ks_table(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           n_bins: int = 10) -> pd.DataFrame:
        """基于分箱生成 KS 表格。"""
        _, ks_detail, _ = calculate_ks(y_true, y_pred)
        ks_detail = ks_detail.copy()

        bins = min(n_bins, len(ks_detail))
        ks_detail['decile'] = pd.qcut(
            np.arange(len(ks_detail)),
            q=bins,
            labels=False,
            duplicates='drop'
        ) + 1

        grouped = ks_detail.groupby('decile').agg({
            'cum_good_rate': 'max',
            'cum_bad_rate': 'max',
            'ks': 'max',
            'y_true': 'count'
        }).rename(columns={'y_true': 'samples'})

        grouped['decile'] = grouped.index

        return grouped.reset_index(drop=True)

    def get_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """汇总评估结果，输出关键指标与建议。"""
        if not evaluation_result:
            raise ValueError("evaluation_result 不能为空")

        basic = evaluation_result.get('basic_metrics', {})
        lift_metrics = evaluation_result.get('lift_metrics', {})

        auc = basic.get('auc', float('nan'))
        ks_value = basic.get('ks', float('nan'))
        precision = basic.get('precision', float('nan'))
        recall = basic.get('recall', float('nan'))

        performance_level = '良好'
        recommendations: List[str] = []

        if auc < 0.6 or ks_value < 0.2:
            performance_level = '需改进'
            recommendations.append('AUC 或 KS 偏低，建议重新评估特征与模型结构。')
        elif auc < 0.7 or ks_value < 0.3:
            performance_level = '一般'
            recommendations.append('模型表现中等，可尝试超参数调优提升效果。')
        else:
            recommendations.append('模型表现稳定，请持续关注数据与特征漂移情况。')

        if precision < 0.5:
            recommendations.append('精确率偏低，建议调整阈值或引入成本敏感策略。')
        if recall < 0.5:
            recommendations.append('召回率偏低，可适当降低阈值或补充区分度更高的特征。')

        return {
            'model_name': evaluation_result.get('model_name', self.model_name),
            'model_performance': performance_level,
            'key_metrics': {
                'auc': auc,
                'ks': ks_value,
                'precision': precision,
                'recall': recall,
                'top_decile_lift': lift_metrics.get('top_10_lift')
            },
            'recommendations': recommendations
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


# ================== 回归模型评估指标 ==================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Square Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    rmse : float
        RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 (Mean Square Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    mse : float
        MSE值
    """
    return mean_squared_error(y_true, y_pred)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    mae : float
        MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R-squared)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    r2 : float
        R²值
    """
    return r2_score(y_true, y_pred)


def calculate_adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    计算调整R² (Adjusted R-squared)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    n_features : int
        特征数量

    Returns:
    --------
    adj_r2 : float
        调整R²值
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n <= n_features + 1:
        return np.nan

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    mape : float
        MAPE值 (百分比形式)
    """
    # 避免除零错误
    mask = y_true != 0
    if not np.any(mask):
        warnings.warn("所有真实值为0，无法计算MAPE")
        return np.inf

    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    smape : float
        SMAPE值 (百分比形式)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if not np.any(mask):
        warnings.warn("所有值接近0，无法计算SMAPE")
        return 0.0

    smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    return smape


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根对数误差 (Root Mean Square Logarithmic Error)

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    rmsle : float
        RMSLE值
    """
    # 确保所有值为正数
    if np.any(y_true < 0) or np.any(y_pred < 0):
        warnings.warn("RMSLE要求所有值为非负数")
        return np.nan

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return np.sqrt(mean_squared_error(log_true, log_pred))


def calculate_huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    计算Huber损失

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    delta : float, default=1.0
        阈值参数

    Returns:
    --------
    huber_loss : float
        Huber损失值
    """
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta

    loss = np.where(condition,
                   0.5 * residual**2,
                   delta * residual - 0.5 * delta**2)

    return np.mean(loss)


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """
    计算分位数损失

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    quantile : float, default=0.5
        分位数 (0-1之间)

    Returns:
    --------
    quantile_loss : float
        分位数损失值
    """
    if not 0 <= quantile <= 1:
        raise ValueError("分位数必须在0-1之间")

    residual = y_true - y_pred
    loss = np.where(residual >= 0,
                   quantile * residual,
                   (quantile - 1) * residual)

    return np.mean(loss)


def calculate_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算回归残差统计

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值

    Returns:
    --------
    residual_stats : dict
        残差统计信息
    """
    residuals = y_true - y_pred

    return {
        'residuals': residuals,
        'abs_residuals': np.abs(residuals),
        'squared_residuals': residuals**2,
        'standardized_residuals': residuals / np.std(residuals) if np.std(residuals) != 0 else residuals,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'residual_skewness': _calculate_skewness(residuals),
        'residual_kurtosis': _calculate_kurtosis(residuals)
    }


def _calculate_skewness(data: np.ndarray) -> float:
    """计算偏度"""
    n = len(data)
    if n < 3:
        return np.nan

    mean_val = np.mean(data)
    std_val = np.std(data)

    if std_val == 0:
        return 0.0

    skewness = np.mean(((data - mean_val) / std_val) ** 3)
    return skewness


def _calculate_kurtosis(data: np.ndarray) -> float:
    """计算峰度"""
    n = len(data)
    if n < 4:
        return np.nan

    mean_val = np.mean(data)
    std_val = np.std(data)

    if std_val == 0:
        return 0.0

    kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
    return kurtosis


def evaluate_regression_model(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            n_features: Optional[int] = None,
                            model_name: str = "Regression Model") -> Dict[str, any]:
    """
    回归模型综合评估

    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    n_features : int, optional
        特征数量，用于计算调整R²
    model_name : str, default="Regression Model"
        模型名称

    Returns:
    --------
    evaluation_result : dict
        评估结果字典
    """
    # 基础指标
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    # 高级指标
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)

    # 残差分析
    residual_stats = calculate_regression_residuals(y_true, y_pred)

    # 其他指标
    explained_var = explained_variance_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    # 调整R²（如果提供特征数量）
    adj_r2 = None
    if n_features is not None:
        adj_r2 = calculate_adjusted_r2(y_true, y_pred, n_features)

    result = {
        'model_name': model_name,
        'basic_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adj_r2,
            'explained_variance': explained_var
        },
        'percentage_metrics': {
            'mape': mape,
            'smape': smape
        },
        'robust_metrics': {
            'median_absolute_error': median_ae,
            'max_error': max_err,
            'huber_loss': calculate_huber_loss(y_true, y_pred)
        },
        'residual_analysis': residual_stats,
        'sample_info': {
            'total_samples': len(y_true),
            'y_true_mean': np.mean(y_true),
            'y_true_std': np.std(y_true),
            'y_pred_mean': np.mean(y_pred),
            'y_pred_std': np.std(y_pred),
            'prediction_range': (np.min(y_pred), np.max(y_pred)),
            'actual_range': (np.min(y_true), np.max(y_true))
        }
    }

    return result


def compare_regression_models(model_results: List[Dict]) -> pd.DataFrame:
    """
    比较多个回归模型

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
        percentage_metrics = result['percentage_metrics']

        comparison_data.append({
            'model_name': result['model_name'],
            'rmse': basic_metrics['rmse'],
            'mae': basic_metrics['mae'],
            'r2': basic_metrics['r2'],
            'adjusted_r2': basic_metrics.get('adjusted_r2'),
            'mape': percentage_metrics['mape'],
            'smape': percentage_metrics['smape'],
            'explained_variance': basic_metrics['explained_variance']
        })

    comparison_df = pd.DataFrame(comparison_data)
    # 按R²降序排列
    comparison_df = comparison_df.sort_values('r2', ascending=False)

    return comparison_df
