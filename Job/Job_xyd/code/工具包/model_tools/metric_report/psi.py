import pandas as pd
import numpy as np
import warnings


def calculate_psi(base_data, test_data, bins=10, min_sample=10, feature_name=None):
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

    Example:
    --------
    >>> base_data = np.random.normal(0, 1, 1000)
    >>> test_data = np.random.normal(0.2, 1.1, 800)
    >>> psi_value, psi_detail = calculate_psi(base_data, test_data, bins=10)
    >>> print(f"PSI值: {psi_value:.4f}")
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
            'bin': base_counts.index.astype(str),
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


def interpret_psi(psi_value):
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


def calculate_multi_feature_psi(base_df, test_df, features=None, bins=10, min_sample=10):
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


def _get_stability_level(psi_value):
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


def generate_psi_report(psi_summary, threshold=0.25):
    """
    生成PSI监控报告

    Parameters:
    -----------
    psi_summary : pd.DataFrame
        PSI汇总结果
    threshold : float, default=0.25
        PSI阈值

    Returns:
    --------
    report : dict
        包含报告内容的字典
    """
    unstable_features = psi_summary[psi_summary['psi_value'] >= threshold]
    stable_features = psi_summary[psi_summary['psi_value'] < threshold]

    report = {
        'summary': {
            'total_features': len(psi_summary),
            'stable_features': len(stable_features),
            'unstable_features': len(unstable_features),
            'stability_rate': len(stable_features) / len(psi_summary) if len(psi_summary) > 0 else 0
        },
        'unstable_features': unstable_features.to_dict('records') if len(unstable_features) > 0 else [],
        'recommendations': []
    }

    # 生成建议
    if len(unstable_features) == 0:
        report['recommendations'].append("所有特征都保持稳定，模型可以继续使用")
    elif len(unstable_features) <= 3:
        report['recommendations'].append("少数特征不稳定，建议监控并考虑重新训练")
        report['recommendations'].extend([f"关注特征: {f['feature']} (PSI={f['psi_value']:.3f})"
                                        for f in unstable_features.to_dict('records')])
    else:
        report['recommendations'].append("多个特征不稳定，强烈建议重新建模")

    return report