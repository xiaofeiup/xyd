"""
按渠道分时段计算特征 PSI（群体稳定性指标）
================================================
功能说明：
- 按 partner_code（渠道）分组
- 在每个渠道内，将 apply_date（格式 YYYYMMDD）按时间中位数分为前后两段
- 计算每个连续/离散特征在前后两段之间的 PSI 值
- 输出每个渠道 × 每个特征的 PSI 结果表

PSI 判定标准（行业通用）：
  PSI < 0.1：稳定，无显著漂移
  0.1 ≤ PSI < 0.25：轻微漂移，需关注
  PSI ≥ 0.25：显著漂移，需排查
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# 核心函数
# ============================================================

def calculate_psi_by_channel(
    df: pd.DataFrame,
    feature_cols: List[str],
    channel_col: str = 'partner_code',
    date_col: str = 'apply_date',
    n_bins: int = 10,
    date_format: str = '%Y%m%d',
    min_samples_per_bin: int = 5,
) -> pd.DataFrame:
    """
    按渠道分组、按时间前后分两段，计算每个特征的 PSI。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据，需包含特征列、渠道列、日期列
    feature_cols : List[str]
        需要计算 PSI 的特征列名列表
    channel_col : str
        渠道列名，默认 'partner_code'
    date_col : str
        日期列名，默认 'apply_date'，格式为 YYYYMMDD 整数或字符串
    n_bins : int
      连续变量分箱数，默认 10
    date_format : str
      日期解析格式，默认 '%Y%m%d'
    min_samples_per_bin : int
      每箱最小样本数阈值（低于此值的箱会合并），默认 5

    Returns
    -------
    pd.DataFrame
        结果表，包含以下列：
        - partner_code : 渠道
        - feature : 特征名
        - psi : PSI 值
        - stability : 稳定性判定 (stable / warning / drift)
        - n_train : 前期样本量
        - n_test : 后期样本量
        - split_date : 时间分割点
    """

    # ---------- 0. 数据预处理 ----------
    df = df.copy()

    # 确保日期列为 datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format=date_format)

    # 移除日期缺失的行
    n_before = len(df)
    df = df.dropna(subset=[date_col, channel_col])
    if len(df) < n_before:
        print(f"⚠️  已移除 {n_before - len(df)} 行日期或渠道缺失的数据")

    channels = df[channel_col].unique()
    print(f"📊 共 {len(channels)} 个渠道, {len(feature_cols)} 个特征, 总样本量 {len(df)}")

    results = []

    # ---------- 1. 按渠道循环 ----------
    for ch in sorted(channels):
        df_ch = df[df[channel_col] == ch].copy()

        if len(df_ch) < n_bins * min_samples_per_bin * 2:
            print(f"  ⏭️  渠道 [{ch}] 样本量不足 ({len(df_ch)})，跳过")
            continue

        # 按时间中位数分割为前后两段
        split_date = df_ch[date_col].median()
        df_train = df_ch[df_ch[date_col] <= split_date]   # 前期（基准）
        df_test = df_ch[df_ch[date_col] > split_date]     # 后期（对比）

        if len(df_train) < n_bins or len(df_test) < n_bins:
            print(f"  ⏭️  渠道 [{ch}] 分割后某段样本量不足，跳过")
            continue

        # ---------- 2. 对每个特征计算 PSI ----------
        for feat in feature_cols:
            if feat not in df_ch.columns:
                continue

            psi_val = _compute_single_psi(
                df_train[feat].reset_index(drop=True),
                df_test[feat].reset_index(drop=True),
                n_bins=n_bins,
                min_samples=min_samples_per_bin,
                feature_name=feat,
            )

            # 稳定性判定
            if psi_val < 0.1:
                stability = 'stable'
            elif psi_val < 0.25:
                stability = 'warning'
            else:
                stability = 'drift'

            results.append({
                channel_col: ch,
                'feature': feat,
                'psi': round(psi_val, 6),
                'stability': stability,
                'n_train': len(df_train),
                'n_test': len(df_test),
                'split_date': split_date.strftime('%Y-%m-%d'),
            })

    result_df = pd.DataFrame(results)
    return result_df


def _compute_single_psi(
    train_series: pd.Series,
    test_series: pd.Series,
    n_bins: int = 10,
    min_samples: int = 5,
    feature_name: str = '',
) -> float:
    """
    计算单个特征在 train vs test 之间的 PSI。

    分箱策略：
    - 连续变量：基于训练集等频分箱 + 边界微调保证单调性
    - 离散变量（唯一值 ≤ 20）：直接按类别分箱
    - 高基数离散变量：视为连续处理
    """
    train = train_series.dropna()
    test = test_series.dropna()

    if len(train) == 0 or len(test) == 0:
        return np.nan

    n_unique = train.nunique()

    # ----- 离散变量分支 -----
    if n_unique <= 20 and pd.api.types.is_object_dtype(train) or \
       (n_unique <= 20 and not np.issubdtype(train.dtype, np.number)):
        # 按类别分箱
        all_categories = set(train.unique()).set(test.unique())
        cat_counts_train = train.value_counts(normalize=True)
        cat_counts_test = test.value_counts(normalize=True)

        psi_sum = 0.0
        for cat in all_categories:
            p_train = cat_counts_train.get(cat, 0)
            p_test = cat_counts_test.get(cat, 0)

            # 防止 log(0)：给极小值
            p_train = max(p_train, 1e-6)
            p_test = max(p_test, 1e-6)

            psi_sum += (p_test - p_train) * np.log(p_test / p_train)

        return psi_sum

    # ----- 连续变量分支 -----
    try:
        # 基于训练集计算等频分箱边界
        _, bin_edges = pd.qcut(train, q=n_bins, retbins=True, duplicates='drop')
        # 确保边界唯一且有序
        bin_edges = np.unique(bin_edges)

        # 将 -inf / +inf 替换为有限值
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    except Exception:
        # qcut 失败时退化为等宽分箱
        _, bin_edges = pd.cut(train, bins=n_bins, retbins=True, duplicates='drop')
        bin_edges = np.unique(bin_edges)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    # 计算两段在每箱中的占比
    train_binned = pd.cut(train, bins=bin_edges, include_lowest=True)
    test_binned = pd.cut(test, bins=bin_edges, include_lowest=True)

    dist_train = train_binned.value_counts(normalize=True).sort_index()
    dist_test = test_binned.value_counts(normalize=True).sort_index()

    # 对齐索引
    all_bins = dist_train.index.union(dist_test.index)
    dist_train = dist_train.reindex(all_bins, fill_value=0)
    dist_test = dist_test.reindex(all_bins, fill_value=0)

    # 计算 PSI
    psi_sum = 0.0
    for i in range(len(dist_train)):
        p_train = dist_train.iloc[i]
        p_test = dist_test.iloc[i]

        p_train = max(p_train, 1e-6)
        p_test = max(p_test, 1e-6)

        psi_sum += (p_test - p_train) * np.log(p_test / p_train)

    return psi_sum


# ============================================================
# 辅助函数：结果汇总与可视化
# ============================================================

def summarize_psi_results(psi_df: pd.DataFrame) -> Dict:
    """汇总 PSI 结果的统计信息"""

    summary = {
        'total_calculations': len(psi_df),
        'stable_count': int((psi_df['stability'] == 'stable').sum()),
        'warning_count': int((psi_df['stability'] == 'warning').sum()),
        'drift_count': int((psi_df['stability'] == 'drift').sum()),
        'channels_analyzed': psi_df['partner_code'].nunique(),
        'features_analyzed': psi_df['feature'].nunique(),
        'max_psi': psi_df['psi'].max() if len(psi_df) > 0 else None,
        'mean_psi': psi_df['psi'].mean() if len(psi_df) > 0 else None,
        'top_drift_features': (
            psi_df[psi_df['stability'] == 'drift']
            .sort_values('psi', ascending=False)[['partner_code', 'feature', 'psi']]
            .head(10)
            .to_dict('records') if (psi_df['stability'] == 'drift').any() else []
        ),
    }

    return summary


def get_feature_psi_heatmap_data(psi_df: pd.DataFrame) -> pd.DataFrame:
    """
    将长格式转为宽格式，用于绘制热力图（渠道 × 特征）
    空值填充为 NaN
    """
    heatmap = psi_df.pivot(index='feature', columns='partner_code', values='psi')
    return heatmap


def get_channel_psi_summary(psi_df: pd.DataFrame) -> pd.DataFrame:
    """
    按渠道汇总 PSI 统计
    """
    summary = psi_df.groupby('partner_code').agg(
        feature_count=('feature', 'count'),
        mean_psi=('psi', 'mean'),
        max_psi=('psi', 'max'),
        stable_cnt=('stability', lambda x: (x == 'stable').sum()),
        warning_cnt=('stability', lambda x: (x == 'warning').sum()),
        drift_cnt=('stability', lambda x: (x == 'drift').sum()),
        drift_rate=('stability', lambda x: (x == 'drift').mean()),
    ).round(4).reset_index()

    summary.columns = [
        'partner_code', 'feature_count', 'mean_psi', 'max_psi',
        'stable_cnt', 'warning_cnt', 'drift_cnt', 'drift_rate'
    ]

    return summary.sort_values('drift_rate', ascending=False)


# ============================================================
# 使用示例
# ============================================================

if __name__ == '__main__':
    # ===== 示例数据构建（演示用）=====
    np.random.seed(42)
    n = 10000

    demo_df = pd.DataFrame({
        'partner_code': np.random.choice(['CH_A', 'CH_B', 'CH_C'], n, p=[0.4, 0.35, 0.25]),
        'apply_date': pd.date_range('2026-01-01', periods=n, freq='h').strftime('%Y%m%d'),
        'age': np.random.normal(40, 10, n),
        'income': np.random.lognormal(10, 1, n),
        'loan_amount': np.random.exponential(50000, n),
        'credit_score': np.random.randint(300, 850, n),
        'employment_years': np.random.exponential(5, n),
        'dti_ratio': np.random.beta(2, 5, n),
        'inquiry_count_3m': np.random.poisson(2, n),
        'region_type': np.random.choice(['urban', 'suburban', 'rural'], n),
    })

    # 定义特征列（排除标签列和ID列）
    feature_columns = [c for c in demo_df.columns
                       if c not in ['partner_code', 'apply_date']]

    # ===== 执行 PSI 计算 =====
    print("=" * 60)
    print("🔍 开始计算按渠道分时段 PSI")
    print("=" * 60)

    psi_result = calculate_psi_by_channel(
        df=demo_df,
        feature_cols=feature_columns,
        channel_col='partner_code',
        date_col='apply_date',
        n_bins=10,
    )

    # ===== 输出结果 =====
    print("\n" + "=" * 60)
    print("📋 PSI 计算结果（完整表）")
    print("=" * 60)
    print(psi_result.to_string(index=False))

    print("\n" + "-" * 40)
    print("📊 汇总统计")
    print("-" * 40)
    summ = summarize_psi_results(psi_result)
    for k, v in summ.items():
        if k != 'top_drift_features':
            print(f"  {k}: {v}")

    if summ['top_drift_features']:
        print("\n  ⚠️  Top 漂移特征（需要关注）:")
        for rec in summ['top_drift_features']:
            print(f"    渠道={rec['partner_code']}, 特征={rec['feature']}, PSI={rec['psi']:.4f}")

    print("\n" + "-" * 40)
    print("📈 各渠道 PSI 汇总")
    print("-" * 40)
    ch_summary = get_channel_psi_summary(psi_result)
    print(ch_summary.to_string(index=False))

    print("\n" + "-" * 40)
    print("🗺️  热力图数据（渠道 × 特征 PSI）")
    print("-" * 40)
    heatmap_data = get_feature_psi_heatmap_data(psi_result)
    print(heatmap_data.round(4).to_string())

    # ===== 保存结果到 Excel =====
    output_path = '/Users/mayongzhi/WorkBuddy/2026-07-31-15-21-04/psi_report.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        psi_result.to_excel(writer, sheet_name='PSI_Detail', index=False)
        ch_summary.to_excel(writer, sheet_name='Channel_Summary', index=False)
        heatmap_data.to_excel(writer, sheet_name='Heatmap_Data')

    print(f"\n✅ 结果已保存至: {output_path}")
