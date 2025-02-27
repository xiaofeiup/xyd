import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_score_metrics(df, score_col, target_col, n_bins=10, method='quantile'):
    """
    计算评分的各项指标，包括WOE、IV、Lift等
    
    参数:
    df: DataFrame, 包含评分和目标变量的数据框
    score_col: str, 评分列名
    target_col: str, 目标变量列名
    n_bins: int, 分箱数量
    method: str, 分箱方法 ('quantile' 或 'equal_width')
    
    返回:
    dict: 包含各项评估指标的字典
    """
    # 复制数据，避免修改原始数据
    df = df.copy()
    
    # 计算分箱边界
    if method == 'quantile':
        bins = pd.qcut(df[score_col], q=n_bins, duplicates='drop', retbins=True)[1]
    else:  # equal_width
        bins = pd.cut(df[score_col], bins=n_bins, retbins=True)[1]
    
    # 进行分箱
    df['score_bin'] = pd.cut(df[score_col], bins=bins, labels=False)
    
    # 计算每个分箱的统计信息
    stats_df = pd.DataFrame()
    
    # 基础统计量
    stats_df['min_score'] = df.groupby('score_bin')[score_col].min()
    stats_df['max_score'] = df.groupby('score_bin')[score_col].max()
    stats_df['count'] = df.groupby('score_bin')[score_col].count()
    stats_df['bad'] = df.groupby('score_bin')[target_col].sum()
    stats_df['good'] = stats_df['count'] - stats_df['bad']
    
    # 计算比率
    total_good = stats_df['good'].sum()
    total_bad = stats_df['bad'].sum()
    
    stats_df['bad_rate'] = stats_df['bad'] / stats_df['count']
    stats_df['good_rate'] = stats_df['good'] / stats_df['count']
    stats_df['total_rate'] = stats_df['count'] / len(df)
    
    # 计算WOE和IV
    stats_df['good_dist'] = stats_df['good'] / total_good
    stats_df['bad_dist'] = stats_df['bad'] / total_bad
    stats_df['woe'] = np.log(stats_df['good_dist'] / stats_df['bad_dist'])
    stats_df['iv_component'] = (stats_df['good_dist'] - stats_df['bad_dist']) * stats_df['woe']
    
    # 计算Lift
    overall_bad_rate = total_bad / (total_good + total_bad)
    stats_df['lift'] = stats_df['bad_rate'] / overall_bad_rate
    
    # 计算KS值
    stats_df['cum_good_pct'] = stats_df['good'].cumsum() / total_good
    stats_df['cum_bad_pct'] = stats_df['bad'].cumsum() / total_bad
    stats_df['ks'] = abs(stats_df['cum_bad_pct'] - stats_df['cum_good_pct'])
    
    # 总IV值
    total_iv = stats_df['iv_component'].sum()
    
    # 计算PSI（需要比较样本）
    # PSI的计算需要两个样本的对比，这里只是展示单个样本的分布
    
    return {
        'stats_df': stats_df,
        'total_iv': total_iv,
        'max_ks': stats_df['ks'].max(),
        'bins': bins
    }

def plot_score_distribution(df, score_col, target_col, bins=50):
    """
    绘制评分分布图
    """
    plt.figure(figsize=(12, 6))
    
    # 分别绘制好坏样本的分布
    for target_value, label in [(1, 'Bad'), (0, 'Good')]:
        subset = df[df[target_col] == target_value][score_col]
        plt.hist(subset, bins=bins, alpha=0.5, label=label, density=True)
    
    plt.title('Score Distribution by Target')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_metrics(stats_df):
    """
    绘制评分的各项指标图表
    """
    # 创建多个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 坏率分布图
    axes[0, 0].plot(range(len(stats_df)), stats_df['bad_rate'], marker='o')
    axes[0, 0].set_title('Bad Rate by Score Bin')
    axes[0, 0].set_xlabel('Score Bin')
    axes[0, 0].set_ylabel('Bad Rate')
    axes[0, 0].grid(True)
    
    # 2. WOE图
    axes[0, 1].plot(range(len(stats_df)), stats_df['woe'], marker='o')
    axes[0, 1].set_title('WOE by Score Bin')
    axes[0, 1].set_xlabel('Score Bin')
    axes[0, 1].set_ylabel('WOE')
    axes[0, 1].grid(True)
    
    # 3. Lift图
    axes[1, 0].plot(range(len(stats_df)), stats_df['lift'], marker='o')
    axes[1, 0].set_title('Lift by Score Bin')
    axes[1, 0].set_xlabel('Score Bin')
    axes[1, 0].set_ylabel('Lift')
    axes[1, 0].grid(True)
    
    # 4. KS曲线
    axes[1, 1].plot(range(len(stats_df)), stats_df['cum_good_pct'], label='Cumulative Good%')
    axes[1, 1].plot(range(len(stats_df)), stats_df['cum_bad_pct'], label='Cumulative Bad%')
    axes[1, 1].plot(range(len(stats_df)), stats_df['ks'], label='KS', linestyle='--')
    axes[1, 1].set_title('KS Curve')
    axes[1, 1].set_xlabel('Score Bin')
    axes[1, 1].set_ylabel('Cumulative %')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_score_model(df, score_col, target_col, n_bins=10, method='quantile'):
    """
    评估评分模型的主函数
    """
    # 1. 计算各项指标
    results = calculate_score_metrics(df, score_col, target_col, n_bins, method)
    
    # 2. 打印主要指标
    print("\n=== 模型评估指标 ===")
    print(f"总体IV值: {results['total_iv']:.4f}")
    print(f"最大KS值: {results['max_ks']:.4f}")
    
    # 3. 显示详细统计信息
    print("\n=== 分箱统计信息 ===")
    print(results['stats_df'])
    
    # 4. 绘制分布图
    plot_score_distribution(df, score_col, target_col)
    
    # 5. 绘制评估指标图
    plot_metrics(results['stats_df'])
    
    return results

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成示例评分（正态分布）
    scores = np.random.normal(600, 50, n_samples)
    
    # 生成目标变量（违约概率随评分降低而增加）
    probs = 1 / (1 + np.exp((scores - 600) / 50))
    targets = np.random.binomial(1, probs)
    
    # 创建数据框
    df = pd.DataFrame({
        'score': scores,
        'target': targets
    })
    
    # 评估模型
    results = evaluate_score_model(df, 'score', 'target', n_bins=10) 