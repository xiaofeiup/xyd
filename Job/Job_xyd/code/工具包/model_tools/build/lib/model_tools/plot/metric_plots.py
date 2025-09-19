#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估绘图函数
包含Lift、Gain、KS等模型评估相关的绘图功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams

# 从gentools模块导入评估函数
from ..gentools import calculate_ks, calculate_lift

# 设置中文字体支持
rcParams["font.sans-serif"] = [
    "SimHei",  # 黑体（Windows/Linux 常见）
    "Microsoft YaHei",  # 微软雅黑（Windows）
    "Arial Unicode MS",  # macOS 常见
    "STHeiti",  # macOS 系统中文字体
]
# 正常显示负号
rcParams["axes.unicode_minus"] = False

def plot_ks_curve(y_true, y_prob, title="KS曲线分析"):
    """
    绘制KS曲线
    
    Parameters:
    y_true: 真实标签
    y_prob: 预测概率
    title: 图表标题
    """
    # 计算KS统计量
    ks_value, df_ks, ks_index = calculate_ks(y_true, y_prob)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制KS曲线
    x_axis = np.arange(len(df_ks)) / len(df_ks)
    
    ax1.plot(x_axis, df_ks['cum_good_rate'], label='累积好样本率 (TPR)', 
             color='blue', linewidth=2)
    ax1.plot(x_axis, df_ks['cum_bad_rate'], label='累积坏样本率 (FPR)', 
             color='red', linewidth=2)
    ax1.plot(x_axis, df_ks['ks'], label='KS曲线', 
             color='green', linewidth=2, linestyle='--')
    
    # 标记最大KS点
    max_ks_x = ks_index / len(df_ks)
    ax1.axvline(x=max_ks_x, color='orange', linestyle=':', alpha=0.7)
    ax1.text(max_ks_x + 0.01, ks_value/2, f'Max KS = {ks_value:.3f}', 
             fontsize=12, color='orange', weight='bold')
    
    ax1.set_xlabel('样本百分比')
    ax1.set_ylabel('累积率')
    ax1.set_title(f'{title}\nKS值: {ks_value:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制KS分布直方图
    ax2.hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='好样本', 
             color='blue', density=True)
    ax2.hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='坏样本', 
             color='red', density=True)
    ax2.set_xlabel('预测概率')
    ax2.set_ylabel('密度')
    ax2.set_title('样本概率分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ks_value, df_ks

def plot_detailed_ks_analysis(y_true, y_prob):
    """
    详细的KS分析图表
    """
    ks_value, df_ks, ks_index = calculate_ks(y_true, y_prob)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'KS曲线详细分析 (KS = {ks_value:.3f})', fontsize=16, weight='bold')
    
    # 1. KS曲线
    x_axis = np.arange(len(df_ks)) / len(df_ks)
    axes[0,0].plot(x_axis, df_ks['cum_good_rate'], 'b-', label='累积好样本率')
    axes[0,0].plot(x_axis, df_ks['cum_bad_rate'], 'r-', label='累积坏样本率')
    axes[0,0].fill_between(x_axis, df_ks['cum_good_rate'], df_ks['cum_bad_rate'], 
                          alpha=0.3, color='green')
    max_ks_x = ks_index / len(df_ks)
    axes[0,0].axvline(x=max_ks_x, color='orange', linestyle='--', alpha=0.8)
    axes[0,0].set_title('KS曲线')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. KS值变化
    axes[0,1].plot(x_axis, df_ks['ks'], 'g-', linewidth=2)
    axes[0,1].axhline(y=ks_value, color='orange', linestyle='--', alpha=0.8)
    axes[0,1].axvline(x=max_ks_x, color='orange', linestyle='--', alpha=0.8)
    axes[0,1].scatter([max_ks_x], [ks_value], color='red', s=100, zorder=5)
    axes[0,1].set_title(f'KS值变化 (最大值: {ks_value:.3f})')
    axes[0,1].set_ylabel('KS值')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. ROC曲线对比
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1,0].plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1,0].set_xlabel('假正率 (FPR)')
    axes[1,0].set_ylabel('真正率 (TPR)')
    axes[1,0].set_title('ROC曲线')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 概率分布对比
    axes[1,1].hist(y_prob[y_true == 0], bins=30, alpha=0.6, label='好样本', 
                   color='blue', density=True)
    axes[1,1].hist(y_prob[y_true == 1], bins=30, alpha=0.6, label='坏样本', 
                   color='red', density=True)
    axes[1,1].set_xlabel('预测概率')
    axes[1,1].set_ylabel('密度')
    axes[1,1].set_title('样本概率分布')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出KS统计摘要
    print("="*50)
    print("KS曲线分析摘要")
    print("="*50)
    print(f"KS值: {ks_value:.4f}")
    print(f"最优切分点位置: {ks_index/len(df_ks):.2%}")
    print(f"最优切分点概率: {df_ks.iloc[ks_index]['y_prob']:.4f}")
    print(f"AUC值: {roc_auc:.4f}")
    
    # KS值评价
    if ks_value >= 0.4:
        评价 = "优秀"
    elif ks_value >= 0.3:
        评价 = "良好"
    elif ks_value >= 0.2:
        评价 = "一般"
    else:
        评价 = "较差"
    
    print(f"模型区分能力: {评价}")
    print("="*50)


def plot_lift_gain(y_true, y_pred, ax_lift=None, ax_gain=None, n_bins=10, label=None):
    """
    绘制lift和gain曲线
    :param y_true: 真实值，list
    :param y_pred: 预测值，list
    :param ax_lift: 绘制lift曲线的ax，matplotlib.axes.Axes
    :param ax_gain: 绘制gain曲线的ax，matplotlib.axes.Axes
    :param n_bins: 分箱数，int
    """

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False)
    grouped = df.groupby('bin')
    total_positives = df['y_true'].sum()
    lift = grouped['y_true'].sum() / (total_positives / n_bins)
    gain = grouped['y_true'].cumsum() / total_positives
    if ax_lift is not None:
        ax_lift.plot(lift, label=label)
        ax_lift.set_title('Lift')
        ax_lift.set_xlabel('Bin')
        ax_lift.set_ylabel('Lift')  
        ax_lift.legend()
    if ax_gain is not None:
        ax_gain.plot(gain, label=label)
        ax_gain.set_title('Gain')
        ax_gain.set_xlabel('Bin')
        ax_gain.set_ylabel('Gain')
        ax_gain.legend()
    return lift, gain


def plot_lift_curve(y_true, y_prob, n_bins=10, title="Lift曲线分析"):
    """
    绘制Lift曲线
    
    Parameters:
    y_true: 真实标签
    y_prob: 预测概率
    n_bins: 分箱数量
    title: 图表标题
    """
    # 计算Lift统计量
    df_lift_summary, df_detail, baseline_rate = calculate_lift(y_true, y_prob, n_bins)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{title}', fontsize=16, weight='bold')
    
    # 1. 累积Lift曲线
    x_axis = df_lift_summary['累积召回率']
    axes[0, 0].plot(x_axis, df_lift_summary['累积Lift'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线 (Lift=1)')
    axes[0, 0].set_xlabel('累积召回率')
    axes[0, 0].set_ylabel('累积Lift')
    axes[0, 0].set_title('累积Lift曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 分箱Lift柱状图
    axes[0, 1].bar(df_lift_summary['分箱'], df_lift_summary['Lift'], 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线 (Lift=1)')
    axes[0, 1].set_xlabel('分箱')
    axes[0, 1].set_ylabel('Lift值')
    axes[0, 1].set_title('各分箱Lift值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 增益图 (Gain Chart)
    random_line = np.linspace(0, 1, len(df_lift_summary))
    axes[1, 0].plot(df_lift_summary['累积召回率'], df_lift_summary['累积召回率'], 
                    'b-o', linewidth=2, label='模型增益')
    axes[1, 0].plot(random_line, random_line, 'r--', alpha=0.7, label='随机模型')
    axes[1, 0].fill_between(df_lift_summary['累积召回率'], random_line[:len(df_lift_summary)], 
                            df_lift_summary['累积召回率'], alpha=0.3, color='green', label='增益面积')
    axes[1, 0].set_xlabel('样本比例')
    axes[1, 0].set_ylabel('捕获正样本比例')
    axes[1, 0].set_title('增益图 (Gain Chart)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 正样本率分布
    axes[1, 1].bar(df_lift_summary['分箱'], df_lift_summary['正样本率'], 
                   color='lightcoral', alpha=0.7, edgecolor='darkred')
    axes[1, 1].axhline(y=baseline_rate, color='blue', linestyle='--', alpha=0.7, 
                       label=f'基准正样本率 ({baseline_rate:.3f})')
    axes[1, 1].set_xlabel('分箱')
    axes[1, 1].set_ylabel('正样本率')
    axes[1, 1].set_title('各分箱正样本率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_lift_summary


def plot_combined_ks_lift_analysis(y_true, y_prob, n_bins=10):
    """
    综合KS和Lift分析
    """
    # 计算KS和Lift
    ks_value, df_ks, ks_index = calculate_ks(y_true, y_prob)
    df_lift_summary, df_detail, baseline_rate = calculate_lift(y_true, y_prob, n_bins)
    
    # 创建综合图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'模型性能综合分析 (KS={ks_value:.3f})', fontsize=16, weight='bold')
    
    # 1. KS曲线
    x_axis = np.arange(len(df_ks)) / len(df_ks)
    axes[0, 0].plot(x_axis, df_ks['cum_good_rate'], 'b-', label='累积好样本率')
    axes[0, 0].plot(x_axis, df_ks['cum_bad_rate'], 'r-', label='累积坏样本率')
    axes[0, 0].fill_between(x_axis, df_ks['cum_good_rate'], df_ks['cum_bad_rate'], 
                            alpha=0.3, color='green')
    max_ks_x = ks_index / len(df_ks)
    axes[0, 0].axvline(x=max_ks_x, color='orange', linestyle='--', alpha=0.8)
    axes[0, 0].set_title(f'KS曲线 (KS={ks_value:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Lift曲线
    axes[0, 1].plot(df_lift_summary['累积召回率'], df_lift_summary['累积Lift'], 
                    'b-o', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('累积召回率')
    axes[0, 1].set_ylabel('累积Lift')
    axes[0, 1].set_title('累积Lift曲线')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 2].plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('假正率')
    axes[0, 2].set_ylabel('真正率')
    axes[0, 2].set_title('ROC曲线')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 分箱Lift
    axes[1, 0].bar(df_lift_summary['分箱'], df_lift_summary['Lift'], 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('分箱')
    axes[1, 0].set_ylabel('Lift值')
    axes[1, 0].set_title('各分箱Lift值')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 概率分布
    axes[1, 1].hist(y_prob[y_true == 0], bins=30, alpha=0.6, label='负样本', 
                    color='blue', density=True)
    axes[1, 1].hist(y_prob[y_true == 1], bins=30, alpha=0.6, label='正样本', 
                    color='red', density=True)
    axes[1, 1].set_xlabel('预测概率')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('样本概率分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 增益图
    random_line = np.linspace(0, 1, len(df_lift_summary))
    axes[1, 2].plot(df_lift_summary['累积召回率'], df_lift_summary['累积召回率'], 
                    'b-o', linewidth=2, label='模型增益')
    axes[1, 2].plot(random_line, random_line, 'r--', alpha=0.7, label='随机模型')
    axes[1, 2].fill_between(df_lift_summary['累积召回率'], random_line[:len(df_lift_summary)], 
                            df_lift_summary['累积召回率'], alpha=0.3, color='green')
    axes[1, 2].set_xlabel('样本比例')
    axes[1, 2].set_ylabel('捕获正样本比例')
    axes[1, 2].set_title('增益图')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ks_value, df_lift_summary


def metric_report_plot(y_true_dict, y_pred_dict, n_bins=10, plot=True):
    """生成模型评估指标并可视化。

    参数说明
    ----------
    y_true_dict / y_pred_dict : dict
        key 为样本类型（如 ``train`` / ``test`` / ``oot``），value 为对应的 ``y_true`` / ``y_pred``。
    n_bins : int, default 10
        lift / gain 分箱数量。
    plot : bool, default True
        是否绘制 ROC、KS、Lift、Gain 曲线。
    """

    if plot:
        for key in y_true_dict.keys():
            y_true = y_true_dict[key]
            y_pred = y_pred_dict[key]
            print(f"绘制{key}的KS和Lift曲线...")
            plot_combined_ks_lift_analysis(y_true, y_pred, n_bins=n_bins) 