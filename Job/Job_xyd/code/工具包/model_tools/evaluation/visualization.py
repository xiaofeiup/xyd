"""
可视化模块

提供模型评估和监控的可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ModelVisualizationTool:
    """
    模型可视化工具

    提供模型评估、监控和分析的可视化功能
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        初始化可视化工具

        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            图形大小
        style : str, default='whitegrid'
            图形样式
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)

    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_scores: np.ndarray,
                      title: str = "ROC曲线",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制ROC曲线

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        title : str, default="ROC曲线"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假正率 (FPR)')
        ax.set_ylabel('真正率 (TPR)')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_ks_curve(self,
                     y_true: np.ndarray,
                     y_scores: np.ndarray,
                     title: str = "KS曲线",
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制KS曲线

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        title : str, default="KS曲线"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        # 计算KS曲线
        df = pd.DataFrame({'score': y_scores, 'target': y_true})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        df['cumsum_good'] = (1 - df['target']).cumsum()
        df['cumsum_bad'] = df['target'].cumsum()

        total_good = (1 - df['target']).sum()
        total_bad = df['target'].sum()

        df['tpr'] = df['cumsum_bad'] / total_bad  # True Positive Rate
        df['fpr'] = df['cumsum_good'] / total_good  # False Positive Rate
        df['ks'] = df['tpr'] - df['fpr']

        max_ks = df['ks'].max()
        max_ks_idx = df['ks'].idxmax()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(range(len(df)), df['tpr'], label='累积坏样本率 (TPR)', color='red', linewidth=2)
        ax.plot(range(len(df)), df['fpr'], label='累积好样本率 (FPR)', color='blue', linewidth=2)
        ax.plot(range(len(df)), df['ks'], label='KS曲线', color='green', linewidth=2)

        # 标记最大KS值点
        ax.axvline(x=max_ks_idx, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(y=max_ks, color='orange', linestyle='--', alpha=0.7)
        ax.text(max_ks_idx, max_ks + 0.05, f'Max KS = {max_ks:.3f}',
                horizontalalignment='center', fontsize=12, fontweight='bold')

        ax.set_xlabel('样本排序')
        ax.set_ylabel('累积比例')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_lift_curve(self,
                       y_true: np.ndarray,
                       y_scores: np.ndarray,
                       bins: int = 10,
                       title: str = "Lift曲线",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Lift曲线

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        bins : int, default=10
            分箱数量
        title : str, default="Lift曲线"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        df = pd.DataFrame({'score': y_scores, 'target': y_true})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        # 分箱
        df['bin'] = pd.cut(range(len(df)), bins, labels=False) + 1

        # 计算每个分箱的统计信息
        bin_stats = df.groupby('bin').agg({
            'target': ['count', 'sum', 'mean']
        }).round(4)

        bin_stats.columns = ['total', 'positive', 'positive_rate']
        bin_stats = bin_stats.reset_index()

        # 计算累积统计
        bin_stats['cumulative_positive'] = bin_stats['positive'].cumsum()
        bin_stats['cumulative_total'] = bin_stats['total'].cumsum()
        bin_stats['cumulative_rate'] = bin_stats['cumulative_positive'] / bin_stats['cumulative_total']

        # 计算Lift
        overall_rate = y_true.mean()
        bin_stats['lift'] = bin_stats['positive_rate'] / overall_rate
        bin_stats['cumulative_lift'] = bin_stats['cumulative_rate'] / overall_rate

        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Lift曲线
        ax1.plot(bin_stats['bin'], bin_stats['lift'], 'o-', color='blue', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线 (Lift=1)')
        ax1.set_xlabel('分箱')
        ax1.set_ylabel('Lift值')
        ax1.set_title('分箱Lift值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 累积Lift曲线
        ax2.plot(bin_stats['bin'], bin_stats['cumulative_lift'], 'o-', color='green', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线 (Lift=1)')
        ax2.set_xlabel('分箱')
        ax2.set_ylabel('累积Lift值')
        ax2.set_title('累积Lift值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_score_distribution(self,
                              y_true: np.ndarray,
                              y_scores: np.ndarray,
                              bins: int = 50,
                              title: str = "评分分布",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制评分分布图

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        bins : int, default=50
            分箱数量
        title : str, default="评分分布"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 分别绘制好坏样本的分布
        good_scores = y_scores[y_true == 0]
        bad_scores = y_scores[y_true == 1]

        ax.hist(good_scores, bins=bins, alpha=0.7, label=f'好样本 (n={len(good_scores)})',
                color='blue', density=True)
        ax.hist(bad_scores, bins=bins, alpha=0.7, label=f'坏样本 (n={len(bad_scores)})',
                color='red', density=True)

        ax.set_xlabel('预测概率')
        ax.set_ylabel('密度')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            title: str = "混淆矩阵",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制混淆矩阵

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        labels : list, optional
            标签名称
        title : str, default="混淆矩阵"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            labels = ['负类', '正类']

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self,
                              feature_names: List[str],
                              importance_scores: np.ndarray,
                              top_k: int = 20,
                              title: str = "特征重要性",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制特征重要性图

        Parameters:
        -----------
        feature_names : list
            特征名称
        importance_scores : np.ndarray
            重要性得分
        top_k : int, default=20
            显示前k个特征
        title : str, default="特征重要性"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        # 创建特征重要性DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_k)

        fig, ax = plt.subplots(figsize=self.figsize)

        bars = ax.barh(importance_df['feature'], importance_df['importance'], color='skyblue')

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')

        ax.set_xlabel('重要性得分')
        ax.set_ylabel('特征')
        ax.set_title(f'{title} (Top {top_k})')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_psi_trends(self,
                       psi_data: pd.DataFrame,
                       feature_col: str = 'feature',
                       psi_col: str = 'psi_value',
                       time_col: str = 'timestamp',
                       threshold: float = 0.25,
                       title: str = "PSI趋势图",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制PSI趋势图

        Parameters:
        -----------
        psi_data : pd.DataFrame
            PSI数据
        feature_col : str, default='feature'
            特征列名
        psi_col : str, default='psi_value'
            PSI值列名
        time_col : str, default='timestamp'
            时间列名
        threshold : float, default=0.25
            PSI阈值
        title : str, default="PSI趋势图"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        # 转换时间列
        psi_data[time_col] = pd.to_datetime(psi_data[time_col])

        # 获取所有特征
        features = psi_data[feature_col].unique()

        fig, ax = plt.subplots(figsize=self.figsize)

        # 为每个特征绘制趋势线
        for feature in features[:10]:  # 限制显示前10个特征
            feature_data = psi_data[psi_data[feature_col] == feature].sort_values(time_col)
            ax.plot(feature_data[time_col], feature_data[psi_col],
                   marker='o', label=feature, alpha=0.7)

        # 添加阈值线
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                  label=f'PSI阈值 ({threshold})')

        ax.set_xlabel('时间')
        ax.set_ylabel('PSI值')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_performance_trends(self,
                                    performance_data: pd.DataFrame,
                                    metrics: List[str] = ['auc', 'ks'],
                                    time_col: str = 'timestamp',
                                    title: str = "模型性能趋势",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制模型性能趋势图

        Parameters:
        -----------
        performance_data : pd.DataFrame
            性能数据
        metrics : list, default=['auc', 'ks']
            指标列表
        time_col : str, default='timestamp'
            时间列名
        title : str, default="模型性能趋势"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        # 转换时间列
        performance_data[time_col] = pd.to_datetime(performance_data[time_col])
        performance_data = performance_data.sort_values(time_col)

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in performance_data.columns:
                axes[i].plot(performance_data[time_col], performance_data[metric],
                           marker='o', linewidth=2, markersize=6, color=f'C{i}')

                # 添加趋势线
                x_numeric = np.arange(len(performance_data))
                z = np.polyfit(x_numeric, performance_data[metric].fillna(0), 1)
                p = np.poly1d(z)
                axes[i].plot(performance_data[time_col], p(x_numeric),
                           linestyle='--', alpha=0.7, color='red', label='趋势线')

                axes[i].set_ylabel(metric.upper())
                axes[i].set_title(f'{metric.upper()}趋势')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

                # 设置y轴范围
                if metric.lower() == 'auc':
                    axes[i].set_ylim(0.5, 1.0)

        axes[-1].set_xlabel('时间')
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_alert_summary(self,
                         alert_data: pd.DataFrame,
                         alert_type_col: str = 'alert_type',
                         severity_col: str = 'severity',
                         time_col: str = 'timestamp',
                         title: str = "报警统计",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制报警统计图

        Parameters:
        -----------
        alert_data : pd.DataFrame
            报警数据
        alert_type_col : str, default='alert_type'
            报警类型列名
        severity_col : str, default='severity'
            严重程度列名
        time_col : str, default='timestamp'
            时间列名
        title : str, default="报警统计"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 报警类型分布
        alert_type_counts = alert_data[alert_type_col].value_counts()
        ax1.pie(alert_type_counts.values, labels=alert_type_counts.index, autopct='%1.1f%%')
        ax1.set_title('报警类型分布')

        # 严重程度分布
        severity_counts = alert_data[severity_col].value_counts()
        colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
        bar_colors = [colors.get(severity, 'gray') for severity in severity_counts.index]
        ax2.bar(severity_counts.index, severity_counts.values, color=bar_colors)
        ax2.set_title('严重程度分布')
        ax2.set_xlabel('严重程度')
        ax2.set_ylabel('数量')

        # 时间趋势
        if time_col in alert_data.columns:
            alert_data[time_col] = pd.to_datetime(alert_data[time_col])
            daily_alerts = alert_data.groupby(alert_data[time_col].dt.date).size()
            ax3.plot(daily_alerts.index, daily_alerts.values, marker='o')
            ax3.set_title('每日报警趋势')
            ax3.set_xlabel('日期')
            ax3.set_ylabel('报警数量')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # 报警类型和严重程度交叉统计
        if len(alert_data) > 0:
            cross_tab = pd.crosstab(alert_data[alert_type_col], alert_data[severity_col])
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Reds', ax=ax4)
            ax4.set_title('报警类型vs严重程度')

        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_dashboard(self,
                        y_true: np.ndarray,
                        y_scores: np.ndarray,
                        feature_importance: Optional[Dict] = None,
                        psi_data: Optional[pd.DataFrame] = None,
                        title: str = "模型监控仪表板",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        创建综合仪表板

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        feature_importance : dict, optional
            特征重要性数据
        psi_data : pd.DataFrame, optional
            PSI数据
        title : str, default="模型监控仪表板"
            图标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        fig : plt.Figure
            图形对象
        """
        fig = plt.figure(figsize=(20, 16))

        # ROC曲线
        ax1 = plt.subplot(2, 3, 1)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_title('ROC曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 评分分布
        ax2 = plt.subplot(2, 3, 2)
        good_scores = y_scores[y_true == 0]
        bad_scores = y_scores[y_true == 1]
        ax2.hist(good_scores, bins=30, alpha=0.7, label='好样本', color='blue', density=True)
        ax2.hist(bad_scores, bins=30, alpha=0.7, label='坏样本', color='red', density=True)
        ax2.set_title('评分分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 特征重要性
        if feature_importance:
            ax3 = plt.subplot(2, 3, 3)
            features = list(feature_importance.keys())[:10]
            importance = [feature_importance[f] for f in features]
            ax3.barh(features, importance, color='skyblue')
            ax3.set_title('特征重要性 (Top 10)')
            ax3.grid(True, alpha=0.3, axis='x')

        # KS曲线
        ax4 = plt.subplot(2, 3, 4)
        df = pd.DataFrame({'score': y_scores, 'target': y_true})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['cumsum_good'] = (1 - df['target']).cumsum()
        df['cumsum_bad'] = df['target'].cumsum()
        total_good = (1 - df['target']).sum()
        total_bad = df['target'].sum()
        df['tpr'] = df['cumsum_bad'] / total_bad
        df['fpr'] = df['cumsum_good'] / total_good
        df['ks'] = df['tpr'] - df['fpr']
        max_ks = df['ks'].max()

        ax4.plot(range(len(df)), df['tpr'], label='TPR', color='red')
        ax4.plot(range(len(df)), df['fpr'], label='FPR', color='blue')
        ax4.plot(range(len(df)), df['ks'], label=f'KS={max_ks:.3f}', color='green')
        ax4.set_title('KS曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Lift图
        ax5 = plt.subplot(2, 3, 5)
        df['bin'] = pd.cut(range(len(df)), 10, labels=False) + 1
        bin_stats = df.groupby('bin').agg({'target': ['count', 'sum', 'mean']}).round(4)
        bin_stats.columns = ['total', 'positive', 'positive_rate']
        bin_stats = bin_stats.reset_index()
        overall_rate = y_true.mean()
        bin_stats['lift'] = bin_stats['positive_rate'] / overall_rate
        ax5.plot(bin_stats['bin'], bin_stats['lift'], 'o-', color='purple', linewidth=2)
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax5.set_title('Lift曲线')
        ax5.grid(True, alpha=0.3)

        # PSI趋势（如果有数据）
        if psi_data is not None and len(psi_data) > 0:
            ax6 = plt.subplot(2, 3, 6)
            if 'feature' in psi_data.columns and 'psi_value' in psi_data.columns:
                features = psi_data['feature'].unique()[:5]  # 显示前5个特征
                for feature in features:
                    feature_data = psi_data[psi_data['feature'] == feature]
                    if len(feature_data) > 1:
                        ax6.plot(range(len(feature_data)), feature_data['psi_value'],
                               marker='o', label=feature, alpha=0.7)
                ax6.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='阈值')
                ax6.set_title('PSI趋势')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=20, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def quick_model_visualization(y_true: np.ndarray,
                            y_scores: np.ndarray,
                            feature_importance: Optional[Dict] = None,
                            save_dir: str = "./visualizations") -> Dict[str, str]:
    """
    快捷函数：生成常用的模型可视化图表

    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_scores : np.ndarray
        预测概率
    feature_importance : dict, optional
        特征重要性数据
    save_dir : str, default="./visualizations"
        保存目录

    Returns:
    --------
    saved_files : dict
        保存的文件路径
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    viz_tool = ModelVisualizationTool()
    saved_files = {}

    # ROC曲线
    roc_path = os.path.join(save_dir, "roc_curve.png")
    viz_tool.plot_roc_curve(y_true, y_scores, save_path=roc_path)
    saved_files['roc_curve'] = roc_path

    # KS曲线
    ks_path = os.path.join(save_dir, "ks_curve.png")
    viz_tool.plot_ks_curve(y_true, y_scores, save_path=ks_path)
    saved_files['ks_curve'] = ks_path

    # Lift曲线
    lift_path = os.path.join(save_dir, "lift_curve.png")
    viz_tool.plot_lift_curve(y_true, y_scores, save_path=lift_path)
    saved_files['lift_curve'] = lift_path

    # 评分分布
    dist_path = os.path.join(save_dir, "score_distribution.png")
    viz_tool.plot_score_distribution(y_true, y_scores, save_path=dist_path)
    saved_files['score_distribution'] = dist_path

    # 特征重要性
    if feature_importance:
        importance_path = os.path.join(save_dir, "feature_importance.png")
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())
        viz_tool.plot_feature_importance(features, np.array(scores), save_path=importance_path)
        saved_files['feature_importance'] = importance_path

    # 综合仪表板
    dashboard_path = os.path.join(save_dir, "dashboard.png")
    viz_tool.create_dashboard(y_true, y_scores, feature_importance, save_path=dashboard_path)
    saved_files['dashboard'] = dashboard_path

    plt.close('all')  # 关闭所有图形以释放内存

    return saved_files