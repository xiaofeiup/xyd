"""
模型交付报告生成器
用于生成标准化的模型交付报告，包含样本情况、模型效果、变量分析等内容
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelDeliveryReport:
    """模型交付报告生成器"""

    def __init__(self, data: pd.DataFrame, target_col: str = 'target',
                 score_col: str = 'score', date_col: str = 'date',
                 sample_type_col: str = 'sample_type'):
        """
        初始化报告生成器

        Args:
            data: 包含预测结果的数据框
            target_col: 目标变量列名
            score_col: 模型分数列名
            date_col: 日期列名
            sample_type_col: 样本类型列名 (all, train, test, oot)
        """
        self.data = data.copy()
        self.target_col = target_col
        self.score_col = score_col
        self.date_col = date_col
        self.sample_type_col = sample_type_col

        # 确保date列为datetime类型
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data['Month'] = self.data[date_col].dt.to_period('M')

    def _calculate_ks(self, y_true: pd.Series, y_score: pd.Series) -> float:
        """计算KS值"""
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ks = max(tpr - fpr)
            return ks
        except:
            return np.nan

    def _calculate_auc(self, y_true: pd.Series, y_score: pd.Series) -> float:
        """计算AUC值"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_score)
        except:
            return np.nan

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """计算PSI值"""
        try:
            # 基于expected确定分箱边界
            _, bin_edges = pd.cut(expected, bins=bins, retbins=True, duplicates='drop')

            # 对两个分布进行分箱
            expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True)
            actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True)

            # 计算每个箱的占比
            expected_pct = expected_binned.value_counts(normalize=True, sort=False)
            actual_pct = actual_binned.value_counts(normalize=True, sort=False)

            # 确保两个分布有相同的索引
            expected_pct = expected_pct.reindex(actual_pct.index, fill_value=0.001)
            actual_pct = actual_pct.fillna(0.001)

            # 计算PSI
            psi = sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
        except:
            return np.nan

    def _create_bins(self, data: pd.Series, bins: int = 10, bin_method: str = 'quantile', retbins: bool = False):
        """
        创建分箱的通用函数

        Args:
            data: 需要分箱的数据
            bins: 分箱数量
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)
            retbins: 是否返回分箱边界

        Returns:
            分箱结果和边界（如果retbins=True）
        """
        if bin_method == 'quantile':
            try:
                if retbins:
                    return pd.qcut(data, q=bins, retbins=True, duplicates='drop')
                else:
                    return pd.qcut(data, q=bins, duplicates='drop')
            except ValueError:
                # 如果等频分箱失败，回退到等距分箱
                if retbins:
                    return pd.cut(data, bins=bins, retbins=True, duplicates='drop')
                else:
                    return pd.cut(data, bins=bins, duplicates='drop')
        else:  # equal
            if retbins:
                return pd.cut(data, bins=bins, retbins=True, duplicates='drop')
            else:
                return pd.cut(data, bins=bins, duplicates='drop')

    def _calculate_iv(self, feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
        """计算IV值"""
        try:
            # 分箱
            feature_binned = pd.cut(feature, bins=bins, duplicates='drop')

            # 计算每个箱的好坏样本数
            crosstab = pd.crosstab(feature_binned, target)

            if crosstab.shape[1] < 2:
                return np.nan

            good = crosstab[0] if 0 in crosstab.columns else 0
            bad = crosstab[1] if 1 in crosstab.columns else 0

            # 避免除零
            good = good.replace(0, 0.001)
            bad = bad.replace(0, 0.001)

            # 计算占比
            good_pct = good / good.sum()
            bad_pct = bad / bad.sum()

            # 计算WOE和IV
            woe = np.log(bad_pct / good_pct)
            iv = sum((bad_pct - good_pct) * woe)

            return iv
        except:
            return np.nan

    def generate_sample_summary(self) -> pd.DataFrame:
        """
        生成样本情况统计表

        Returns:
            样本情况汇总表
        """
        if 'Month' not in self.data.columns:
            raise ValueError("需要日期列来生成按月统计")

        # 按月统计
        monthly_stats = []
        for month in sorted(self.data['Month'].unique()):
            month_data = self.data[self.data['Month'] == month]
            good = (month_data[self.target_col] == 0).sum()
            bad = (month_data[self.target_col] == 1).sum()
            total = len(month_data)
            bad_rate = bad / total if total > 0 else 0

            monthly_stats.append({
                'Month': str(month),
                'Good': good,
                'Bad': bad,
                'Total': total,
                'Bad_Rate': f"{bad_rate:.4f}"
            })

        # 总计行
        total_good = (self.data[self.target_col] == 0).sum()
        total_bad = (self.data[self.target_col] == 1).sum()
        total_total = len(self.data)
        total_bad_rate = total_bad / total_total if total_total > 0 else 0

        monthly_stats.append({
            'Month': '合计',
            'Good': total_good,
            'Bad': total_bad,
            'Total': total_total,
            'Bad_Rate': f"{total_bad_rate:.4f}"
        })

        return pd.DataFrame(monthly_stats)

    def generate_model_performance(self, base_type: str = 'all') -> pd.DataFrame:
        """
        生成模型效果统计表

        Args:
            base_type: PSI计算的基准样本类型

        Returns:
            模型效果汇总表
        """
        performance_stats = []

        # 获取基准数据用于PSI计算
        if base_type in self.data[self.sample_type_col].values:
            base_data = self.data[self.data[self.sample_type_col] == base_type]
            base_scores = base_data[self.score_col]
        else:
            base_scores = None

        # 按样本类型统计
        sample_types = ['all'] + [t for t in self.data[self.sample_type_col].unique() if t != 'all']

        for sample_type in sample_types:
            if sample_type == 'all':
                sample_data = self.data
            else:
                sample_data = self.data[self.data[self.sample_type_col] == sample_type]

            if len(sample_data) == 0:
                continue

            good = (sample_data[self.target_col] == 0).sum()
            bad = (sample_data[self.target_col] == 1).sum()
            total = len(sample_data)
            bad_rate = bad / total if total > 0 else 0

            # 计算KS和AUC
            ks = self._calculate_ks(sample_data[self.target_col], sample_data[self.score_col])
            auc = self._calculate_auc(sample_data[self.target_col], sample_data[self.score_col])

            # 计算PSI
            if base_scores is not None and sample_type != base_type:
                psi = self._calculate_psi(base_scores, sample_data[self.score_col])
            else:
                psi = 0.0

            # 计算lift
            df_lift, baseline_rate = self._calculate_lift_for_plot(sample_data[self.target_col], sample_data[self.score_col])

            performance_stats.append({
                '样本类型': sample_type,
                'Good': good,
                'Bad': bad,
                'Total': total,
                'Bad_Rate': f"{bad_rate:.4f}",
                'KS': f"{ks:.4f}" if not pd.isna(ks) else "N/A",
                'AUC': f"{auc:.4f}" if not pd.isna(auc) else "N/A",
                'TOP10_lift':f"{df_lift.iloc[-1]['Lift']:.4f}" if len(df_lift) > 0 else "N/A",
                'PSI': f"{psi:.4f}" if not pd.isna(psi) else "N/A"
            })

        return pd.DataFrame(performance_stats)

    def generate_metric_report_plot(self, n_bins: int = 10, save_dir: str = None) -> Dict[str, Any]:
        """
        生成模型评估指标图表（类似metric_report_plot）

        Args:
            n_bins: 分箱数量
            save_dir: 图片保存目录（临时目录）

        Returns:
            包含图片路径和汇总数据的字典
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        import tempfile
        import os

        # 设置中文字体支持
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "STHeiti"]
        plt.rcParams["axes.unicode_minus"] = False

        sample_types = [t for t in self.data[self.sample_type_col].unique() if t != 'all']

        if save_dir is None:
            save_dir = tempfile.mkdtemp()

        result = {
            'images': {},
            'metrics_summary': [],
            'lift_details': []
        }

        for sample_type in sample_types:
            sample_data = self.data[self.data[self.sample_type_col] == sample_type]

            if len(sample_data) == 0:
                continue

            y_true = sample_data[self.target_col].values
            y_prob = sample_data[self.score_col].values

            # 计算统计量
            ks_value, df_ks, ks_index = self._calculate_ks_for_plot(y_true, y_prob)
            df_lift, baseline_rate = self._calculate_lift_for_plot(y_true, y_prob, n_bins)

            # 计算AUC
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
            except:
                fpr, tpr = [0, 1], [0, 1]
                roc_auc = np.nan

            # 创建综合图表
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{sample_type} 模型性能综合分析 (KS={ks_value:.3f})', fontsize=16, weight='bold')

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
            axes[0, 1].plot(df_lift['累积召回率'], df_lift['累积Lift'],
                            'b-o', linewidth=2, markersize=6)
            axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('累积召回率')
            axes[0, 1].set_ylabel('累积Lift')
            axes[0, 1].set_title('累积Lift曲线')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. ROC曲线
            axes[0, 2].plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc:.3f})')
            axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 2].set_xlabel('假正率')
            axes[0, 2].set_ylabel('真正率')
            axes[0, 2].set_title('ROC曲线')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # 4. 分箱Lift
            axes[1, 0].bar(df_lift['分箱'], df_lift['Lift'],
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
            random_line = np.linspace(0, 1, len(df_lift))
            axes[1, 2].plot(df_lift['累积召回率'], df_lift['累积召回率'],
                            'b-o', linewidth=2, label='模型增益')
            axes[1, 2].plot(random_line, random_line, 'r--', alpha=0.7, label='随机模型')
            axes[1, 2].fill_between(df_lift['累积召回率'], random_line[:len(df_lift)],
                                    df_lift['累积召回率'], alpha=0.3, color='green')
            axes[1, 2].set_xlabel('样本比例')
            axes[1, 2].set_ylabel('捕获正样本比例')
            axes[1, 2].set_title('增益图')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图片
            img_path = os.path.join(save_dir, f'metric_report_{sample_type}.png')
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            result['images'][sample_type] = img_path

            # 保存汇总指标
            result['metrics_summary'].append({
                '样本类型': sample_type,
                '样本量': len(sample_data),
                '正样本数': int(y_true.sum()),
                '正样本率': f"{baseline_rate:.4f}",
                'KS': f"{ks_value:.4f}" if not pd.isna(ks_value) else "N/A",
                'AUC': f"{roc_auc:.4f}" if not pd.isna(roc_auc) else "N/A",
                'Top10%_Lift': f"{df_lift.iloc[-1]['Lift']:.4f}" if len(df_lift) > 0 else "N/A"
            })

            # 保存Lift详情
            df_lift_copy = df_lift.copy()
            df_lift_copy['样本类型'] = sample_type
            result['lift_details'].append(df_lift_copy)

        return result

    def _calculate_ks_for_plot(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame, int]:
        """计算KS统计量用于绘图"""
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

        df['bad'] = df['y_true']
        df['good'] = 1 - df['y_true']

        total_good = df['good'].sum()
        total_bad = df['bad'].sum()

        if total_good == 0 or total_bad == 0:
            return np.nan, df, 0

        df['cum_good'] = df['good'].cumsum()
        df['cum_bad'] = df['bad'].cumsum()

        df['cum_good_rate'] = df['cum_good'] / total_good
        df['cum_bad_rate'] = df['cum_bad'] / total_bad

        df['ks'] = df['cum_bad_rate'] - df['cum_good_rate']

        ks_value = df['ks'].max()
        ks_index = df['ks'].idxmax()

        return ks_value, df, ks_index

    def _calculate_lift_for_plot(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  n_bins: int = 10) -> Tuple[pd.DataFrame, float]:
        """计算Lift统计表用于绘图"""
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

        total_samples = len(df)
        total_positive = df['y_true'].sum()
        baseline_rate = total_positive / total_samples if total_samples > 0 else 0

        # 分箱
        try:
            df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop') + 1
        except:
            df['decile'] = pd.cut(df['y_prob'], bins=n_bins, labels=False, duplicates='drop') + 1

        # 分箱统计
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

        df_lift = pd.DataFrame(lift_summary)

        # 计算累积统计
        df_lift['累积样本数'] = df_lift['样本数'].cumsum()
        df_lift['累积正样本数'] = df_lift['正样本数'].cumsum()
        df_lift['累积正样本率'] = df_lift['累积正样本数'] / df_lift['累积样本数']
        df_lift['累积Lift'] = df_lift['累积正样本率'] / baseline_rate if baseline_rate > 0 else 0
        df_lift['累积召回率'] = df_lift['累积正样本数'] / total_positive if total_positive > 0 else 0

        return df_lift, baseline_rate

    def generate_score_bins_analysis(self, bins: int = 10, bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成模型效果分箱分析

        Args:
            bins: 分箱数量
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)

        Returns:
            分箱分析结果
        """
        bin_analysis = []

        sample_types = ['all'] + [t for t in self.data[self.sample_type_col].unique() if t != 'all']

        # 基于all样本确定分箱边界
        if 'all' in self.data[self.sample_type_col].unique():
            all_data = self.data
        else:
            all_data = self.data

        # 根据分箱方式选择分箱函数
        _, bin_edges = self._create_bins(all_data[self.score_col], bins=bins, bin_method=bin_method, retbins=True)

        for sample_type in sample_types:
            if sample_type == 'all':
                sample_data = self.data
            else:
                sample_data = self.data[self.data[self.sample_type_col] == sample_type]

            if len(sample_data) == 0:
                continue

            # 分箱
            sample_data_copy = sample_data.copy()
            sample_data_copy['score_bin'] = pd.cut(sample_data_copy[self.score_col],
                                                  bins=bin_edges, include_lowest=True)

            # 按分箱统计
            bin_stats = sample_data_copy.groupby('score_bin').agg({
                self.target_col: ['count', 'sum', 'mean']
            }).round(4)

            bin_stats.columns = ['total_users', 'bad_users', 'bad_rate']
            bin_stats['good_users'] = bin_stats['total_users'] - bin_stats['bad_users']
            bin_stats['total_pct'] = bin_stats['total_users'] / len(sample_data)
            bin_stats['bad_pct'] = bin_stats['bad_users'] / bin_stats['bad_users'].sum()

            # 计算累计指标
            bin_stats = bin_stats.sort_index(ascending=False)  # 按分数降序
            bin_stats['cum_bad_rate'] = (bin_stats['bad_users'].cumsum() /
                                       bin_stats['total_users'].cumsum())
            bin_stats['cum_bad_pct'] = bin_stats['bad_users'].cumsum() / bin_stats['bad_users'].sum()

            # 计算KS和LIFT
            bin_stats['ks'] = np.abs(bin_stats['cum_bad_pct'] -
                                   (bin_stats['total_users'].cumsum() / bin_stats['total_users'].sum()))
            bin_stats['lift'] = bin_stats['bad_rate'] / (bin_stats['bad_users'].sum() / bin_stats['total_users'].sum())

            # PSI计算（与all样本对比）
            if sample_type != 'all':
                all_sample_data = self.data.copy()
                all_sample_data['score_bin'] = pd.cut(all_sample_data[self.score_col],
                                                    bins=bin_edges, include_lowest=True)
                all_bin_pct = all_sample_data.groupby('score_bin').size() / len(all_sample_data)
                current_bin_pct = bin_stats['total_pct']

                # 确保索引一致
                all_bin_pct = all_bin_pct.reindex(current_bin_pct.index, fill_value=0.001)
                current_bin_pct = current_bin_pct.fillna(0.001)

                psi_values = (current_bin_pct - all_bin_pct) * np.log(current_bin_pct / all_bin_pct)
            else:
                psi_values = pd.Series([0] * len(bin_stats), index=bin_stats.index)

            bin_stats['psi'] = psi_values

            # 重新排序列
            bin_stats = bin_stats[['good_users', 'bad_users', 'total_users', 'total_pct',
                                 'bad_pct', 'bad_rate', 'ks', 'lift', 'cum_bad_rate',
                                 'cum_bad_pct', 'psi']]

            # 添加样本类型信息
            for idx, row in bin_stats.iterrows():
                bin_analysis.append({
                    '样本类型': sample_type,
                    '分箱': str(idx),
                    '好用户': int(row['good_users']),
                    '坏用户': int(row['bad_users']),
                    '总用户': int(row['total_users']),
                    '总用户占比': f"{row['total_pct']:.4f}",
                    '坏用户占比': f"{row['bad_pct']:.4f}",
                    '坏用户率': f"{row['bad_rate']:.4f}",
                    'KS': f"{row['ks']:.4f}",
                    'LIFT': f"{row['lift']:.4f}",
                    '累计坏用户率': f"{row['cum_bad_rate']:.4f}",
                    '累计坏用户占比': f"{row['cum_bad_pct']:.4f}",
                    'PSI稳定性': f"{row['psi']:.4f}"
                })

            # 添加总计行
            total_good = bin_stats['good_users'].sum()
            total_bad = bin_stats['bad_users'].sum()
            total_total = bin_stats['total_users'].sum()

            bin_analysis.append({
                '样本类型': sample_type,
                '分箱': '总计',
                '好用户': int(total_good),
                '坏用户': int(total_bad),
                '总用户': int(total_total),
                '总用户占比': "1.0000",
                '坏用户占比': "1.0000",
                '坏用户率': f"{total_bad/total_total:.4f}" if total_total > 0 else "0.0000",
                'KS': f"{bin_stats['ks'].max():.4f}",
                'LIFT': "N/A",
                '累计坏用户率': f"{total_bad/total_total:.4f}" if total_total > 0 else "0.0000",
                '累计坏用户占比': "1.0000",
                'PSI稳定性': f"{bin_stats['psi'].sum():.4f}"
            })

        return pd.DataFrame(bin_analysis)

    def generate_top_features_effectiveness(self, features: List[str],
                                          feature_names: Dict[str, str] = None,
                                          top_n: int = 10,
                                          bins: int = 5,
                                          bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成TOP10变量有效性分析 - 分箱级别统计

        Args:
            features: 特征列名列表
            feature_names: 特征中文名映射
            top_n: 返回前N个特征
            bins: 分箱数量
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)

        Returns:
            变量有效性分析结果 - 每行一个分箱
        """
        if feature_names is None:
            feature_names = {f: f for f in features}

        effectiveness_results = []

        # 计算每个特征的总体有效性(gain)
        feature_gains = {}
        for feature in features:
            if feature not in self.data.columns:
                continue

            # 计算总体IV作为gain指标
            try:
                all_iv = self._calculate_iv(self.data[feature], self.data[self.target_col])
                feature_gains[feature] = all_iv if not pd.isna(all_iv) else 0
            except:
                feature_gains[feature] = 0

        # 选择TOP N特征
        top_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for feature, total_gain in top_features:
            if feature not in self.data.columns:
                continue

            try:
                # 对特征进行分箱
                self.data[f'{feature}_bin'] = self._create_bins(self.data[feature], bins=bins, bin_method=bin_method)
                bin_labels = self.data[f'{feature}_bin'].cat.categories

                # 按分箱统计
                for bin_label in bin_labels:
                    bin_data = self.data[self.data[f'{feature}_bin'] == bin_label]
                    if len(bin_data) == 0:
                        continue

                    bin_result = {
                        '指标英文': feature,
                        '指标中文': feature_names.get(feature, feature),
                        '数据类型': str(self.data[feature].dtype),
                        '分箱': str(bin_label),
                        'total_gain': f"{total_gain:.4f}"
                    }

                    # 按样本类型统计每个分箱
                    sample_types = ['train', 'test', 'oot']
                    for sample_type in sample_types:
                        sample_bin_data = bin_data[bin_data[self.sample_type_col] == sample_type]

                        if len(sample_bin_data) == 0:
                            bin_result[f'N_{sample_type}'] = 0
                            bin_result[f'分布占比_{sample_type}'] = "0.0000"
                            bin_result[f'坏样本数量_{sample_type}'] = 0
                            bin_result[f'逾期率_{sample_type}'] = "0.0000"
                            continue

                        n_sample = len(sample_bin_data)
                        bad_count = (sample_bin_data[self.target_col] == 1).sum()
                        bad_rate = bad_count / n_sample if n_sample > 0 else 0

                        # 计算在该样本类型中的分布占比
                        total_sample_type = len(self.data[self.data[self.sample_type_col] == sample_type])
                        distribution_ratio = n_sample / total_sample_type if total_sample_type > 0 else 0

                        bin_result[f'N_{sample_type}'] = n_sample
                        bin_result[f'分布占比_{sample_type}'] = f"{distribution_ratio:.4f}"
                        bin_result[f'坏样本数量_{sample_type}'] = bad_count
                        bin_result[f'逾期率_{sample_type}'] = f"{bad_rate:.4f}"

                    effectiveness_results.append(bin_result)

            except Exception as e:
                print(f"特征 {feature} 分箱失败: {e}")
                continue

        return pd.DataFrame(effectiveness_results)

    def generate_top_features_effectiveness_monthly(self, features: List[str],
                                                   feature_names: Dict[str, str] = None,
                                                   sample_type: str = 'test',
                                                   top_n: int = 10,
                                                   bins: int = 5,
                                                   bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成按月拆分的变量有效性分析 - 分箱级别按月统计

        Args:
            features: 特征列名列表
            feature_names: 特征中文名映射
            sample_type: 分析的样本类型
            top_n: 返回前N个特征
            bins: 分箱数量

        Returns:
            按月拆分的变量有效性分析结果 - 每行一个分箱
        """
        if 'Month' not in self.data.columns:
            raise ValueError("需要日期列来生成按月统计")

        if feature_names is None:
            feature_names = {f: f for f in features}

        # 筛选指定样本类型的数据
        sample_data = self.data[self.data[self.sample_type_col] == sample_type]
        if len(sample_data) == 0:
            return pd.DataFrame()

        # 获取月份列表
        months = sorted(sample_data['Month'].unique())
        month_strs = [str(m) for m in months]

        # 计算特征重要性并选择TOP N
        feature_gains = {}
        for feature in features:
            if feature not in sample_data.columns:
                continue
            try:
                iv = self._calculate_iv(sample_data[feature], sample_data[self.target_col])
                feature_gains[feature] = iv if not pd.isna(iv) else 0
            except:
                feature_gains[feature] = 0

        top_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)[:top_n]

        monthly_results = []

        for feature, total_gain in top_features:
            if feature not in sample_data.columns:
                continue

            try:
                # 对特征进行分箱（基于全量该类型样本）
                sample_data_copy = sample_data.copy()
                sample_data_copy[f'{feature}_bin'] = self._create_bins(sample_data_copy[feature], bins=bins, bin_method=bin_method)
                bin_labels = sample_data_copy[f'{feature}_bin'].cat.categories

                # 按分箱统计
                for bin_label in bin_labels:
                    bin_data = sample_data_copy[sample_data_copy[f'{feature}_bin'] == bin_label]
                    if len(bin_data) == 0:
                        continue

                    bin_result = {
                        '指标英文': feature,
                        '指标中文': feature_names.get(feature, feature),
                        '数据类型': str(sample_data[feature].dtype),
                        '分箱': str(bin_label),
                        'total_gain': f"{total_gain:.4f}"
                    }

                    # 按月统计每个分箱
                    for month in months:
                        month_bin_data = bin_data[bin_data['Month'] == month]
                        month_str = str(month)

                        if len(month_bin_data) == 0:
                            bin_result[f'N_{sample_type}_{month_str}'] = 0
                            bin_result[f'分布占比_{sample_type}_{month_str}'] = "0.0000"
                            bin_result[f'坏样本数量_{sample_type}_{month_str}'] = 0
                            bin_result[f'逾期率_{sample_type}_{month_str}'] = "0.0000"
                            continue

                        n_month = len(month_bin_data)
                        bad_count = (month_bin_data[self.target_col] == 1).sum()
                        bad_rate = bad_count / n_month if n_month > 0 else 0

                        # 计算在该月该样本类型中的分布占比
                        total_month_sample = len(sample_data_copy[sample_data_copy['Month'] == month])
                        distribution_ratio = n_month / total_month_sample if total_month_sample > 0 else 0

                        bin_result[f'N_{sample_type}_{month_str}'] = n_month
                        bin_result[f'分布占比_{sample_type}_{month_str}'] = f"{distribution_ratio:.4f}"
                        bin_result[f'坏样本数量_{sample_type}_{month_str}'] = bad_count
                        bin_result[f'逾期率_{sample_type}_{month_str}'] = f"{bad_rate:.4f}"

                    monthly_results.append(bin_result)

            except Exception as e:
                print(f"特征 {feature} 按月分箱失败: {e}")
                continue

        return pd.DataFrame(monthly_results)

    def generate_top_features_stability(self, features: List[str],
                                       feature_names: Dict[str, str] = None,
                                       top_n: int = 10,
                                       bins: int = 5,
                                       bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成TOP10变量稳定性分析 - 分箱级别统计

        Args:
            features: 特征列名列表
            feature_names: 特征中文名映射
            top_n: 返回前N个特征
            bins: 分箱数量

        Returns:
            变量稳定性分析结果 - 每行一个分箱
        """
        if feature_names is None:
            feature_names = {f: f for f in features}

        # 计算特征重要性并选择TOP N
        feature_gains = {}
        for feature in features:
            if feature not in self.data.columns:
                continue
            try:
                iv = self._calculate_iv(self.data[feature], self.data[self.target_col])
                feature_gains[feature] = iv if not pd.isna(iv) else 0
            except:
                feature_gains[feature] = 0

        top_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)[:top_n]

        stability_results = []

        # 获取train数据作为基准
        train_data = self.data[self.data[self.sample_type_col] == 'train']

        for feature, total_gain in top_features:
            if feature not in self.data.columns:
                continue

            try:
                # 基于train数据确定分箱边界，确保一致性
                if len(train_data) > 0:
                    _, bin_edges = pd.cut(train_data[feature], bins=bins, retbins=True, duplicates='drop')
                else:
                    _, bin_edges = pd.cut(self.data[feature], bins=bins, retbins=True, duplicates='drop')

                # 对所有数据应用相同分箱
                self.data[f'{feature}_bin'] = pd.cut(self.data[feature], bins=bin_edges, include_lowest=True)
                bin_labels = self.data[f'{feature}_bin'].cat.categories

                # 按分箱统计
                for bin_label in bin_labels:
                    bin_data = self.data[self.data[f'{feature}_bin'] == bin_label]
                    if len(bin_data) == 0:
                        continue

                    bin_result = {
                        '指标英文': feature,
                        '指标中文': feature_names.get(feature, feature),
                        '数据类型': str(self.data[feature].dtype),
                        '分箱': str(bin_label),
                        'total_gain': f"{total_gain:.4f}"
                    }

                    # 分样本类型分析每个分箱
                    sample_types = ['train', 'test', 'oot']

                    # 先计算train分箱的分布作为基准
                    train_bin_data = bin_data[bin_data[self.sample_type_col] == 'train']
                    train_total = len(self.data[self.data[self.sample_type_col] == 'train'])
                    train_bin_ratio = len(train_bin_data) / train_total if train_total > 0 else 0

                    for sample_type in sample_types:
                        sample_bin_data = bin_data[bin_data[self.sample_type_col] == sample_type]
                        sample_total = len(self.data[self.data[self.sample_type_col] == sample_type])

                        if len(sample_bin_data) == 0 or sample_total == 0:
                            bin_result[f'N_{sample_type}'] = 0
                            bin_result[f'{sample_type}_iv'] = "0.0000"
                            bin_result[f'psi_{sample_type}'] = "0.0000"
                            continue

                        n_sample = len(sample_bin_data)
                        bin_result[f'N_{sample_type}'] = n_sample

                        # 计算该分箱的IV贡献
                        try:
                            good_count = (sample_bin_data[self.target_col] == 0).sum()
                            bad_count = (sample_bin_data[self.target_col] == 1).sum()

                            # 计算总体的好坏样本数
                            sample_type_data = self.data[self.data[self.sample_type_col] == sample_type]
                            total_good = (sample_type_data[self.target_col] == 0).sum()
                            total_bad = (sample_type_data[self.target_col] == 1).sum()

                            if total_good > 0 and total_bad > 0 and good_count > 0 and bad_count > 0:
                                good_rate = good_count / total_good
                                bad_rate = bad_count / total_bad
                                woe = np.log(bad_rate / good_rate)
                                iv_contrib = (bad_rate - good_rate) * woe
                                bin_result[f'{sample_type}_iv'] = f"{iv_contrib:.4f}"
                            else:
                                bin_result[f'{sample_type}_iv'] = "0.0000"
                        except:
                            bin_result[f'{sample_type}_iv'] = "0.0000"

                        # 计算PSI（以train为基准）
                        if sample_type == 'train':
                            bin_result[f'psi_{sample_type}'] = "0.0000"
                        else:
                            try:
                                current_bin_ratio = n_sample / sample_total
                                if train_bin_ratio > 0 and current_bin_ratio > 0:
                                    psi_contrib = (current_bin_ratio - train_bin_ratio) * np.log(current_bin_ratio / train_bin_ratio)
                                    bin_result[f'psi_{sample_type}'] = f"{psi_contrib:.4f}"
                                else:
                                    bin_result[f'psi_{sample_type}'] = "0.0000"
                            except:
                                bin_result[f'psi_{sample_type}'] = "0.0000"

                    stability_results.append(bin_result)

            except Exception as e:
                print(f"特征 {feature} 稳定性分析失败: {e}")
                continue

        return pd.DataFrame(stability_results)

    def generate_feature_time_distribution(self, features: List[str],
                                         feature_names: Dict[str, str] = None,
                                         top_n: int = 10,
                                         bins: int = 5,
                                         bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成单特征时间分布TOP10分析

        Args:
            features: 特征列名列表
            feature_names: 特征中文名映射
            top_n: 返回前N个特征
            bins: 分箱数量
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)

        Returns:
            特征时间分布分析结果
        """
        if 'Month' not in self.data.columns:
            raise ValueError("需要日期列来生成时间分布分析")

        if feature_names is None:
            feature_names = {f: f for f in features}

        # 获取月份列表
        months = sorted(self.data['Month'].unique())
        month_strs = [str(m) for m in months]

        # 计算特征重要性并选择TOP N
        feature_gains = {}
        for feature in features:
            if feature not in self.data.columns:
                continue
            try:
                iv = self._calculate_iv(self.data[feature], self.data[self.target_col])
                feature_gains[feature] = iv if not pd.isna(iv) else 0
            except:
                feature_gains[feature] = 0

        top_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)[:top_n]

        distribution_results = []

        for feature, _ in top_features:
            if feature not in self.data.columns:
                continue

            # 对特征进行分箱
            try:
                self.data[f'{feature}_bin'] = self._create_bins(self.data[feature], bins=bins, bin_method=bin_method)
                bin_labels = self.data[f'{feature}_bin'].cat.categories
            except:
                continue

            # 按分箱和月份统计
            for bin_label in bin_labels:
                bin_data = self.data[self.data[f'{feature}_bin'] == bin_label]
                if len(bin_data) == 0:
                    continue

                # 创建结果字典，按指定顺序
                result = {}

                # 1. 数据类型(特征)和分箱
                result['数据类型'] = feature_names.get(feature, feature)
                result['score_bin'] = str(bin_label)

                # 按月统计数据
                monthly_data = {}
                total_num = 0
                total_bad = 0

                for month in months:
                    month_data = bin_data[bin_data['Month'] == month]
                    month_str = str(month)

                    num = len(month_data)
                    bad_num = (month_data[self.target_col] == 1).sum()
                    bad_ratio = bad_num / num if num > 0 else 0
                    lift = (bad_ratio / (self.data[self.target_col].mean())) if num > 0 else 0

                    monthly_data[month_str] = {
                        'num': num,
                        'bad_num': bad_num,
                        'bad_ratio': bad_ratio,
                        'lift': lift
                    }

                    total_num += num
                    total_bad += bad_num

                # 2. 各月的num (n列)
                for month in months:
                    month_str = str(month)
                    result[f'num_{month_str}'] = monthly_data[month_str]['num']

                # 3. 合计数量
                result['合计数量'] = total_num

                # 4. 各月的ratio (n列) - 该分箱在该月的样本数占全部样本总数的比例
                for month in months:
                    month_str = str(month)
                    ratio = monthly_data[month_str]['num'] / len(self.data) if len(self.data) > 0 else 0
                    result[f'ratio_{month_str}'] = f"{ratio:.4f}"

                # 5. 合计占比
                result['合计占比'] = f"{total_num / len(self.data):.4f}"

                # 6. 各月的bad_ratio (n列)
                for month in months:
                    month_str = str(month)
                    result[f'bad_ratio_{month_str}'] = f"{monthly_data[month_str]['bad_ratio']:.4f}"

                # 7. 各月的bad_num (n列)
                for month in months:
                    month_str = str(month)
                    result[f'bad_num_{month_str}'] = monthly_data[month_str]['bad_num']

                # 8. 各月的lift (n列)
                for month in months:
                    month_str = str(month)
                    result[f'lift_{month_str}'] = f"{monthly_data[month_str]['lift']:.4f}"

                distribution_results.append(result)

        return pd.DataFrame(distribution_results)

    def generate_model_score_distribution(self, bins: int = 10, bin_method: str = 'quantile') -> pd.DataFrame:
        """
        生成模型分时间分布分析（复用特征时间分布逻辑）

        Args:
            bins: 分箱数量
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)

        Returns:
            模型分时间分布分析结果
        """
        if 'Month' not in self.data.columns:
            raise ValueError("需要日期列来生成时间分布分析")

        # 获取月份列表
        months = sorted(self.data['Month'].unique())
        month_strs = [str(m) for m in months]

        # 对模型分进行分箱
        try:
            self.data['score_bin'] = self._create_bins(self.data[self.score_col], bins=bins, bin_method=bin_method)
            bin_labels = self.data['score_bin'].cat.categories
        except:
            return pd.DataFrame()

        distribution_results = []

        # 按分箱统计
        for bin_label in bin_labels:
            bin_data = self.data[self.data['score_bin'] == bin_label]
            if len(bin_data) == 0:
                continue

            # 创建结果字典，按指定顺序
            result = {}

            # 1. 数据类型和分箱
            result['数据类型'] = '模型分'
            result['score_bin'] = str(bin_label)

            # 按月统计数据
            monthly_data = {}
            total_num = 0
            total_bad = 0

            for month in months:
                month_data = bin_data[bin_data['Month'] == month]
                month_str = str(month)

                num = len(month_data)
                bad_num = (month_data[self.target_col] == 1).sum()
                bad_ratio = bad_num / num if num > 0 else 0
                lift = (bad_ratio / (self.data[self.target_col].mean())) if num > 0 else 0

                monthly_data[month_str] = {
                    'num': num,
                    'bad_num': bad_num,
                    'bad_ratio': bad_ratio,
                    'lift': lift
                }

                total_num += num
                total_bad += bad_num

            # 2. 各月的num (n列)
            for month in months:
                month_str = str(month)
                result[f'num_{month_str}'] = monthly_data[month_str]['num']

            # 3. 合计数量
            result['合计数量'] = total_num

            # 4. 各月的ratio (n列) - 该分箱在该月的样本数占全部样本总数的比例
            for month in months:
                month_str = str(month)
                ratio = monthly_data[month_str]['num'] / len(self.data) if len(self.data) > 0 else 0
                result[f'ratio_{month_str}'] = f"{ratio:.4f}"

            # 5. 合计占比
            result['合计占比'] = f"{total_num / len(self.data):.4f}"

            # 6. 各月的bad_ratio (n列)
            for month in months:
                month_str = str(month)
                result[f'bad_ratio_{month_str}'] = f"{monthly_data[month_str]['bad_ratio']:.4f}"

            # 7. 各月的bad_num (n列)
            for month in months:
                month_str = str(month)
                result[f'bad_num_{month_str}'] = monthly_data[month_str]['bad_num']

            # 8. 各月的lift (n列)
            for month in months:
                month_str = str(month)
                result[f'lift_{month_str}'] = f"{monthly_data[month_str]['lift']:.4f}"

            distribution_results.append(result)

        return pd.DataFrame(distribution_results)

    def generate_full_report(self, features: List[str] = None,
                           feature_names: Dict[str, str] = None,
                           save_path: str = None,
                           bin_method: str = 'quantile') -> Dict[str, pd.DataFrame]:
        """
        生成完整的模型交付报告

        Args:
            features: 特征列名列表
            feature_names: 特征中文名映射
            save_path: 保存路径（可选）
            bin_method: 分箱方式，'quantile'(等频分箱)或'equal'(等距分箱)

        Returns:
            包含所有报告表格的字典
        """
        report = {}

        try:
            # 1. 样本情况
            report['样本情况'] = self.generate_sample_summary()
            print("✓ 样本情况统计完成")
        except Exception as e:
            print(f"✗ 样本情况统计失败: {e}")
            report['样本情况'] = pd.DataFrame()

        try:
            # 2. 模型效果
            report['模型效果'] = self.generate_model_performance()
            print("✓ 模型效果分析完成")
        except Exception as e:
            print(f"✗ 模型效果分析失败: {e}")
            report['模型效果'] = pd.DataFrame()

        try:
            # 3. 模型效果分箱
            report['模型效果分箱'] = self.generate_score_bins_analysis(bin_method=bin_method)
            print("✓ 模型效果分箱分析完成")
        except Exception as e:
            print(f"✗ 模型效果分箱分析失败: {e}")
            report['模型效果分箱'] = pd.DataFrame()

        try:
            # 3.5 模型评估指标报告（metric_report_plot的图表版本）
            metric_plot_result = self.generate_metric_report_plot(n_bins=10)
            # report['评估指标汇总'] = pd.DataFrame(metric_plot_result['metrics_summary'])
            if metric_plot_result['lift_details']:
                report['Lift分箱详情'] = pd.concat(metric_plot_result['lift_details'], ignore_index=True)
            else:
                report['Lift分箱详情'] = pd.DataFrame()
            # 保存图片路径供后续插入Excel
            report['_metric_images'] = metric_plot_result['images']
            print("✓ 评估指标报告生成完成")
        except Exception as e:
            print(f"✗ 评估指标报告生成失败: {e}")
            # report['评估指标汇总'] = pd.DataFrame()
            report['Lift分箱详情'] = pd.DataFrame()
            report['_metric_images'] = {}

        if features:
            try:
                # 4. TOP10变量有效性
                report['TOP10变量有效性'] = self.generate_top_features_effectiveness(features, feature_names, bin_method=bin_method)
                print("✓ TOP10变量有效性分析完成")
            except Exception as e:
                print(f"✗ TOP10变量有效性分析失败: {e}")
                report['TOP10变量有效性'] = pd.DataFrame()

            try:
                # 4-1. 按月拆分的变量有效性（以test为例）
                report['变量有效性按月拆分'] = self.generate_top_features_effectiveness_monthly(features, feature_names, bin_method=bin_method)
                print("✓ 变量有效性按月拆分分析完成")
            except Exception as e:
                print(f"✗ 变量有效性按月拆分分析失败: {e}")
                report['变量有效性按月拆分'] = pd.DataFrame()

            try:
                # 5. TOP10变量稳定性
                report['TOP10变量稳定性'] = self.generate_top_features_stability(features, feature_names, bin_method=bin_method)
                print("✓ TOP10变量稳定性分析完成")
            except Exception as e:
                print(f"✗ TOP10变量稳定性分析失败: {e}")
                report['TOP10变量稳定性'] = pd.DataFrame()

            try:
                # 6. 单特征时间分布TOP10
                report['单特征时间分布TOP10'] = self.generate_feature_time_distribution(features, feature_names, bin_method=bin_method)
                print("✓ 单特征时间分布分析完成")
            except Exception as e:
                print(f"✗ 单特征时间分布分析失败: {e}")
                report['单特征时间分布TOP10'] = pd.DataFrame()

        try:
            # 7. 模型分表现
            report['模型分表现'] = self.generate_model_score_distribution(bin_method=bin_method)
            print("✓ 模型分表现分析完成")
        except Exception as e:
            print(f"✗ 模型分表现分析失败: {e}")
            report['模型分表现'] = pd.DataFrame()

        # 保存报告
        if save_path:
            try:
                from openpyxl import Workbook
                from openpyxl.drawing.image import Image as OpenpyxlImage
                import os

                # 获取图片信息
                metric_images = report.pop('_metric_images', {})

                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    # 先写入所有数据表格
                    for sheet_name, df in report.items():
                        if not df.empty and not sheet_name.startswith('_'):
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # 为每个样本类型的图片创建单独的sheet
                    for sample_type, img_path in metric_images.items():
                        if os.path.exists(img_path):
                            sheet_name = f'{sample_type}_评估图表'
                            # 创建一个空的sheet
                            empty_df = pd.DataFrame({'': ['']})
                            empty_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

                            # 获取worksheet并插入图片
                            ws = writer.sheets[sheet_name]
                            img = OpenpyxlImage(img_path)
                            # 调整图片大小（可选）
                            img.width = 1200
                            img.height = 720
                            ws.add_image(img, 'A1')

                print(f"✓ 报告已保存至: {save_path}")

                # 清理临时图片文件
                for img_path in metric_images.values():
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except:
                        pass

            except Exception as e:
                print(f"✗ 报告保存失败: {e}")
                import traceback
                traceback.print_exc()

        # 移除内部使用的图片路径信息
        report.pop('_metric_images', None)

        return report


def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)

    n_samples = 10000
    months = pd.date_range('2023-01-01', '2023-12-01', freq='MS')

    data = []
    for i in range(n_samples):
        month = np.random.choice(months)
        sample_type = np.random.choice(['train', 'test', 'oot'], p=[0.6, 0.2, 0.2])

        # 生成特征
        feature1 = np.random.normal(0, 1)
        feature2 = np.random.normal(0, 1)
        feature3 = np.random.normal(0, 1)

        # 生成目标变量（与特征相关）
        prob = 1 / (1 + np.exp(-(0.5 * feature1 + 0.3 * feature2 - 0.2 * feature3)))
        target = np.random.binomial(1, prob)

        # 生成模型分数
        score = 0.5 * feature1 + 0.3 * feature2 - 0.2 * feature3 + np.random.normal(0, 0.1)

        data.append({
            'date': month,
            'sample_type': sample_type,
            'target': target,
            'score': score,
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # 测试代码
    print("生成示例数据...")
    test_data = create_sample_data()

    print("初始化报告生成器...")
    reporter = ModelDeliveryReport(test_data)

    print("生成完整报告...")
    features = ['feature1', 'feature2', 'feature3']
    feature_names = {
        'feature1': '特征1',
        'feature2': '特征2',
        'feature3': '特征3'
    }

    report = reporter.generate_full_report(
        features=features,
        feature_names=feature_names,
        save_path='model_delivery_report.xlsx'
    )

    print("\n报告生成完成！包含以下表格：")
    for name, df in report.items():
        print(f"- {name}: {len(df)} 行")
