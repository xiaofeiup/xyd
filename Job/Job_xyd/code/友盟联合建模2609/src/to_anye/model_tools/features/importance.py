"""
特征重要性分析模块

提供多种特征重要性计算方法，包括基于模型的和基于统计的方法
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器

    支持多种重要性计算方法
    """

    def __init__(self, random_state: int = 42):
        """
        初始化分析器

        Parameters:
        -----------
        random_state : int, default=42
            随机种子
        """
        self.random_state = random_state
        self.importance_results_ = {}

    def calculate_tree_importance(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 method: str = 'random_forest') -> pd.DataFrame:
        """
        计算基于树模型的特征重要性

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        method : str, default='random_forest'
            树模型类型 ('random_forest', 'gbdt')

        Returns:
        --------
        importance_df : pd.DataFrame
            特征重要性结果
        """
        if method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif method == 'gbdt':
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的树模型类型: {method}")

        # 训练模型
        model.fit(X, y)

        # 获取特征重要性
        importance_scores = model.feature_importances_

        # 构建结果
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores,
            'method': method
        }).sort_values('importance', ascending=False)

        # 计算重要性百分比
        importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

        self.importance_results_[method] = importance_df

        return importance_df

    def calculate_permutation_importance(self,
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       model=None,
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        计算排列重要性

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        model : sklearn estimator, optional
            模型，如果为None则使用RandomForest
        n_repeats : int, default=10
            重复次数

        Returns:
        --------
        importance_df : pd.DataFrame
            排列重要性结果
        """
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )

        # 训练模型
        model.fit(X, y)

        # 计算排列重要性
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )

        # 构建结果
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'method': 'permutation'
        }).sort_values('importance_mean', ascending=False)

        # 计算重要性百分比
        total_importance = importance_df['importance_mean'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = importance_df['importance_mean'] / total_importance * 100
        else:
            importance_df['importance_pct'] = 0

        self.importance_results_['permutation'] = importance_df

        return importance_df

    def calculate_univariate_importance(self,
                                      X: pd.DataFrame,
                                      y: pd.Series) -> pd.DataFrame:
        """
        计算单变量重要性（基于IV值）

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量

        Returns:
        --------
        importance_df : pd.DataFrame
            单变量重要性结果
        """
        from .selection import calculate_iv

        importance_scores = []

        for feature in X.columns:
            try:
                iv_score = calculate_iv(X[feature], y)
                importance_scores.append(iv_score)
            except Exception as e:
                warnings.warn(f"特征 {feature} IV计算失败: {str(e)}")
                importance_scores.append(0.0)

        # 构建结果
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores,
            'method': 'iv'
        }).sort_values('importance', ascending=False)

        # 计算重要性百分比
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = importance_df['importance'] / total_importance * 100
        else:
            importance_df['importance_pct'] = 0

        self.importance_results_['iv'] = importance_df

        return importance_df

    def calculate_correlation_importance(self,
                                       X: pd.DataFrame,
                                       y: pd.Series) -> pd.DataFrame:
        """
        计算相关性重要性

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量

        Returns:
        --------
        importance_df : pd.DataFrame
            相关性重要性结果
        """
        # 计算与目标变量的相关性
        correlations = X.corrwith(y).abs()

        # 构建结果
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': correlations.values,
            'method': 'correlation'
        }).sort_values('importance', ascending=False)

        # 去除NaN值
        importance_df = importance_df.dropna()

        # 计算重要性百分比
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = importance_df['importance'] / total_importance * 100
        else:
            importance_df['importance_pct'] = 0

        self.importance_results_['correlation'] = importance_df

        return importance_df

    def calculate_all_importance(self,
                               X: pd.DataFrame,
                               y: pd.Series,
                               methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        计算所有类型的特征重要性

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        methods : list, optional
            要计算的方法列表

        Returns:
        --------
        all_importance : dict
            所有重要性结果
        """
        if methods is None:
            methods = ['random_forest', 'gbdt', 'permutation', 'iv', 'correlation']

        results = {}

        for method in methods:
            try:
                if method in ['random_forest', 'gbdt']:
                    results[method] = self.calculate_tree_importance(X, y, method)
                elif method == 'permutation':
                    results[method] = self.calculate_permutation_importance(X, y)
                elif method == 'iv':
                    results[method] = self.calculate_univariate_importance(X, y)
                elif method == 'correlation':
                    results[method] = self.calculate_correlation_importance(X, y)
                else:
                    warnings.warn(f"不支持的方法: {method}")

                print(f"✓ {method} 重要性计算完成")

            except Exception as e:
                warnings.warn(f"方法 {method} 计算失败: {str(e)}")
                print(f"✗ {method} 重要性计算失败: {str(e)}")

        self.importance_results_.update(results)
        return results

    @staticmethod
    def _get_score_column(importance_df: pd.DataFrame) -> str:
        """
        返回重要性 DataFrame 中表示分数的列名。

        不同方法的分数列名不同：基于树/IV/相关性的方法使用 'importance'，
        而排列重要性使用 'importance_mean'。这里统一兼容。
        """
        if 'importance' in importance_df.columns:
            return 'importance'
        if 'importance_mean' in importance_df.columns:
            return 'importance_mean'
        raise KeyError("重要性结果中找不到分数列（'importance' 或 'importance_mean'）")

    def get_consensus_ranking(self,
                            methods: Optional[List[str]] = None,
                            top_k: Optional[int] = None) -> pd.DataFrame:
        """
        获取综合排名

        Parameters:
        -----------
        methods : list, optional
            参与综合排名的方法
        top_k : int, optional
            返回前k个特征

        Returns:
        --------
        consensus_df : pd.DataFrame
            综合排名结果
        """
        if not self.importance_results_:
            raise ValueError("请先计算特征重要性")

        available_methods = list(self.importance_results_.keys())

        if methods is None:
            methods = available_methods
        else:
            methods = [m for m in methods if m in available_methods]

        if not methods:
            raise ValueError("没有可用的重要性计算结果")

        # 收集所有特征的排名
        feature_rankings = {}

        for method in methods:
            importance_df = self.importance_results_[method]
            score_col = self._get_score_column(importance_df)
            # 重要性已降序排列，名次按行顺序而非原始索引计算
            for rank, (_, row) in enumerate(importance_df.iterrows(), start=1):
                feature = row['feature']

                if feature not in feature_rankings:
                    feature_rankings[feature] = {}

                feature_rankings[feature][f'{method}_rank'] = rank
                feature_rankings[feature][f'{method}_score'] = row[score_col]

        # 转换为DataFrame
        consensus_data = []
        for feature, ranks in feature_rankings.items():
            # 计算平均排名
            rank_values = [v for k, v in ranks.items() if k.endswith('_rank')]
            avg_rank = np.mean(rank_values) if rank_values else float('inf')

            # 计算标准化分数的平均值
            score_values = []
            for method in methods:
                score_key = f'{method}_score'
                if score_key in ranks:
                    score_values.append(ranks[score_key])

            avg_score = np.mean(score_values) if score_values else 0

            row_data = {
                'feature': feature,
                'avg_rank': avg_rank,
                'avg_score': avg_score,
                'methods_count': len([k for k in ranks.keys() if k.endswith('_rank')])
            }

            # 添加各方法的具体排名和分数
            row_data.update(ranks)

            consensus_data.append(row_data)

        # 创建综合排名DataFrame
        consensus_df = pd.DataFrame(consensus_data)
        consensus_df = consensus_df.sort_values('avg_rank')

        if top_k:
            consensus_df = consensus_df.head(top_k)

        return consensus_df

    def get_stable_features(self,
                          methods: Optional[List[str]] = None,
                          top_k_per_method: int = 20,
                          min_appearances: int = 2) -> List[str]:
        """
        获取稳定的重要特征

        Parameters:
        -----------
        methods : list, optional
            参与分析的方法
        top_k_per_method : int, default=20
            每个方法选择的top特征数量
        min_appearances : int, default=2
            最少出现在几个方法的top列表中

        Returns:
        --------
        stable_features : list
            稳定的重要特征列表
        """
        if not self.importance_results_:
            raise ValueError("请先计算特征重要性")

        available_methods = list(self.importance_results_.keys())

        if methods is None:
            methods = available_methods
        else:
            methods = [m for m in methods if m in available_methods]

        # 统计每个特征在各方法top_k中的出现次数
        feature_counts = {}

        for method in methods:
            importance_df = self.importance_results_[method]
            top_features = importance_df.head(top_k_per_method)['feature'].tolist()

            for feature in top_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # 筛选出现次数达到要求的特征
        stable_features = [
            feature for feature, count in feature_counts.items()
            if count >= min_appearances
        ]

        return stable_features

    def plot_importance_comparison(self,
                                 methods: Optional[List[str]] = None,
                                 top_k: int = 15,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        绘制重要性对比图

        Parameters:
        -----------
        methods : list, optional
            要对比的方法
        top_k : int, default=15
            显示前k个特征
        figsize : tuple, default=(12, 8)
            图形大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("需要安装matplotlib和seaborn来绘制图形")
            return

        if not self.importance_results_:
            raise ValueError("请先计算特征重要性")

        available_methods = list(self.importance_results_.keys())

        if methods is None:
            methods = available_methods
        else:
            methods = [m for m in methods if m in available_methods]

        # 获取综合排名的top特征
        consensus_df = self.get_consensus_ranking(methods, top_k)
        top_features = consensus_df['feature'].tolist()

        # 准备绘图数据
        plot_data = []
        for method in methods:
            importance_df = self.importance_results_[method]
            score_col = self._get_score_column(importance_df)
            method_data = importance_df[importance_df['feature'].isin(top_features)]

            for _, row in method_data.iterrows():
                plot_data.append({
                    'feature': row['feature'],
                    'importance': row[score_col],
                    'method': method
                })

        plot_df = pd.DataFrame(plot_data)

        # 创建图形
        plt.figure(figsize=figsize)

        # 使用pivot创建热力图数据
        heatmap_data = plot_df.pivot(index='feature', columns='method', values='importance')

        # 按综合排名排序特征
        feature_order = [f for f in top_features if f in heatmap_data.index]
        heatmap_data = heatmap_data.reindex(feature_order)

        # 绘制热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Feature Importance'})

        plt.title(f'特征重要性对比 (Top {top_k})')
        plt.xlabel('重要性计算方法')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.show()

    def export_importance_report(self, filepath: str) -> None:
        """
        导出重要性分析报告

        Parameters:
        -----------
        filepath : str
            导出文件路径
        """
        if not self.importance_results_:
            raise ValueError("请先计算特征重要性")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 导出各方法的重要性结果
            for method, importance_df in self.importance_results_.items():
                importance_df.to_excel(writer, sheet_name=f'{method}_importance', index=False)

            # 导出综合排名
            try:
                consensus_df = self.get_consensus_ranking()
                consensus_df.to_excel(writer, sheet_name='consensus_ranking', index=False)
            except Exception as e:
                warnings.warn(f"导出综合排名失败: {str(e)}")

            # 导出稳定特征
            try:
                stable_features = self.get_stable_features()
                stable_df = pd.DataFrame({'stable_features': stable_features})
                stable_df.to_excel(writer, sheet_name='stable_features', index=False)
            except Exception as e:
                warnings.warn(f"导出稳定特征失败: {str(e)}")

        print(f"重要性分析报告已导出到: {filepath}")

    # 便捷方法别名，用于向后兼容
    def calculate_random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """计算随机森林重要性（便捷方法）"""
        return self.calculate_tree_importance(X, y, method='random_forest')

    def calculate_iv_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """计算IV重要性（便捷方法）"""
        return self.calculate_univariate_importance(X, y)

    def calculate_gbdt_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """计算GBDT重要性（便捷方法）"""
        return self.calculate_tree_importance(X, y, method='gbdt')