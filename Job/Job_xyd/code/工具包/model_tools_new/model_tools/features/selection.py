"""
特征选择模块

提供多种特征选择方法，包括基于统计的、基于模型的特征选择
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def filter_features_by_single_value_ratio(data: pd.DataFrame,
                                         threshold: float = 0.95,
                                         exclude_cols: Optional[List[str]] = None) -> Tuple[List[str], pd.DataFrame]:
    """
    基于单一值占比筛选特征

    如果某个特征的最频繁值（包含空值）占总数据量的比例超过阈值，则去掉该特征

    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    threshold : float, default=0.95
        单一值占比阈值
    exclude_cols : list, optional
        排除检查的列名列表（如目标变量）

    Returns:
    --------
    selected_features : list
        保留的特征列表
    filter_log : pd.DataFrame
        筛选日志记录
    """
    exclude_cols = exclude_cols or []
    check_cols = [col for col in data.columns if col not in exclude_cols]

    filter_results = []
    selected_features = []

    for col in check_cols:
        # 计算每个值的频次（包括NaN）
        value_counts = data[col].value_counts(dropna=False)
        total_count = len(data)

        if len(value_counts) == 0:
            # 空列
            max_ratio = 1.0
            most_frequent_value = None
            reason = "空列"
            keep_feature = False
        else:
            # 最频繁值及其占比
            most_frequent_value = value_counts.index[0]
            max_count = value_counts.iloc[0]
            max_ratio = max_count / total_count

            if max_ratio >= threshold:
                reason = f"最频繁值占比{max_ratio:.2%}超过阈值{threshold:.2%}"
                keep_feature = False
            else:
                reason = f"最频繁值占比{max_ratio:.2%}，通过筛选"
                keep_feature = True
                selected_features.append(col)

        filter_results.append({
            'feature': col,
            'keep_feature': keep_feature,
            'most_frequent_value': most_frequent_value,
            'max_value_ratio': max_ratio,
            'unique_count': data[col].nunique(dropna=False),
            'missing_ratio': data[col].isna().sum() / total_count,
            'filter_reason': reason
        })

    filter_log = pd.DataFrame(filter_results)

    return selected_features, filter_log


def calculate_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
    """
    计算Information Value (IV)

    Parameters:
    -----------
    feature : pd.Series
        特征变量
    target : pd.Series
        目标变量 (0/1)
    bins : int, default=10
        分箱数量

    Returns:
    --------
    iv : float
        IV值
    """
    try:
        # 处理缺失值
        df = pd.DataFrame({'feature': feature, 'target': target}).dropna()

        if len(df) == 0:
            return 0.0

        # 分箱
        df['bin'] = pd.qcut(df['feature'], q=bins, duplicates='drop', precision=3)

        # 计算每个分箱的统计量
        grouped = df.groupby('bin')['target'].agg(['count', 'sum'])
        grouped['good'] = grouped['count'] - grouped['sum']  # 好客户数
        grouped['bad'] = grouped['sum']  # 坏客户数

        # 计算总的好坏客户数
        total_good = grouped['good'].sum()
        total_bad = grouped['bad'].sum()

        if total_good == 0 or total_bad == 0:
            return 0.0

        # 计算分布占比
        grouped['good_rate'] = grouped['good'] / total_good
        grouped['bad_rate'] = grouped['bad'] / total_bad

        # 计算WOE和IV
        grouped['woe'] = np.log(grouped['bad_rate'] / grouped['good_rate'])
        grouped['iv_component'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']

        # 处理无穷大值
        grouped = grouped.replace([np.inf, -np.inf], 0)

        return grouped['iv_component'].sum()

    except Exception as e:
        warnings.warn(f"计算IV时出错: {str(e)}")
        return 0.0


class FeatureSelector:
    """
    特征选择器

    集成多种特征选择方法，包括基于单一值占比的预筛选
    """

    def __init__(self,
                 method: str = 'iv',
                 k_features: Optional[int] = None,
                 threshold: Optional[float] = None,
                 single_value_threshold: float = 0.95,
                 iv_threshold: float = 0.1):
        """
        初始化特征选择器

        Parameters:
        -----------
        method : str, default='iv'
            选择方法 ('iv', 'chi2', 'f_classif', 'mutual_info', 'rfe', 'lasso')
        k_features : int, optional
            选择的特征数量
        threshold : float, optional
            选择阈值
        single_value_threshold : float, default=0.95
            单一值占比阈值，超过此比例的特征将被预先过滤
        iv_threshold : float, default=0.1
            IV阈值，小于此值的特征将被过滤
        """
        self.method = method
        self.k_features = k_features
        self.threshold = threshold
        self.single_value_threshold = single_value_threshold
        self.iv_threshold = iv_threshold
        self.selected_features_ = None
        self.feature_scores_ = None
        self.filter_log_ = None
        self.pre_filtered_features_ = None
        self.iv_log_ = None

    def _fit_iv_selection(self, X: pd.DataFrame, y: pd.Series):
        """
        基于IV的特征选择

        添加IV阈值筛选：小于阈值的特征会被去掉
        """
        data = X.copy()
        data['target'] = y

        # 计算所有特征的IV值
        iv_results = []
        for feature in X.columns:
            try:
                iv_value = calculate_iv(data[feature], data['target'])

                # 判断是否通过IV阈值
                if iv_value < self.iv_threshold:
                    reason = f"IV值{iv_value:.4f}小于阈值{self.iv_threshold}"
                    keep_feature = False
                else:
                    reason = f"IV值{iv_value:.4f}通过阈值筛选"
                    keep_feature = True

                # IV值解释
                if iv_value < 0.02:
                    interpretation = "无预测能力"
                elif iv_value < 0.1:
                    interpretation = "弱预测能力"
                elif iv_value < 0.3:
                    interpretation = "中等预测能力"
                elif iv_value < 0.5:
                    interpretation = "强预测能力"
                else:
                    interpretation = "过强预测能力(可能过拟合)"

                iv_results.append({
                    'feature': feature,
                    'iv_value': iv_value,
                    'interpretation': interpretation,
                    'keep_feature': keep_feature,
                    'filter_reason': reason
                })

            except Exception as e:
                warnings.warn(f"特征 {feature} IV计算失败: {str(e)}")
                iv_results.append({
                    'feature': feature,
                    'iv_value': 0.0,
                    'interpretation': "计算失败",
                    'keep_feature': False,
                    'filter_reason': f"IV计算失败: {str(e)}"
                })

        # 保存IV筛选日志
        self.iv_log_ = pd.DataFrame(iv_results).sort_values('iv_value', ascending=False)

        # 通过IV阈值的特征
        iv_passed_features = self.iv_log_[self.iv_log_['keep_feature']]['feature'].tolist()

        # 设置特征得分
        self.feature_scores_ = dict(zip(self.iv_log_['feature'], self.iv_log_['iv_value']))

        # 根据参数进一步筛选
        if self.k_features:
            # 选择IV最高的k个特征（在通过阈值的特征中）
            top_features = self.iv_log_[self.iv_log_['keep_feature']].head(self.k_features)
            self.selected_features_ = top_features['feature'].tolist()
        elif self.threshold:
            # 使用自定义阈值（这里threshold指代其他阈值，不是iv_threshold）
            self.selected_features_ = self.iv_log_[
                (self.iv_log_['keep_feature']) &
                (self.iv_log_['iv_value'] >= self.threshold)
            ]['feature'].tolist()
        else:
            # 默认选择所有通过IV阈值的特征
            self.selected_features_ = iv_passed_features

        print(f"IV筛选完成：通过IV阈值({self.iv_threshold})的特征数量: {len(iv_passed_features)}")
        print(f"最终选择特征数量: {len(self.selected_features_)}")

    def fit(self, X: pd.DataFrame, y: pd.Series, target_col: Optional[str] = None) -> 'FeatureSelector':
        """
        拟合特征选择器

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        target_col : str, optional
            目标变量列名，用于排除预筛选

        Returns:
        --------
        self : FeatureSelector
            拟合后的选择器
        """
        # 第一步：基于单一值占比预筛选
        exclude_cols = [target_col] if target_col else []
        self.pre_filtered_features_, self.filter_log_ = filter_features_by_single_value_ratio(
            X, threshold=self.single_value_threshold, exclude_cols=exclude_cols
        )

        print(f"预筛选完成：{len(X.columns)} -> {len(self.pre_filtered_features_)} 个特征")

        # 使用预筛选后的特征进行后续选择
        X_filtered = X[self.pre_filtered_features_]

        if len(self.pre_filtered_features_) == 0:
            print("警告：预筛选后没有剩余特征")
            self.selected_features_ = []
            self.feature_scores_ = {}
            return self

        # 第二步：应用指定的特征选择方法
        if self.method == 'iv':
            self._fit_iv_selection(X_filtered, y)
        elif self.method == 'rfe':
            self._fit_rfe_selection(X_filtered, y)
        elif self.method == 'lasso':
            self._fit_lasso_selection(X_filtered, y)
        elif self.method == 'chi2':
            self._fit_statistical_selection(X_filtered, y, chi2)
        elif self.method == 'f_classif':
            self._fit_statistical_selection(X_filtered, y, f_classif)
        elif self.method == 'mutual_info':
            self._fit_mutual_info_selection(X_filtered, y)
        else:
            raise ValueError(f"不支持的选择方法: {self.method}")

        return self

    def get_iv_log(self) -> pd.DataFrame:
        """
        获取IV筛选日志

        Returns:
        --------
        iv_log : pd.DataFrame
            IV计算和筛选的详细记录
        """
        if self.iv_log_ is None:
            raise ValueError("请先调用fit方法")

        return self.iv_log_

    def get_filter_log(self) -> pd.DataFrame:
        """
        获取预筛选日志

        Returns:
        --------
        filter_log : pd.DataFrame
            详细的预筛选记录
        """
        if self.filter_log_ is None:
            raise ValueError("请先调用fit方法")

        return self.filter_log_

    def get_selection_summary(self) -> Dict:
        """
        获取特征选择总结

        Returns:
        --------
        summary : dict
            选择总结信息
        """
        if self.filter_log_ is None or self.iv_log_ is None:
            raise ValueError("请先调用fit方法")

        pre_filter_removed = len(self.filter_log_[~self.filter_log_['keep_feature']])
        iv_filter_removed = len(self.iv_log_[~self.iv_log_['keep_feature']])
        final_selected = len(self.selected_features_) if self.selected_features_ else 0

        return {
            'original_features': len(self.filter_log_),
            'pre_filter_removed': pre_filter_removed,
            'pre_filter_remaining': len(self.pre_filtered_features_),
            'iv_filter_removed': iv_filter_removed,
            'iv_filter_remaining': len(self.iv_log_[self.iv_log_['keep_feature']]),
            'final_selected': final_selected,
            'single_value_threshold': self.single_value_threshold,
            'iv_threshold': self.iv_threshold
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，只保留选中的特征

        Parameters:
        -----------
        X : pd.DataFrame
            输入数据

        Returns:
        --------
        X_selected : pd.DataFrame
            选中特征的数据
        """
        if self.selected_features_ is None:
            raise ValueError("请先调用fit方法")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        拟合并转换数据

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        target_col : str, optional
            目标变量列名

        Returns:
        --------
        X_selected : pd.DataFrame
            选中特征的数据
        """
        return self.fit(X, y, target_col).transform(X)

    def _fit_rfe_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于递归特征消除的选择"""
        estimator = LogisticRegression(random_state=42, max_iter=1000)

        n_features = self.k_features or max(1, len(X.columns) // 2)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)

        self.selected_features_ = X.columns[rfe.support_].tolist()
        self.feature_scores_ = dict(zip(X.columns, rfe.ranking_))

    def _fit_lasso_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于Lasso的特征选择"""
        from sklearn.linear_model import LassoCV

        # 使用交叉验证选择最优alpha
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X, y)

        # 选择系数非零的特征
        selected_mask = lasso.coef_ != 0
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.feature_scores_ = dict(zip(X.columns, np.abs(lasso.coef_)))

    def _fit_statistical_selection(self, X: pd.DataFrame, y: pd.Series, score_func):
        """基于统计检验的特征选择"""
        # 处理负值（chi2要求非负值）
        if score_func == chi2:
            X_processed = X.copy()
            # 将负值设为0
            X_processed[X_processed < 0] = 0
        else:
            X_processed = X

        if self.k_features:
            selector = SelectKBest(score_func=score_func, k=self.k_features)
        else:
            selector = SelectKBest(score_func=score_func, k='all')

        selector.fit(X_processed, y)

        self.feature_scores_ = dict(zip(X.columns, selector.scores_))

        if self.k_features:
            self.selected_features_ = X.columns[selector.get_support()].tolist()
        elif self.threshold:
            # 使用分数阈值
            scores = selector.scores_
            selected_mask = scores >= self.threshold
            self.selected_features_ = X.columns[selected_mask].tolist()
        else:
            # 选择前50%的特征
            k = max(1, len(X.columns) // 2)
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X_processed, y)
            self.selected_features_ = X.columns[selector.get_support()].tolist()

    def _fit_mutual_info_selection(self, X: pd.DataFrame, y: pd.Series):
        """基于互信息的特征选择"""
        scores = mutual_info_classif(X, y, random_state=42)
        self.feature_scores_ = dict(zip(X.columns, scores))

        if self.k_features:
            # 选择得分最高的k个特征
            sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
            self.selected_features_ = [f[0] for f in sorted_features[:self.k_features]]
        elif self.threshold:
            self.selected_features_ = [f for f, score in self.feature_scores_.items() if score >= self.threshold]
        else:
            # 选择得分大于中位数的特征
            median_score = np.median(scores)
            self.selected_features_ = [f for f, score in self.feature_scores_.items() if score >= median_score]