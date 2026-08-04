"""
特征编码模块

提供WOE编码、目标编码等特征转换方法
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


class WOEEncoder:
    """
    WOE编码器

    Weight of Evidence编码，常用于信用风险建模
    """

    def __init__(self,
                 bins: int = 10,
                 min_bin_size: float = 0.05,
                 smooth: float = 0.5):
        """
        初始化WOE编码器

        Parameters:
        -----------
        bins : int, default=10
            分箱数量
        min_bin_size : float, default=0.05
            最小分箱大小（占总样本比例）
        smooth : float, default=0.5
            平滑参数，防止除零
        """
        self.bins = bins
        self.min_bin_size = min_bin_size
        self.smooth = smooth
        self.woe_mappings_ = {}
        self.iv_values_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WOEEncoder':
        """
        拟合WOE编码器

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量 (0/1)

        Returns:
        --------
        self : WOEEncoder
            拟合后的编码器
        """
        self.woe_mappings_ = {}
        self.iv_values_ = {}

        for column in X.columns:
            woe_mapping, iv_value = self._calculate_woe(X[column], y, column)
            self.woe_mappings_[column] = woe_mapping
            self.iv_values_[column] = iv_value

        return self

    def _calculate_woe(self, feature: pd.Series, target: pd.Series, feature_name: str) -> Tuple[Dict, float]:
        """
        计算单个特征的WOE映射

        Parameters:
        -----------
        feature : pd.Series
            特征数据
        target : pd.Series
            目标变量
        feature_name : str
            特征名称

        Returns:
        --------
        woe_mapping : dict
            WOE映射字典
        iv_value : float
            IV值
        """
        # 创建数据框
        df = pd.DataFrame({
            'feature': feature,
            'target': target
        }).dropna()

        if len(df) == 0:
            return {}, 0.0

        try:
            # 对数值型特征进行分箱
            if pd.api.types.is_numeric_dtype(feature):
                df['bin'] = pd.qcut(df['feature'], q=self.bins, duplicates='drop', precision=3)
            else:
                # 分类特征直接使用原值作为分箱
                df['bin'] = df['feature']

            # 计算每个分箱的统计量
            grouped = df.groupby('bin')['target'].agg(['count', 'sum'])
            grouped['good'] = grouped['count'] - grouped['sum']  # 好样本数
            grouped['bad'] = grouped['sum']  # 坏样本数

            # 计算总的好坏样本数
            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()

            if total_good == 0 or total_bad == 0:
                return {}, 0.0

            # 添加平滑
            grouped['good_smooth'] = grouped['good'] + self.smooth
            grouped['bad_smooth'] = grouped['bad'] + self.smooth
            total_good_smooth = total_good + self.smooth * len(grouped)
            total_bad_smooth = total_bad + self.smooth * len(grouped)

            # 计算分布占比
            grouped['good_rate'] = grouped['good_smooth'] / total_good_smooth
            grouped['bad_rate'] = grouped['bad_smooth'] / total_bad_smooth

            # 计算WOE
            grouped['woe'] = np.log(grouped['bad_rate'] / grouped['good_rate'])

            # 计算IV
            grouped['iv_component'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']
            iv_value = grouped['iv_component'].sum()

            # 创建WOE映射
            woe_mapping = {}

            if pd.api.types.is_numeric_dtype(feature):
                # 数值型特征：使用区间映射
                for bin_interval, woe_value in zip(grouped.index, grouped['woe']):
                    woe_mapping[bin_interval] = woe_value
            else:
                # 分类特征：直接映射
                for category, woe_value in zip(grouped.index, grouped['woe']):
                    woe_mapping[category] = woe_value

            return woe_mapping, iv_value

        except Exception as e:
            warnings.warn(f"特征 {feature_name} WOE计算失败: {str(e)}")
            return {}, 0.0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用WOE编码

        Parameters:
        -----------
        X : pd.DataFrame
            待编码的特征数据

        Returns:
        --------
        X_woe : pd.DataFrame
            WOE编码后的数据
        """
        if not self.woe_mappings_:
            raise ValueError("请先调用fit方法")

        X_woe = X.copy()

        for column in X.columns:
            if column in self.woe_mappings_:
                X_woe[column] = self._apply_woe_mapping(X[column], self.woe_mappings_[column])

        return X_woe

    def _apply_woe_mapping(self, feature: pd.Series, woe_mapping: Dict) -> pd.Series:
        """
        应用WOE映射到特征

        Parameters:
        -----------
        feature : pd.Series
            特征数据
        woe_mapping : dict
            WOE映射字典

        Returns:
        --------
        woe_feature : pd.Series
            WOE编码后的特征
        """
        if not woe_mapping:
            return feature.fillna(0)

        # 判断是数值型还是分类型
        if pd.api.types.is_numeric_dtype(feature):
            # 数值型：使用区间映射
            woe_values = []
            for value in feature:
                if pd.isna(value):
                    woe_values.append(0)  # 缺失值处理
                    continue

                # 找到对应的区间
                matched_woe = 0
                for interval, woe in woe_mapping.items():
                    try:
                        if value in interval:
                            matched_woe = woe
                            break
                    except:
                        # 处理可能的区间比较错误
                        continue

                woe_values.append(matched_woe)

            return pd.Series(woe_values, index=feature.index)

        else:
            # 分类型：直接映射
            return feature.map(woe_mapping).fillna(0)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        拟合并转换

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量

        Returns:
        --------
        X_woe : pd.DataFrame
            WOE编码后的数据
        """
        return self.fit(X, y).transform(X)

    def get_woe_summary(self) -> pd.DataFrame:
        """
        获取WOE编码总结

        Returns:
        --------
        summary_df : pd.DataFrame
            WOE编码总结
        """
        if not self.woe_mappings_:
            raise ValueError("请先调用fit方法")

        summary_data = []
        for feature, woe_mapping in self.woe_mappings_.items():
            iv_value = self.iv_values_.get(feature, 0)

            summary_data.append({
                'feature': feature,
                'iv_value': iv_value,
                'num_bins': len(woe_mapping),
                'has_mapping': len(woe_mapping) > 0
            })

        return pd.DataFrame(summary_data).sort_values('iv_value', ascending=False)


class TargetEncoder:
    """
    目标编码器

    用目标变量的均值来编码分类特征
    """

    def __init__(self,
                 smooth: float = 1.0,
                 min_samples_leaf: int = 1):
        """
        初始化目标编码器

        Parameters:
        -----------
        smooth : float, default=1.0
            平滑参数
        min_samples_leaf : int, default=1
            叶子节点最小样本数
        """
        self.smooth = smooth
        self.min_samples_leaf = min_samples_leaf
        self.target_encodings_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TargetEncoder':
        """
        拟合目标编码器

        Parameters:
        -----------
        X : pd.DataFrame
            分类特征数据
        y : pd.Series
            目标变量

        Returns:
        --------
        self : TargetEncoder
            拟合后的编码器
        """
        self.global_mean_ = y.mean()
        self.target_encodings_ = {}

        for column in X.columns:
            encoding_map = self._calculate_target_encoding(X[column], y)
            self.target_encodings_[column] = encoding_map

        return self

    def _calculate_target_encoding(self, feature: pd.Series, target: pd.Series) -> Dict:
        """
        计算单个特征的目标编码

        Parameters:
        -----------
        feature : pd.Series
            特征数据
        target : pd.Series
            目标变量

        Returns:
        --------
        encoding_map : dict
            目标编码映射
        """
        # 计算每个类别的统计量
        grouped = pd.DataFrame({'feature': feature, 'target': target}).groupby('feature')['target']
        counts = grouped.count()
        means = grouped.mean()

        encoding_map = {}

        for category in counts.index:
            count = counts[category]
            mean = means[category]

            # 应用平滑
            if count >= self.min_samples_leaf:
                # 使用贝叶斯平滑
                smoothed_mean = (count * mean + self.smooth * self.global_mean_) / (count + self.smooth)
            else:
                # 样本太少，使用全局均值
                smoothed_mean = self.global_mean_

            encoding_map[category] = smoothed_mean

        return encoding_map

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用目标编码

        Parameters:
        -----------
        X : pd.DataFrame
            待编码的特征数据

        Returns:
        --------
        X_encoded : pd.DataFrame
            目标编码后的数据
        """
        if not self.target_encodings_:
            raise ValueError("请先调用fit方法")

        X_encoded = X.copy()

        for column in X.columns:
            if column in self.target_encodings_:
                X_encoded[column] = X[column].map(self.target_encodings_[column]).fillna(self.global_mean_)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        拟合并转换

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量

        Returns:
        --------
        X_encoded : pd.DataFrame
            目标编码后的数据
        """
        return self.fit(X, y).transform(X)


def create_polynomial_features(X: pd.DataFrame,
                             degree: int = 2,
                             include_bias: bool = False,
                             interaction_only: bool = False) -> pd.DataFrame:
    """
    创建多项式特征

    Parameters:
    -----------
    X : pd.DataFrame
        输入特征
    degree : int, default=2
        多项式度数
    include_bias : bool, default=False
        是否包含偏置项
    interaction_only : bool, default=False
        是否只包含交互项

    Returns:
    --------
    X_poly : pd.DataFrame
        多项式特征
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only
    )

    X_poly_array = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    return pd.DataFrame(X_poly_array, columns=feature_names, index=X.index)


def create_interaction_features(X: pd.DataFrame,
                               feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """
    创建交互特征

    Parameters:
    -----------
    X : pd.DataFrame
        输入特征
    feature_pairs : list of tuples, optional
        指定的特征对，如果为None则创建所有可能的交互

    Returns:
    --------
    X_interact : pd.DataFrame
        包含交互特征的数据
    """
    X_interact = X.copy()

    if feature_pairs is None:
        # 创建所有可能的两两交互
        feature_pairs = []
        features = X.columns.tolist()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feature_pairs.append((features[i], features[j]))

    # 创建交互特征
    for feat1, feat2 in feature_pairs:
        if feat1 in X.columns and feat2 in X.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            X_interact[interaction_name] = X[feat1] * X[feat2]

    return X_interact