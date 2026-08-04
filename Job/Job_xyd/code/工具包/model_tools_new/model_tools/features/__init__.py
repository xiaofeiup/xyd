"""
特征工程模块

提供特征选择、特征重要性分析、特征编码、数据分析等功能
"""

from .selection import (
    FeatureSelector,
    filter_features_by_single_value_ratio,
    calculate_iv
)

from .importance import FeatureImportanceAnalyzer

from .encoding import (
    WOEEncoder,
    TargetEncoder,
    create_polynomial_features,
    create_interaction_features
)

from .data_analysis import (
    DataAnalyzer,
    analyze_missing
)
from .streaming_feature_selection import StreamingFeatureSelector

__all__ = [
    # 特征选择
    'FeatureSelector',
    'filter_features_by_single_value_ratio',
    'calculate_iv',

    # 特征重要性
    'FeatureImportanceAnalyzer',

    # 特征编码
    'WOEEncoder',
    'TargetEncoder',
    'create_polynomial_features',
    'create_interaction_features',

    # 数据分析
    'DataAnalyzer',
    'analyze_missing',
    'StreamingFeatureSelector'
]
