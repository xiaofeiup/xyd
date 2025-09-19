"""
评估模块

提供模型评估指标计算功能，包括AUC、KS、PSI、Lift等
"""

from .metrics import (
    calculate_auc,
    calculate_ks,
    calculate_lift,
    calculate_psi,
    calculate_multi_feature_psi,
    interpret_psi,
    calculate_gain,
    ModelEvaluator
)

__all__ = [
    # 核心指标计算
    'calculate_auc',
    'calculate_ks',
    'calculate_lift',
    'calculate_psi',
    'calculate_multi_feature_psi',
    'interpret_psi',
    'calculate_gain',

    # 模型评估器
    'ModelEvaluator'
]