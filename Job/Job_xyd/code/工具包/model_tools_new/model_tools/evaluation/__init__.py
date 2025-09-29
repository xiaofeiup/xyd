"""
评估模块

提供模型评估指标计算功能，包括AUC、KS、PSI、Lift等
"""

from .metrics import (
    # 分类模型指标
    calculate_auc,
    calculate_ks,
    calculate_lift,
    calculate_psi,
    calculate_multi_feature_psi,
    interpret_psi,
    calculate_gain,
    ModelEvaluator,

    # 回归模型指标
    calculate_rmse,
    calculate_mse,
    calculate_mae,
    calculate_r2,
    calculate_adjusted_r2,
    calculate_mape,
    calculate_smape,
    calculate_rmsle,
    calculate_huber_loss,
    calculate_quantile_loss,
    calculate_regression_residuals,
    evaluate_regression_model,
    compare_regression_models
)

from .roi_evaluation import (
    ModelROIEvaluator
)

__all__ = [
    # 分类模型指标
    'calculate_auc',
    'calculate_ks',
    'calculate_lift',
    'calculate_psi',
    'calculate_multi_feature_psi',
    'interpret_psi',
    'calculate_gain',

    # 回归模型指标
    'calculate_rmse',
    'calculate_mse',
    'calculate_mae',
    'calculate_r2',
    'calculate_adjusted_r2',
    'calculate_mape',
    'calculate_smape',
    'calculate_rmsle',
    'calculate_huber_loss',
    'calculate_quantile_loss',
    'calculate_regression_residuals',
    'evaluate_regression_model',
    'compare_regression_models',

    # 模型评估器
    'ModelEvaluator',
    'ModelROIEvaluator'
]