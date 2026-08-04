"""
超参数优化模块

提供基于Optuna的智能超参数搜索功能，包括：
- 考虑参数约束关系的搜索空间定义
- 多种优化目标（AUC、KS、ROI等）
- 常见ML模型的预定义配置
- 业务指标导向的优化
"""

from .hyperparameter_tuner import (
    HyperparameterTuner,
    create_lgb_tuner,
    create_xgb_tuner
)

from .parameter_spaces import (
    ParameterSpace,
    LightGBMSpace,
    XGBoostSpace,
    RandomForestSpace,
    SVMSpace,
    LogisticRegressionSpace,
    CreditScoringSpace,
    get_parameter_space,
    create_custom_space,
    PARAMETER_SPACES
)

from .objectives import (
    ObjectiveFunction,
    AUCObjective,
    KSObjective,
    LiftObjective,
    F1Objective,
    PrecisionRecallObjective,
    BusinessROIObjective,
    StabilityAwareObjective,
    MultiObjective,
    CustomObjective,
    RMSEObjective,
    MAEObjective,
    R2Objective,
    get_objective_function,
    create_credit_scoring_objective,
    OBJECTIVE_FUNCTIONS
)

__all__ = [
    # 主要优化器
    'HyperparameterTuner',
    'create_lgb_tuner',
    'create_xgb_tuner',

    # 参数空间
    'ParameterSpace',
    'LightGBMSpace',
    'XGBoostSpace',
    'RandomForestSpace',
    'SVMSpace',
    'LogisticRegressionSpace',
    'CreditScoringSpace',
    'get_parameter_space',
    'create_custom_space',
    'PARAMETER_SPACES',

    # 目标函数
    'ObjectiveFunction',
    'AUCObjective',
    'KSObjective',
    'LiftObjective',
    'F1Objective',
    'PrecisionRecallObjective',
    'BusinessROIObjective',
    'StabilityAwareObjective',
    'MultiObjective',
    'CustomObjective',
    'RMSEObjective',
    'MAEObjective',
    'R2Objective',
    'get_objective_function',
    'create_credit_scoring_objective',
    'OBJECTIVE_FUNCTIONS'
]