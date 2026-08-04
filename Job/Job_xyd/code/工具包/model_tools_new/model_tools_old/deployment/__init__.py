"""
模型部署模块

提供 pkl(joblib) -> PMML 转换、模型加载预测、以及
线下 pkl 与线上 PMML 预测一致性验证的工具。
"""

from .model_deployment import (
    ModelDeployer,
    ConsistencyResult,
    convert_pkl_to_pmml,
    verify_pkl_pmml_consistency,
)

__all__ = [
    'ModelDeployer',
    'ConsistencyResult',
    'convert_pkl_to_pmml',
    'verify_pkl_pmml_consistency',
]
