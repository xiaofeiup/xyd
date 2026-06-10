"""
Model Tools 2.0 - 专业的机器学习模型工具包

提供特征工程、模型评估、监控与报告等全链路能力。
"""

from importlib import import_module
from typing import Dict, Any, Optional, Set

__version__ = "2.0.0"
__author__ = "Model Tools Team"
__email__ = "model-tools@example.com"

# 核心模块
from . import evaluation
from . import monitoring
from . import features
from . import utils

# 评估模块快捷入口
from .evaluation.metrics import (
    calculate_auc,
    calculate_ks,
    calculate_psi,
    ModelEvaluator,
)
from .evaluation.reports import (
    ModelReportGenerator,
    generate_model_report,
)
from .evaluation.visualization import (
    ModelVisualizationTool,
    quick_model_visualization,
)
from .evaluation.analysis import (
    ModelAnalyzer,
    analyze_model_comprehensive,
)

# 监控模块快捷入口
from .monitoring.stability import ModelStabilityMonitor
from .monitoring.drift import FeatureDriftDetector
from .monitoring.alerting import AlertManager, AlertEngine
from .monitoring.workflow import MonitoringWorkflow
from .monitoring.performance import (
    PerformanceTracker,
    ResourceMonitor,
)

# 特征模块快捷入口
from .features.selection import (
    FeatureSelector,
    filter_features_by_single_value_ratio,
)
from .features.importance import FeatureImportanceAnalyzer
from .features.encoding import WOEEncoder, TargetEncoder
from .features.validation import (
    FeatureValidator,
    validate_dataset,
    compare_datasets,
)

# 通用工具
from .utils.data_processing import check_data_quality
from .utils.memory_tools import whos
from .utils.config import (
    ConfigManager,
    get_config,
    set_config,
)
from .utils.decorators import (
    timer,
    retry,
    validate_inputs,
    cache_result,
    log_calls,
    deprecated,
    monitor_performance,
    rate_limit,
    ml_function,
)

__all__ = [
    # 元信息
    '__version__',
    '__author__',
    '__email__',

    # 子模块
    'evaluation',
    'monitoring',
    'features',
    'utils',

    # 评估
    'calculate_auc',
    'calculate_ks',
    'calculate_psi',
    'ModelEvaluator',
    'ModelReportGenerator',
    'generate_model_report',
    'ModelVisualizationTool',
    'quick_model_visualization',
    'ModelAnalyzer',
    'analyze_model_comprehensive',

    # 监控
    'ModelStabilityMonitor',
    'FeatureDriftDetector',
    'AlertManager',
    'AlertEngine',
    'MonitoringWorkflow',
    'PerformanceTracker',
    'ResourceMonitor',

    # 特征工程
    'FeatureSelector',
    'filter_features_by_single_value_ratio',
    'FeatureImportanceAnalyzer',
    'WOEEncoder',
    'TargetEncoder',
    'FeatureValidator',
    'validate_dataset',
    'compare_datasets',

    # 工具函数 / 装饰器
    'check_data_quality',
    'whos',
    'ConfigManager',
    'get_config',
    'set_config',
    'timer',
    'retry',
    'validate_inputs',
    'cache_result',
    'log_calls',
    'deprecated',
    'monitor_performance',
    'rate_limit',
    'ml_function',

    # 通用信息函数
    'get_version',
    'get_module_info',
    'print_welcome',
]

# ------------------ 可选模块 (optimization) ------------------
_OPTIONAL_EXPORTS: Set[str] = {
    'optimization',
    'HyperparameterTuner',
    'create_lgb_tuner',
    'create_xgb_tuner',
    'AUCObjective',
    'KSObjective',
    'BusinessROIObjective',
    'create_credit_scoring_objective',
    'get_parameter_space',
    'CreditScoringSpace',
}
_missing_optional_names: Set[str] = set()
_optional_error: Optional[Exception] = None

try:
    _optimization_module = import_module('.optimization', __name__)
    from .optimization.hyperparameter_tuner import (
        HyperparameterTuner,
        create_lgb_tuner,
        create_xgb_tuner,
    )
    from .optimization.objectives import (
        AUCObjective,
        KSObjective,
        BusinessROIObjective,
        create_credit_scoring_objective,
    )
    from .optimization.parameter_spaces import (
        get_parameter_space,
        CreditScoringSpace,
    )
except Exception as exc:
    _optional_error = exc
    _missing_optional_names = set(_OPTIONAL_EXPORTS)
else:
    optimization = _optimization_module  # type: ignore[assignment]
    __all__.append('optimization')
    __all__.extend([
        'HyperparameterTuner',
        'create_lgb_tuner',
        'create_xgb_tuner',
        'AUCObjective',
        'KSObjective',
        'BusinessROIObjective',
        'create_credit_scoring_objective',
        'get_parameter_space',
        'CreditScoringSpace',
    ])
finally:
    if '_optimization_module' in locals():
        del _optimization_module


def __getattr__(name: str) -> Any:  # pragma: no cover - 动态属性访问
    """为缺失的可选依赖提供友好的错误信息。"""
    if name in _missing_optional_names:
        message = (
            "使用 'model_tools.optimization' 相关功能需要安装可选依赖，"
            "请运行 `pip install model-tools[ml]` 或手动安装 optuna 等依赖。"
        )
        raise ImportError(message) from _optional_error
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_version() -> str:
    """获取版本信息。"""
    return __version__


def get_module_info() -> Dict[str, Any]:
    """返回模块的结构化说明。"""
    modules = {
        'evaluation': {
            'description': '模型评估指标计算和分析',
            'submodules': {
                'metrics': 'AUC、KS、PSI 等评估指标',
                'reports': '自动化评估与监控报告',
                'visualization': '评估与监控可视化',
                'analysis': '模型深度诊断分析',
            },
        },
        'monitoring': {
            'description': '模型稳定性与特征漂移监控',
            'submodules': {
                'stability': '模型性能稳定性跟踪',
                'drift': '特征漂移检测',
                'alerting': '报警与通知系统',
                'workflow': '自动化监控流程',
                'performance': '性能与资源监控',
            },
        },
        'features': {
            'description': '特征工程与特征质量管理',
            'submodules': {
                'selection': '特征筛选与降维',
                'importance': '特征重要性评估',
                'encoding': '特征编码转换',
                'validation': '特征质量验证',
            },
        },
        'utils': {
            'description': '数据处理、配置与通用工具',
            'submodules': {
                'data_processing': '数据质量检查与清洗',
                'config': '配置读取与环境合并',
                'decorators': '常用性能/校验装饰器',
                'roi_calculator': '信贷 ROI 计算工具',
            },
        },
    }

    if 'optimization' not in _missing_optional_names:
        modules['optimization'] = {
            'description': '基于 Optuna 的智能超参数调优',
            'submodules': {
                'hyperparameter_tuner': '通用调优器及便捷创建方法',
                'parameter_spaces': '常见模型的参数空间定义',
                'objectives': '业务与模型指标优化目标',
            },
        }

    quick_start = {
        'data_quality': 'mt.check_data_quality(data, target_col="target")',
        'feature_selection': 'mt.FeatureSelector(method="iv").fit_transform(X, y)',
        'model_evaluation': 'mt.ModelEvaluator("model").evaluate_binary_classification(y, y_pred)',
        'model_monitoring': 'mt.ModelStabilityMonitor("model").monitor_performance(y, y_pred)',
    }

    if 'optimization' not in _missing_optional_names:
        quick_start['hyperparameter_tuning'] = 'mt.create_lgb_tuner(LGBMClassifier).optimize(X, y, n_trials=100)'
        quick_start['business_objective'] = 'mt.BusinessROIObjective(loan_amounts)(y_true, y_pred)'

    return {
        'name': 'model_tools',
        'version': __version__,
        'description': '专业的机器学习模型工具包',
        'author': __author__,
        'email': __email__,
        'modules': modules,
        'quick_start': quick_start,
    }


def print_welcome() -> None:
    """打印模块简介与快速上手提示。"""
    info = get_module_info()

    print('=' * 60)
    print(f"🚀 {info['name'].upper()} v{info['version']}")
    print(f"📊 {info['description']}")
    print('=' * 60)

    print('\n📦 可用模块:')
    for module_name, module_info in info['modules'].items():
        print(f"  • {module_name}: {module_info['description']}")
        for sub_name, sub_desc in module_info['submodules'].items():
            print(f"    - {sub_name}: {sub_desc}")

    print('\n🏃‍♂️ 快速开始:')
    for func_name, example in info['quick_start'].items():
        print(f"  • {func_name}: {example}")

    print('\n📚 更多帮助:')
    print('  • help(model_tools)')
    print('  • model_tools.get_module_info()')
    print('  • 文档: https://model-tools.readthedocs.io/')
    print('=' * 60)


# 初始化默认配置（若可用）
try:
    from .utils.config import global_config, DEFAULT_CONFIG

    if hasattr(global_config, 'config'):
        global_config.config.update(DEFAULT_CONFIG)
    else:
        global_config.config = DEFAULT_CONFIG.copy()
except Exception:
    pass
