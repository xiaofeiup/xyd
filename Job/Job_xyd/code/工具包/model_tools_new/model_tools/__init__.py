"""
Model Tools 2.0 - 专业的机器学习模型工具包

提供特征工程、模型评估、监控和报告等全套工具

主要功能:
- 特征选择和重要性分析
- 模型评估指标 (AUC, KS, PSI, Lift等)
- 模型稳定性监控和漂移检测
- 自动化报警系统
- 可视化和报告生成
- 实用装饰器和工具

快速开始:
    import model_tools as mt

    # 数据质量检查
    quality_report = mt.check_data_quality(data, target_col='target')

    # 特征选择
    selector = mt.FeatureSelector(method='iv')
    X_selected = selector.fit_transform(X, y)

    # 模型评估
    auc = mt.calculate_auc(y_true, y_scores)
    ks, _, _ = mt.calculate_ks(y_true, y_scores)

    # 模型监控
    monitor = mt.ModelStabilityMonitor('my_model')
    monitor.set_baseline(y_train, y_pred_train)
    result = monitor.monitor_performance(y_test, y_pred_test)
"""

__version__ = "2.0.0"
__author__ = "Model Tools Team"
__email__ = "model-tools@example.com"

# 核心模块导入
from . import evaluation
from . import monitoring
from . import features
from . import utils

# 评估模块的快捷导入
from .evaluation.metrics import (
    calculate_auc,
    calculate_ks,
    calculate_psi,
    ModelEvaluator
)

from .evaluation.reports import (
    ModelReportGenerator,
    generate_model_report
)

from .evaluation.visualization import (
    ModelVisualizationTool,
    quick_model_visualization
)

from .evaluation.analysis import (
    ModelAnalyzer,
    analyze_model_comprehensive
)

# 监控模块的快捷导入
from .monitoring.stability import ModelStabilityMonitor
from .monitoring.drift import FeatureDriftDetector
from .monitoring.alerting import AlertManager, AlertEngine
from .monitoring.workflow import MonitoringWorkflow
from .monitoring.performance import PerformanceTracker, ResourceMonitor

# 特征模块的快捷导入
from .features.selection import (
    FeatureSelector,
    filter_features_by_single_value_ratio
)
from .features.importance import FeatureImportanceAnalyzer
from .features.encoding import WOEEncoder, TargetEncoder
from .features.validation import (
    FeatureValidator,
    validate_dataset,
    compare_datasets
)

# 工具模块的快捷导入
from .utils.data_processing import check_data_quality
from .utils.config import (
    ConfigManager,
    get_config,
    set_config
)

# 常用装饰器
from .utils.decorators import (
    timer,
    retry,
    validate_inputs,
    cache_result,
    log_calls,
    deprecated,
    monitor_performance,
    rate_limit,
    ml_function
)

__all__ = [
    # 版本和元信息
    '__version__',
    '__author__',
    '__email__',

    # 子模块
    'evaluation',
    'monitoring',
    'features',
    'utils',

    # 评估功能
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

    # 监控功能
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

    # 工具函数
    'check_data_quality',
    'ConfigManager',
    'get_config',
    'set_config',

    # 装饰器
    'timer',
    'retry',
    'validate_inputs',
    'cache_result',
    'log_calls',
    'deprecated',
    'monitor_performance',
    'rate_limit',
    'ml_function',

    # 元函数
    'get_version',
    'get_module_info',
    'print_welcome'
]

# 模块信息和帮助函数
def get_version():
    """获取版本信息"""
    return __version__

def get_module_info():
    """获取模块详细信息"""
    return {
        'name': 'model_tools',
        'version': __version__,
        'description': '专业的机器学习模型工具包',
        'author': __author__,
        'email': __email__,
        'modules': {
            'evaluation': {
                'description': '模型评估指标计算和分析',
                'submodules': {
                    'metrics': 'AUC, KS, PSI等评估指标',
                    'reports': '自动化报告生成',
                    'visualization': '可视化图表',
                    'analysis': '深度模型分析'
                }
            },
            'monitoring': {
                'description': '模型稳定性监控和报警',
                'submodules': {
                    'stability': '模型性能稳定性监控',
                    'drift': '特征漂移检测',
                    'alerting': '自动化报警系统',
                    'workflow': '监控工作流程',
                    'performance': '性能和资源监控'
                }
            },
            'features': {
                'description': '特征工程和特征选择',
                'submodules': {
                    'selection': '特征选择算法',
                    'importance': '特征重要性分析',
                    'encoding': '特征编码转换',
                    'validation': '特征质量验证'
                }
            },
            'utils': {
                'description': '数据处理和配置管理工具',
                'submodules': {
                    'data_processing': '数据质量检查',
                    'config': '配置管理',
                    'decorators': '实用装饰器'
                }
            }
        },
        'quick_start': {
            'data_quality': 'mt.check_data_quality(data, target_col="target")',
            'feature_selection': 'mt.FeatureSelector(method="iv").fit_transform(X, y)',
            'model_evaluation': 'mt.calculate_auc(y_true, y_scores)',
            'model_monitoring': 'mt.ModelStabilityMonitor("model").monitor_performance(y, y_pred)'
        }
    }

def print_welcome():
    """打印欢迎信息和使用指南"""
    info = get_module_info()
    print("=" * 60)
    print(f"🚀 {info['name'].upper()} v{info['version']}")
    print(f"📊 {info['description']}")
    print("=" * 60)

    print("\n📦 可用模块:")
    for module_name, module_info in info['modules'].items():
        print(f"  • {module_name}: {module_info['description']}")
        for sub_name, sub_desc in module_info['submodules'].items():
            print(f"    - {sub_name}: {sub_desc}")

    print("\n🏃‍♂️ 快速开始:")
    for func_name, example in info['quick_start'].items():
        print(f"  • {func_name}: {example}")

    print(f"\n📚 获取帮助:")
    print(f"  • help(model_tools)")
    print(f"  • model_tools.get_module_info()")
    print(f"  • 文档: https://model-tools.readthedocs.io/")
    print("=" * 60)

# 初始化默认配置
try:
    from .utils.config import global_config, DEFAULT_CONFIG
    if hasattr(global_config, 'config'):
        global_config.config.update(DEFAULT_CONFIG)
    else:
        global_config.config = DEFAULT_CONFIG.copy()
except ImportError:
    # 如果配置模块导入失败，忽略错误
    pass