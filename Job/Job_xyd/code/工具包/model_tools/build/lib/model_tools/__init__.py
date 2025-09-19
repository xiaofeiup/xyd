# model_tools 工具包初始化 
"""
model_tools: 机器学习模型工具包

包含以下子模块:
- metric_report: 指标报告生成
- feature_select: 特征选择工具
- plot: 数据可视化工具
- gentools: 通用评估工具
- model: 模型相关工具
"""

# 导入各个子模块的主要功能
from .metric_report import (
    generate_metric_table,
    numerical_univerate,
    analyze_lift_performance
)

from .feature_select import (
    by_missing_nunique_iv,
)

from .plot import (
    plot_ks_curve,
    plot_detailed_ks_analysis,
    plot_lift_gain,
    plot_lift_curve,
    plot_combined_ks_lift_analysis,
    metric_report_plot
)

from .gentools import (
    calc_auc,
    calculate_ks,
    calculate_lift,
    calc_gain
)

# 定义包的版本
__version__ = "0.1.0"

# 定义公共API
__all__ = [
    # metric_report
    'generate_metric_table',
    'numerical_univerate', 
    'analyze_lift_performance',
    
    # feature_select
    'by_missing_nunique_iv',
    
    # plot
    'plot_ks_curve',
    'plot_detailed_ks_analysis',
    'plot_lift_gain',
    'plot_lift_curve',
    'plot_combined_ks_lift_analysis',
    'metric_report_plot',
    
    # gentools
    'calc_auc',
    'calculate_ks',
    'calculate_lift',
    'calc_gain'
] 