"""
工具模块

提供数据处理、配置管理等通用工具功能
"""

from .data_processing import (
    check_data_quality,
    clean_column_names,
    handle_missing_values,
    detect_outliers,
    remove_outliers,
    split_features_target,
    sample_data,
    save_data_report
)

from .config import (
    ConfigManager,
    DEFAULT_CONFIG,
    create_default_config,
    load_config_with_env,
    get_config,
    set_config,
    load_global_config
)

from .roi_calculator import (
    CreditMetrics,
    CreditROICalculator,
    batch_roi_analysis
)

from .decorators import (
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

from .memory_tools import (
    whos
)

from .model_scores_db import (
    init_score_db,
    ensure_indexes,
    write_model_scores,
    import_csv_to_db,
    read_model_scores,
    SCORE_TABLE_SCHEMA,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_NAME
)

__all__ = [
    # 数据处理
    'check_data_quality',
    'clean_column_names',
    'handle_missing_values',
    'detect_outliers',
    'remove_outliers',
    'split_features_target',
    'sample_data',
    'save_data_report',

    # 配置管理
    'ConfigManager',
    'DEFAULT_CONFIG',
    'create_default_config',
    'load_config_with_env',
    'get_config',
    'set_config',
    'load_global_config',

    # ROI计算
    'CreditMetrics',
    'CreditROICalculator',
    'batch_roi_analysis',

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

    # 内存分析
    'whos',

    # 模型分数数据库
    'init_score_db',
    'ensure_indexes',
    'write_model_scores',
    'import_csv_to_db',
    'read_model_scores',
    'SCORE_TABLE_SCHEMA',
    'DEFAULT_DB_PATH',
    'DEFAULT_TABLE_NAME'
]