"""
配置管理模块

提供配置文件读取、环境变量管理等功能
"""

import json
import yaml
import os
from typing import Dict, Any, Optional
import warnings


class ConfigManager:
    """
    配置管理器

    支持JSON、YAML等格式的配置文件
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Parameters:
        -----------
        config_path : str, optional
            配置文件路径
        """
        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        Parameters:
        -----------
        config_path : str
            配置文件路径

        Returns:
        --------
        config : dict
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        file_ext = os.path.splitext(config_path)[1].lower()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext == '.json':
                    self.config = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")

            self.config_path = config_path
            return self.config

        except Exception as e:
            raise ValueError(f"加载配置文件失败: {str(e)}")

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        保存配置文件

        Parameters:
        -----------
        config_path : str, optional
            保存路径，如果为None则使用当前路径
        """
        save_path = config_path or self.config_path

        if not save_path:
            raise ValueError("未指定保存路径")

        file_ext = os.path.splitext(save_path)[1].lower()

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if file_ext == '.json':
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")

        except Exception as e:
            raise ValueError(f"保存配置文件失败: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Parameters:
        -----------
        key : str
            配置键，支持点号分隔的嵌套键
        default : any
            默认值

        Returns:
        --------
        value : any
            配置值
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值

        Parameters:
        -----------
        key : str
            配置键，支持点号分隔的嵌套键
        value : any
            配置值
        """
        keys = key.split('.')
        config = self.config

        # 创建嵌套结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        批量更新配置

        Parameters:
        -----------
        updates : dict
            更新的配置字典
        """
        for key, value in updates.items():
            self.set(key, value)

    def get_env_var(self, var_name: str, default: Any = None) -> Any:
        """
        获取环境变量

        Parameters:
        -----------
        var_name : str
            环境变量名
        default : any
            默认值

        Returns:
        --------
        value : any
            环境变量值
        """
        return os.environ.get(var_name, default)

    def set_env_var(self, var_name: str, value: str) -> None:
        """
        设置环境变量

        Parameters:
        -----------
        var_name : str
            环境变量名
        value : str
            环境变量值
        """
        os.environ[var_name] = str(value)

    def get_merged_config(self, env_prefix: str = 'MODEL_TOOLS_') -> Dict[str, Any]:
        """
        获取合并了环境变量的配置

        Parameters:
        -----------
        env_prefix : str, default='MODEL_TOOLS_'
            环境变量前缀

        Returns:
        --------
        merged_config : dict
            合并后的配置
        """
        merged_config = self.config.copy()

        # 遍历环境变量
        for var_name, var_value in os.environ.items():
            if var_name.startswith(env_prefix):
                # 移除前缀并转换为配置键
                config_key = var_name[len(env_prefix):].lower().replace('_', '.')

                # 尝试转换数据类型
                try:
                    # 尝试解析为数字
                    if '.' in var_value:
                        parsed_value = float(var_value)
                    else:
                        parsed_value = int(var_value)
                except ValueError:
                    # 尝试解析为布尔值
                    if var_value.lower() in ['true', 'false']:
                        parsed_value = var_value.lower() == 'true'
                    else:
                        parsed_value = var_value

                self.set_config_value(merged_config, config_key, parsed_value)

        return merged_config

    @staticmethod
    def set_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
        """
        在配置字典中设置嵌套值

        Parameters:
        -----------
        config : dict
            配置字典
        key : str
            配置键
        value : any
            配置值
        """
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value


# 默认配置
DEFAULT_CONFIG = {
    'model_evaluation': {
        'metrics': {
            'auc_threshold': 0.7,
            'ks_threshold': 0.3,
            'psi_threshold': 0.25
        },
        'cross_validation': {
            'n_folds': 5,
            'random_state': 42
        }
    },
    'feature_selection': {
        'iv_threshold': 0.1,
        'single_value_threshold': 0.95,
        'correlation_threshold': 0.8,
        'max_features': 50
    },
    'monitoring': {
        'drift_detection': {
            'psi_threshold': 0.25,
            'bins': 10,
            'min_sample': 10
        },
        'performance_monitoring': {
            'auc_drop_threshold': 0.05,
            'ks_drop_threshold': 0.1
        }
    },
    'alerting': {
        'enable_email': False,
        'enable_log': True,
        'enable_webhook': False,
        'log_level': 'WARNING'
    },
    'data_processing': {
        'missing_value_strategy': 'median',
        'outlier_detection_method': 'iqr',
        'outlier_threshold': 1.5
    }
}


def create_default_config(config_path: str) -> None:
    """
    创建默认配置文件

    Parameters:
    -----------
    config_path : str
        配置文件路径
    """
    config_manager = ConfigManager()
    config_manager.config = DEFAULT_CONFIG
    config_manager.save_config(config_path)
    print(f"默认配置文件已创建: {config_path}")


def load_config_with_env(config_path: Optional[str] = None,
                        env_prefix: str = 'MODEL_TOOLS_') -> Dict[str, Any]:
    """
    加载配置并合并环境变量

    Parameters:
    -----------
    config_path : str, optional
        配置文件路径
    env_prefix : str, default='MODEL_TOOLS_'
        环境变量前缀

    Returns:
    --------
    config : dict
        合并后的配置
    """
    config_manager = ConfigManager(config_path)

    if not config_manager.config:
        # 如果没有配置文件，使用默认配置
        config_manager.config = DEFAULT_CONFIG
        warnings.warn("未找到配置文件，使用默认配置")

    return config_manager.get_merged_config(env_prefix)


# 全局配置实例
global_config = ConfigManager()
global_config.config = DEFAULT_CONFIG


def get_config(key: str, default: Any = None) -> Any:
    """
    获取全局配置值

    Parameters:
    -----------
    key : str
        配置键
    default : any
        默认值

    Returns:
    --------
    value : any
        配置值
    """
    return global_config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    设置全局配置值

    Parameters:
    -----------
    key : str
        配置键
    value : any
        配置值
    """
    global_config.set(key, value)


def load_global_config(config_path: str) -> None:
    """
    加载全局配置文件

    Parameters:
    -----------
    config_path : str
        配置文件路径
    """
    global_config.load_config(config_path)