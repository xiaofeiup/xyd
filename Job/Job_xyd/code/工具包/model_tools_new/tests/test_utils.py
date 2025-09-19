"""
测试utils模块
"""

import pytest
import pandas as pd
import numpy as np
import time
import logging
from unittest.mock import patch

from model_tools.utils.data_processing import check_data_quality
from model_tools.utils.config import ConfigManager, get_config, set_config
from model_tools.utils.decorators import (
    timer, retry, validate_inputs, cache_result,
    log_calls, deprecated, handle_exceptions,
    check_data_types, monitor_performance, rate_limit
)


class TestDataProcessing:
    """测试数据处理工具"""

    def test_check_data_quality_basic(self, sample_data):
        """测试基础数据质量检查"""
        result = check_data_quality(sample_data, target_col='target')

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'basic_info' in result
        assert 'missing_analysis' in result
        assert 'target_analysis' in result
        assert 'data_types' in result

        # 验证基础信息
        basic_info = result['basic_info']
        assert basic_info['total_rows'] == len(sample_data)
        assert basic_info['total_columns'] == len(sample_data.columns)

    def test_check_data_quality_without_target(self, sample_data):
        """测试无目标变量的数据质量检查"""
        data_without_target = sample_data.drop('target', axis=1)
        result = check_data_quality(data_without_target)

        assert isinstance(result, dict)
        assert 'basic_info' in result
        assert 'target_analysis' not in result

    def test_check_data_quality_with_issues(self):
        """测试有问题的数据质量检查"""
        # 创建有问题的数据
        problem_data = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'missing_feature': [1, np.nan, np.nan, np.nan, 5],
            'constant_feature': [1, 1, 1, 1, 1],
            'target': [0, 1, 0, 1, 0]
        })

        result = check_data_quality(problem_data, target_col='target')

        # 应该识别出数据质量问题
        missing_analysis = result['missing_analysis']
        assert missing_analysis['columns_with_missing'] > 0


class TestConfigManager:
    """测试配置管理"""

    def test_config_manager_init(self):
        """测试ConfigManager初始化"""
        config_manager = ConfigManager()
        assert hasattr(config_manager, 'config')

    def test_set_and_get_config(self):
        """测试设置和获取配置"""
        config_manager = ConfigManager()

        # 设置配置
        config_manager.set('test.key', 'test_value')

        # 获取配置
        value = config_manager.get('test.key')
        assert value == 'test_value'

    def test_get_config_with_default(self):
        """测试获取不存在配置的默认值"""
        config_manager = ConfigManager()

        # 获取不存在的配置，应该返回默认值
        value = config_manager.get('nonexistent.key', 'default_value')
        assert value == 'default_value'

    def test_global_config_functions(self):
        """测试全局配置函数"""
        # 设置配置
        set_config('global.test.key', 'global_value')

        # 获取配置
        value = get_config('global.test.key')
        assert value == 'global_value'

        # 获取不存在的配置
        default_value = get_config('nonexistent.key', 'default')
        assert default_value == 'default'


class TestDecorators:
    """测试装饰器功能"""

    def test_timer_decorator(self):
        """测试计时装饰器"""
        @timer
        def slow_function():
            time.sleep(0.1)
            return "done"

        with patch('builtins.print') as mock_print:
            result = slow_function()
            assert result == "done"
            mock_print.assert_called()  # 应该有打印输出

    def test_retry_decorator(self):
        """测试重试装饰器"""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Still failing")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    def test_retry_decorator_all_failures(self):
        """测试重试装饰器全部失败情况"""
        @retry(max_attempts=2, delay=0.01)
        def always_failing_function():
            raise ValueError("Always failing")

        with pytest.raises(ValueError):
            always_failing_function()

    def test_validate_inputs_decorator(self):
        """测试输入验证装饰器"""
        @validate_inputs(
            x=lambda x: isinstance(x, int) and x > 0,
            y=lambda y: isinstance(y, str)
        )
        def test_function(x, y):
            return f"{x}_{y}"

        # 正常情况
        result = test_function(5, "hello")
        assert result == "5_hello"

        # 验证失败情况
        with pytest.raises(ValueError):
            test_function(-1, "hello")  # x不满足条件

        with pytest.raises(ValueError):
            test_function(5, 123)  # y不满足条件

    def test_cache_result_decorator(self):
        """测试结果缓存装饰器"""
        call_count = 0

        @cache_result(ttl=1.0, max_size=5)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # 第一次调用
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # 第二次调用相同参数（应该使用缓存）
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # 没有增加

        # 调用不同参数
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_log_calls_decorator(self):
        """测试函数调用日志装饰器"""
        logger = logging.getLogger("test_logger")

        @log_calls(logger=logger, include_args=True)
        def test_function(x, y=None):
            return x + (y or 0)

        with patch.object(logger, 'log') as mock_log:
            result = test_function(5, y=3)
            assert result == 8
            assert mock_log.call_count >= 2  # 至少有调用开始和结束的日志

    def test_deprecated_decorator(self):
        """测试废弃警告装饰器"""
        @deprecated(reason="This function is old", alternative="new_function")
        def old_function():
            return "old result"

        with pytest.warns(DeprecationWarning):
            result = old_function()
            assert result == "old result"

    def test_handle_exceptions_decorator(self):
        """测试异常处理装饰器"""
        @handle_exceptions(default_return="error_handled", reraise=False)
        def error_function():
            raise ValueError("Something went wrong")

        result = error_function()
        assert result == "error_handled"

        # 测试重新抛出异常
        @handle_exceptions(default_return=None, reraise=True)
        def error_function_reraise():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            error_function_reraise()

    def test_check_data_types_decorator(self):
        """测试数据类型检查装饰器"""
        @check_data_types(
            data=pd.DataFrame,
            value=(int, float)
        )
        def process_data(data, value):
            return len(data) * value

        # 正常情况
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = process_data(df, 2.5)
        assert result == 7.5

        # 类型错误
        with pytest.raises(TypeError):
            process_data([1, 2, 3], 2.5)  # data不是DataFrame

        with pytest.raises(TypeError):
            process_data(df, "not_number")  # value不是数字

    def test_monitor_performance_decorator(self):
        """测试性能监控装饰器"""
        @monitor_performance(include_memory=False)
        def test_function():
            time.sleep(0.05)
            return "done"

        with patch('builtins.print') as mock_print:
            result = test_function()
            assert result == "done"
            mock_print.assert_called()  # 应该有性能信息输出

    def test_rate_limit_decorator(self):
        """测试速率限制装饰器"""
        @rate_limit(max_calls=2, period=1.0)
        def limited_function():
            return "called"

        # 前两次调用应该成功
        result1 = limited_function()
        result2 = limited_function()
        assert result1 == "called"
        assert result2 == "called"

        # 第三次调用应该被限制
        with pytest.raises(RuntimeError):
            limited_function()


class TestDecoratorCombinations:
    """测试装饰器组合使用"""

    def test_multiple_decorators(self):
        """测试多个装饰器组合"""
        call_count = 0

        @timer
        @retry(max_attempts=2, delay=0.01)
        @validate_inputs(x=lambda x: isinstance(x, int))
        def complex_function(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return x * 2

        with patch('builtins.print'):  # 抑制timer输出
            result = complex_function(5)
            assert result == 10
            assert call_count == 2

    def test_decorator_with_dataframe(self):
        """测试DataFrame相关装饰器组合"""
        @timer
        @check_data_types(data=pd.DataFrame)
        def process_dataframe(data):
            return len(data)

        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        with patch('builtins.print'):  # 抑制timer输出
            result = process_dataframe(df)
            assert result == 3


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe_quality_check(self):
        """测试空DataFrame的质量检查"""
        empty_df = pd.DataFrame()
        result = check_data_quality(empty_df)

        assert result['basic_info']['total_rows'] == 0
        assert result['basic_info']['total_columns'] == 0

    def test_single_row_dataframe_quality_check(self):
        """测试单行DataFrame的质量检查"""
        single_row_df = pd.DataFrame({'col1': [1], 'col2': [2]})
        result = check_data_quality(single_row_df)

        assert result['basic_info']['total_rows'] == 1
        assert result['basic_info']['total_columns'] == 2

    def test_config_with_none_values(self):
        """测试配置中的None值"""
        config_manager = ConfigManager()

        # 设置None值
        config_manager.set('test.none', None)

        # 获取None值
        value = config_manager.get('test.none', 'default')
        assert value is None

    def test_decorator_with_no_arguments(self):
        """测试无参数函数的装饰器"""
        @timer
        def no_arg_function():
            return "no args"

        with patch('builtins.print'):
            result = no_arg_function()
            assert result == "no args"

    def test_cache_decorator_with_complex_args(self):
        """测试缓存装饰器处理复杂参数"""
        call_count = 0

        @cache_result(max_size=10)
        def function_with_complex_args(data_dict, data_list):
            nonlocal call_count
            call_count += 1
            return sum(data_dict.values()) + sum(data_list)

        # 第一次调用
        result1 = function_with_complex_args({'a': 1, 'b': 2}, [3, 4])
        assert result1 == 10
        assert call_count == 1

        # 相同参数再次调用（应该使用缓存）
        result2 = function_with_complex_args({'a': 1, 'b': 2}, [3, 4])
        assert result2 == 10
        assert call_count == 1  # 没有增加