"""
装饰器工具模块

提供各种实用的装饰器功能
"""

import time
import functools
import logging
import warnings
from typing import Callable, Any, Optional, Dict, Union
from datetime import datetime
import pandas as pd
import numpy as np


def timer(func: Callable = None, *,
          logger: Optional[logging.Logger] = None,
          message: str = None) -> Callable:
    """
    计时装饰器

    Parameters:
    -----------
    func : callable, optional
        被装饰的函数
    logger : logging.Logger, optional
        日志记录器
    message : str, optional
        自定义消息

    Returns:
    --------
    wrapper : callable
        装饰后的函数
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                log_message = message or f"{f.__name__} 执行时间: {execution_time:.4f}秒"

                if logger:
                    logger.info(log_message)
                else:
                    print(log_message)

                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                error_message = f"{f.__name__} 执行失败 (耗时: {execution_time:.4f}秒): {e}"

                if logger:
                    logger.error(error_message)
                else:
                    print(error_message)

                raise

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def retry(max_attempts: int = 3,
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: tuple = (Exception,),
          logger: Optional[logging.Logger] = None) -> Callable:
    """
    重试装饰器

    Parameters:
    -----------
    max_attempts : int, default=3
        最大重试次数
    delay : float, default=1.0
        初始延迟时间（秒）
    backoff : float, default=2.0
        延迟递增倍数
    exceptions : tuple, default=(Exception,)
        需要重试的异常类型
    logger : logging.Logger, optional
        日志记录器

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        log_message = f"{func.__name__} 第{attempt + 1}次尝试失败: {e}, {current_delay:.2f}秒后重试"
                        if logger:
                            logger.warning(log_message)
                        else:
                            print(log_message)

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        log_message = f"{func.__name__} 所有{max_attempts}次尝试均失败"
                        if logger:
                            logger.error(log_message)
                        else:
                            print(log_message)

            raise last_exception

        return wrapper

    return decorator


def validate_inputs(**validators) -> Callable:
    """
    输入验证装饰器

    Parameters:
    -----------
    **validators : dict
        参数验证器字典，键为参数名，值为验证函数

    Returns:
    --------
    decorator : callable
        装饰器函数

    Examples:
    ---------
    @validate_inputs(
        data=lambda x: isinstance(x, pd.DataFrame),
        threshold=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1
    )
    def process_data(data, threshold):
        return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 验证每个参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"参数 '{param_name}' 验证失败: {value}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def cache_result(ttl: Optional[float] = None,
                max_size: Optional[int] = 128,
                key_func: Optional[Callable] = None) -> Callable:
    """
    结果缓存装饰器

    Parameters:
    -----------
    ttl : float, optional
        缓存生存时间（秒），None表示永久缓存
    max_size : int, optional
        最大缓存条目数
    key_func : callable, optional
        自定义键生成函数

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        access_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = str(args) + str(sorted(kwargs.items()))

            current_time = time.time()

            # 检查缓存是否过期
            if cache_key in cache:
                if ttl is None or (current_time - access_times[cache_key]) < ttl:
                    return cache[cache_key]
                else:
                    # 缓存过期，删除
                    del cache[cache_key]
                    del access_times[cache_key]

            # 检查缓存大小限制
            if max_size and len(cache) >= max_size:
                # 删除最旧的条目
                oldest_key = min(access_times.keys(), key=access_times.get)
                del cache[oldest_key]
                del access_times[oldest_key]

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[cache_key] = result
            access_times[cache_key] = current_time

            return result

        # 添加缓存管理方法
        wrapper.cache_clear = lambda: cache.clear() or access_times.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'max_size': max_size,
            'ttl': ttl
        }

        return wrapper

    return decorator


def log_calls(logger: Optional[logging.Logger] = None,
              level: int = logging.INFO,
              include_args: bool = True,
              include_result: bool = False) -> Callable:
    """
    函数调用日志装饰器

    Parameters:
    -----------
    logger : logging.Logger, optional
        日志记录器
    level : int, default=logging.INFO
        日志级别
    include_args : bool, default=True
        是否记录参数
    include_result : bool, default=False
        是否记录返回值

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)

            # 准备参数信息
            if include_args:
                args_str = f"args={args[:3]}{'...' if len(args) > 3 else ''}, kwargs={list(kwargs.keys())}"
            else:
                args_str = ""

            # 记录函数调用
            func_logger.log(level, f"调用 {func.__name__}({args_str})")

            try:
                result = func(*args, **kwargs)

                # 记录返回值（如果需要）
                if include_result:
                    result_str = str(result)[:100] + '...' if len(str(result)) > 100 else str(result)
                    func_logger.log(level, f"{func.__name__} 返回: {result_str}")
                else:
                    func_logger.log(level, f"{func.__name__} 执行成功")

                return result

            except Exception as e:
                func_logger.error(f"{func.__name__} 执行失败: {e}")
                raise

        return wrapper

    return decorator


def deprecated(reason: str = None,
               alternative: str = None,
               removal_version: str = None) -> Callable:
    """
    废弃警告装饰器

    Parameters:
    -----------
    reason : str, optional
        废弃原因
    alternative : str, optional
        替代方案
    removal_version : str, optional
        移除版本

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"函数 {func.__name__} 已废弃"

            if reason:
                warning_msg += f": {reason}"

            if alternative:
                warning_msg += f". 请使用 {alternative} 替代"

            if removal_version:
                warning_msg += f". 将在版本 {removal_version} 中移除"

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_exceptions(default_return: Any = None,
                     exceptions: tuple = (Exception,),
                     logger: Optional[logging.Logger] = None,
                     reraise: bool = False) -> Callable:
    """
    异常处理装饰器

    Parameters:
    -----------
    default_return : any, optional
        异常时的默认返回值
    exceptions : tuple, default=(Exception,)
        要处理的异常类型
    logger : logging.Logger, optional
        日志记录器
    reraise : bool, default=False
        是否重新抛出异常

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                error_msg = f"{func.__name__} 发生异常: {e}"

                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)

                if reraise:
                    raise
                else:
                    return default_return

        return wrapper

    return decorator


def check_data_types(**type_checks) -> Callable:
    """
    数据类型检查装饰器

    Parameters:
    -----------
    **type_checks : dict
        参数类型检查字典

    Returns:
    --------
    decorator : callable
        装饰器函数

    Examples:
    ---------
    @check_data_types(
        data=pd.DataFrame,
        target=(pd.Series, np.ndarray),
        threshold=(int, float)
    )
    def process_data(data, target, threshold):
        return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"参数 '{param_name}' 期望类型 {expected_type}, "
                            f"但得到 {type(value)}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def monitor_performance(include_memory: bool = False,
                       logger: Optional[logging.Logger] = None) -> Callable:
    """
    性能监控装饰器

    Parameters:
    -----------
    include_memory : bool, default=False
        是否监控内存使用
    logger : logging.Logger, optional
        日志记录器

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = None

            if include_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time

                perf_info = f"{func.__name__} 性能信息: 执行时间={execution_time:.4f}秒"

                if include_memory and start_memory is not None:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024
                        memory_usage = end_memory - start_memory
                        perf_info += f", 内存变化={memory_usage:.2f}MB"
                    except:
                        pass

                if logger:
                    logger.info(perf_info)
                else:
                    print(perf_info)

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                error_msg = f"{func.__name__} 执行失败 (耗时: {execution_time:.4f}秒): {e}"

                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)

                raise

        return wrapper

    return decorator


def rate_limit(max_calls: int,
               period: float = 60.0,
               per_instance: bool = False) -> Callable:
    """
    速率限制装饰器

    Parameters:
    -----------
    max_calls : int
        时间窗口内最大调用次数
    period : float, default=60.0
        时间窗口（秒）
    per_instance : bool, default=False
        是否按实例分别限制

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        call_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # 确定限制键
            if per_instance and args:
                limit_key = id(args[0])  # 使用第一个参数（通常是self）的id
            else:
                limit_key = 'global'

            # 初始化调用时间列表
            if limit_key not in call_times:
                call_times[limit_key] = []

            # 清理过期的调用时间
            call_times[limit_key] = [
                call_time for call_time in call_times[limit_key]
                if current_time - call_time < period
            ]

            # 检查是否超过限制
            if len(call_times[limit_key]) >= max_calls:
                oldest_call = min(call_times[limit_key])
                wait_time = period - (current_time - oldest_call)
                raise RuntimeError(
                    f"速率限制: {period}秒内最多调用{max_calls}次, "
                    f"请等待{wait_time:.2f}秒"
                )

            # 记录调用时间
            call_times[limit_key].append(current_time)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def ensure_dataframe(convert_cols: bool = True,
                    required_cols: Optional[list] = None) -> Callable:
    """
    确保输入为DataFrame的装饰器

    Parameters:
    -----------
    convert_cols : bool, default=True
        是否自动转换列名为字符串
    required_cols : list, optional
        必需的列名列表

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查第一个参数是否为DataFrame
            if args and not isinstance(args[0], pd.DataFrame):
                if hasattr(args[0], '__array__') or isinstance(args[0], (list, tuple)):
                    # 尝试转换为DataFrame
                    args = (pd.DataFrame(args[0]),) + args[1:]
                else:
                    raise TypeError(f"第一个参数必须是DataFrame或可转换的数组类型")

            df = args[0]

            # 转换列名
            if convert_cols:
                df.columns = df.columns.astype(str)
                args = (df,) + args[1:]

            # 检查必需列
            if required_cols:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"缺少必需的列: {missing_cols}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def auto_plot_save(save_dir: str = "./plots",
                  filename_prefix: str = None,
                  format: str = "png",
                  dpi: int = 300) -> Callable:
    """
    自动保存图表的装饰器

    Parameters:
    -----------
    save_dir : str, default="./plots"
        保存目录
    filename_prefix : str, optional
        文件名前缀
    format : str, default="png"
        图片格式
    dpi : int, default=300
        图片分辨率

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            try:
                import matplotlib.pyplot as plt
                import os

                # 创建保存目录
                os.makedirs(save_dir, exist_ok=True)

                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = filename_prefix or func.__name__
                filename = f"{prefix}_{timestamp}.{format}"
                filepath = os.path.join(save_dir, filename)

                # 保存图表
                plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
                print(f"图表已保存到: {filepath}")

            except Exception as e:
                print(f"保存图表失败: {e}")

            return result

        return wrapper

    return decorator


# 组合装饰器示例
def ml_function(include_timing: bool = True,
               include_logging: bool = True,
               include_validation: bool = True) -> Callable:
    """
    机器学习函数的组合装饰器

    Parameters:
    -----------
    include_timing : bool, default=True
        是否包含计时
    include_logging : bool, default=True
        是否包含日志
    include_validation : bool, default=True
        是否包含数据验证

    Returns:
    --------
    decorator : callable
        组合装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        # 应用装饰器
        wrapped_func = func

        if include_validation:
            wrapped_func = check_data_types(
                data=pd.DataFrame,
                target=(pd.Series, np.ndarray, type(None))
            )(wrapped_func)

        if include_logging:
            wrapped_func = log_calls(
                level=logging.INFO,
                include_args=True,
                include_result=False
            )(wrapped_func)

        if include_timing:
            wrapped_func = timer(wrapped_func)

        return wrapped_func

    return decorator


# 使用示例
if __name__ == "__main__":
    # 示例：组合使用多个装饰器
    @timer
    @retry(max_attempts=3)
    @validate_inputs(
        x=lambda x: isinstance(x, (int, float)),
        y=lambda y: isinstance(y, (int, float))
    )
    def example_function(x, y):
        """示例函数"""
        if x < 0 or y < 0:
            raise ValueError("参数必须为正数")
        return x * y

    # 示例：机器学习函数装饰器
    @ml_function(include_timing=True, include_logging=True, include_validation=True)
    def train_model(data: pd.DataFrame, target: pd.Series = None):
        """训练模型示例"""
        time.sleep(0.1)  # 模拟训练时间
        return {"accuracy": 0.95, "model": "trained_model"}