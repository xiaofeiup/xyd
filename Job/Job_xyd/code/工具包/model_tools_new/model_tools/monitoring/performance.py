"""
性能监控模块

提供模型性能指标的实时监控功能
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging


class PerformanceTracker:
    """
    性能跟踪器

    跟踪模型训练和预测的性能指标
    """

    def __init__(self, model_name: str = "model"):
        """
        初始化性能跟踪器

        Parameters:
        -----------
        model_name : str, default="model"
            模型名称
        """
        self.model_name = model_name
        self.performance_log = []
        self.logger = logging.getLogger(f"PerformanceTracker_{model_name}")

    def track_training_time(self, start_time: float, end_time: float,
                           dataset_size: int, **kwargs) -> Dict:
        """
        跟踪训练时间

        Parameters:
        -----------
        start_time : float
            开始时间戳
        end_time : float
            结束时间戳
        dataset_size : int
            数据集大小
        **kwargs : dict
            其他参数

        Returns:
        --------
        timing_info : dict
            时间信息
        """
        training_time = end_time - start_time

        timing_info = {
            'timestamp': datetime.now().isoformat(),
            'type': 'training',
            'model_name': self.model_name,
            'training_time_seconds': training_time,
            'dataset_size': dataset_size,
            'time_per_sample': training_time / dataset_size if dataset_size > 0 else 0,
            **kwargs
        }

        self.performance_log.append(timing_info)

        self.logger.info(f"训练完成: {training_time:.2f}秒, 数据量: {dataset_size}")

        return timing_info

    def track_prediction_time(self, start_time: float, end_time: float,
                             batch_size: int, **kwargs) -> Dict:
        """
        跟踪预测时间

        Parameters:
        -----------
        start_time : float
            开始时间戳
        end_time : float
            结束时间戳
        batch_size : int
            批次大小
        **kwargs : dict
            其他参数

        Returns:
        --------
        timing_info : dict
            时间信息
        """
        prediction_time = end_time - start_time

        timing_info = {
            'timestamp': datetime.now().isoformat(),
            'type': 'prediction',
            'model_name': self.model_name,
            'prediction_time_seconds': prediction_time,
            'batch_size': batch_size,
            'throughput_samples_per_sec': batch_size / prediction_time if prediction_time > 0 else 0,
            **kwargs
        }

        self.performance_log.append(timing_info)

        return timing_info

    def track_memory_usage(self, memory_mb: float, operation: str = "unknown") -> Dict:
        """
        跟踪内存使用

        Parameters:
        -----------
        memory_mb : float
            内存使用量（MB）
        operation : str, default="unknown"
            操作类型

        Returns:
        --------
        memory_info : dict
            内存信息
        """
        memory_info = {
            'timestamp': datetime.now().isoformat(),
            'type': 'memory',
            'model_name': self.model_name,
            'operation': operation,
            'memory_mb': memory_mb
        }

        self.performance_log.append(memory_info)

        return memory_info

    def get_performance_summary(self, hours: int = 24) -> Dict:
        """
        获取性能总结

        Parameters:
        -----------
        hours : int, default=24
            总结时间范围（小时）

        Returns:
        --------
        summary : dict
            性能总结
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        recent_logs = [
            log for log in self.performance_log
            if log['timestamp'] >= cutoff_str
        ]

        summary = {
            'period_hours': hours,
            'total_operations': len(recent_logs),
            'training_operations': len([log for log in recent_logs if log['type'] == 'training']),
            'prediction_operations': len([log for log in recent_logs if log['type'] == 'prediction']),
            'memory_records': len([log for log in recent_logs if log['type'] == 'memory'])
        }

        # 计算训练性能统计
        training_logs = [log for log in recent_logs if log['type'] == 'training']
        if training_logs:
            training_times = [log['training_time_seconds'] for log in training_logs]
            summary['training_stats'] = {
                'avg_training_time': np.mean(training_times),
                'max_training_time': np.max(training_times),
                'min_training_time': np.min(training_times)
            }

        # 计算预测性能统计
        prediction_logs = [log for log in recent_logs if log['type'] == 'prediction']
        if prediction_logs:
            throughputs = [log['throughput_samples_per_sec'] for log in prediction_logs]
            summary['prediction_stats'] = {
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'min_throughput': np.min(throughputs)
            }

        # 内存使用统计
        memory_logs = [log for log in recent_logs if log['type'] == 'memory']
        if memory_logs:
            memory_usages = [log['memory_mb'] for log in memory_logs]
            summary['memory_stats'] = {
                'avg_memory_mb': np.mean(memory_usages),
                'max_memory_mb': np.max(memory_usages),
                'peak_memory_mb': np.max(memory_usages)
            }

        return summary

    def export_performance_log(self, filepath: str) -> None:
        """
        导出性能日志

        Parameters:
        -----------
        filepath : str
            导出文件路径
        """
        df = pd.DataFrame(self.performance_log)
        df.to_csv(filepath, index=False)
        self.logger.info(f"性能日志已导出到: {filepath}")


class ResourceMonitor:
    """
    资源监控器

    监控系统资源使用情况
    """

    def __init__(self):
        """初始化资源监控器"""
        self.monitoring = False
        self.resource_log = []

    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控

        Parameters:
        -----------
        interval : float, default=1.0
            监控间隔（秒）
        """
        import threading
        import psutil

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    # 获取CPU和内存使用率
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()

                    resource_info = {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_mb': memory.used / 1024 / 1024,
                        'memory_total_mb': memory.total / 1024 / 1024
                    }

                    self.resource_log.append(resource_info)

                    time.sleep(interval)

                except Exception as e:
                    logging.warning(f"资源监控出错: {e}")
                    break

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        logging.info("资源监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        logging.info("资源监控已停止")

    def get_resource_stats(self) -> Dict:
        """
        获取资源统计

        Returns:
        --------
        stats : dict
            资源统计信息
        """
        if not self.resource_log:
            return {'message': '暂无监控数据'}

        df = pd.DataFrame(self.resource_log)

        stats = {
            'monitoring_duration_minutes': len(df),  # 假设每分钟一个记录
            'cpu_stats': {
                'avg_cpu_percent': df['cpu_percent'].mean(),
                'max_cpu_percent': df['cpu_percent'].max(),
                'min_cpu_percent': df['cpu_percent'].min()
            },
            'memory_stats': {
                'avg_memory_percent': df['memory_percent'].mean(),
                'max_memory_percent': df['memory_percent'].max(),
                'peak_memory_used_mb': df['memory_used_mb'].max()
            }
        }

        return stats


def timing_decorator(tracker: PerformanceTracker):
    """
    性能计时装饰器

    Parameters:
    -----------
    tracker : PerformanceTracker
        性能跟踪器实例

    Returns:
    --------
    decorator : callable
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # 尝试获取数据大小信息
            data_size = 0
            if args and hasattr(args[0], '__len__'):
                data_size = len(args[0])

            if 'fit' in func.__name__ or 'train' in func.__name__:
                tracker.track_training_time(start_time, end_time, data_size)
            elif 'predict' in func.__name__:
                tracker.track_prediction_time(start_time, end_time, data_size)

            return result
        return wrapper
    return decorator