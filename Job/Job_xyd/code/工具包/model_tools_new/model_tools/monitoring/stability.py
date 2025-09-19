"""
模型稳定性监控模块

提供模型性能稳定性监控功能，包括AUC、KS等关键指标的跟踪
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ModelStabilityMonitor:
    """
    模型稳定性监控器

    监控模型性能指标的变化，当指标出现显著波动时触发报警
    """

    def __init__(self,
                 model_name: str,
                 baseline_metrics: Optional[Dict] = None,
                 thresholds: Optional[Dict] = None):
        """
        初始化监控器

        Parameters:
        -----------
        model_name : str
            模型名称
        baseline_metrics : dict, optional
            基准指标值
        thresholds : dict, optional
            监控阈值配置
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics or {}

        # 默认阈值配置
        default_thresholds = {
            'auc_drop_threshold': 0.05,      # AUC下降阈值
            'ks_drop_threshold': 0.1,        # KS下降阈值
            'significant_change_threshold': 0.15  # 显著变化阈值
        }
        self.thresholds = {**default_thresholds, **(thresholds or {})}

        # 监控历史
        self.monitoring_history = []

        # 设置日志
        self.logger = logging.getLogger(f"ModelStabilityMonitor_{model_name}")

    def set_baseline(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     timestamp: Optional[str] = None) -> Dict:
        """
        设置基准指标

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        timestamp : str, optional
            时间戳

        Returns:
        --------
        baseline_metrics : dict
            基准指标
        """
        from ..evaluation.metrics import calculate_auc, calculate_ks

        auc = calculate_auc(y_true, y_pred)
        ks, _, _ = calculate_ks(y_true, y_pred)

        self.baseline_metrics = {
            'auc': auc,
            'ks': ks,
            'timestamp': timestamp or datetime.now().isoformat(),
            'sample_size': len(y_true),
            'positive_rate': np.mean(y_true)
        }

        self.logger.info(f"基准指标已设置: AUC={auc:.4f}, KS={ks:.4f}")
        return self.baseline_metrics

    def monitor_performance(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           timestamp: Optional[str] = None) -> Dict:
        """
        监控模型性能

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        timestamp : str, optional
            时间戳

        Returns:
        --------
        monitor_result : dict
            监控结果
        """
        if not self.baseline_metrics:
            raise ValueError("请先设置基准指标")

        from ..evaluation.metrics import calculate_auc, calculate_ks

        # 计算当前指标
        current_auc = calculate_auc(y_true, y_pred)
        current_ks, _, _ = calculate_ks(y_true, y_pred)

        # 计算指标变化
        auc_change = current_auc - self.baseline_metrics['auc']
        ks_change = current_ks - self.baseline_metrics['ks']

        # 判断是否触发报警
        alerts = []

        if auc_change < -self.thresholds['auc_drop_threshold']:
            alerts.append({
                'type': 'AUC_DROP',
                'message': f"AUC显著下降: {self.baseline_metrics['auc']:.4f} -> {current_auc:.4f} "
                          f"(下降{abs(auc_change):.4f})",
                'severity': 'HIGH' if abs(auc_change) > 0.1 else 'MEDIUM'
            })

        if ks_change < -self.thresholds['ks_drop_threshold']:
            alerts.append({
                'type': 'KS_DROP',
                'message': f"KS显著下降: {self.baseline_metrics['ks']:.4f} -> {current_ks:.4f} "
                          f"(下降{abs(ks_change):.4f})",
                'severity': 'HIGH' if abs(ks_change) > 0.2 else 'MEDIUM'
            })

        # 确定整体状态
        if len(alerts) == 0:
            status = 'STABLE'
        elif any(alert['severity'] == 'HIGH' for alert in alerts):
            status = 'DEGRADED'
        else:
            status = 'WARNING'

        # 构建监控结果
        result = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'model_name': self.model_name,
            'status': status,
            'metrics': {
                'current_auc': current_auc,
                'current_ks': current_ks,
                'baseline_auc': self.baseline_metrics['auc'],
                'baseline_ks': self.baseline_metrics['ks'],
                'auc_change': auc_change,
                'ks_change': ks_change
            },
            'alerts': alerts,
            'sample_size': len(y_true),
            'positive_rate': np.mean(y_true)
        }

        # 记录监控历史
        self.monitoring_history.append(result)

        self.logger.info(f"性能监控完成: AUC={current_auc:.4f}, KS={current_ks:.4f}")

        return result

    def get_stability_trend(self, days: int = 30) -> pd.DataFrame:
        """
        获取稳定性趋势

        Parameters:
        -----------
        days : int, default=30
            趋势天数

        Returns:
        --------
        trend_df : pd.DataFrame
            趋势数据
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()

        recent_records = [
            record for record in self.monitoring_history
            if record['timestamp'] >= cutoff_str
        ]

        if not recent_records:
            return pd.DataFrame()

        trend_data = []
        for record in recent_records:
            trend_data.append({
                'timestamp': record['timestamp'],
                'auc': record['metrics']['current_auc'],
                'ks': record['metrics']['current_ks'],
                'auc_change': record['metrics']['auc_change'],
                'ks_change': record['metrics']['ks_change'],
                'alert_count': len(record['alerts']),
                'sample_size': record['sample_size']
            })

        return pd.DataFrame(trend_data)

    def get_summary_report(self, days: int = 7) -> Dict:
        """
        获取监控汇总报告

        Parameters:
        -----------
        days : int, default=7
            汇总天数

        Returns:
        --------
        summary : dict
            监控汇总
        """
        trend_df = self.get_stability_trend(days)

        if trend_df.empty:
            return {'message': f"最近{days}天没有监控记录"}

        # 统计报警
        total_alerts = trend_df['alert_count'].sum()

        # 计算稳定性指标
        auc_std = trend_df['auc'].std()
        ks_std = trend_df['ks'].std()

        stability_score = max(0, 100 - (auc_std * 100 + ks_std * 100))

        return {
            'period': f"最近{days}天",
            'total_monitoring_records': len(trend_df),
            'total_alerts': int(total_alerts),
            'stability_score': round(stability_score, 2),
            'auc_volatility': round(auc_std, 4),
            'ks_volatility': round(ks_std, 4),
            'baseline_metrics': self.baseline_metrics,
            'latest_metrics': {
                'auc': trend_df.iloc[-1]['auc'] if len(trend_df) > 0 else None,
                'ks': trend_df.iloc[-1]['ks'] if len(trend_df) > 0 else None
            }
        }


class PerformanceMonitor:
    """
    性能监控器 - 简化版本，专注于核心指标监控
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.history = []

    def log_performance(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       timestamp: Optional[str] = None) -> Dict:
        """
        记录性能指标

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        timestamp : str, optional
            时间戳

        Returns:
        --------
        performance : dict
            性能指标
        """
        from ..evaluation.metrics import calculate_auc, calculate_ks

        performance = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'auc': calculate_auc(y_true, y_pred),
            'ks': calculate_ks(y_true, y_pred)[0],
            'sample_size': len(y_true),
            'positive_rate': np.mean(y_true)
        }

        self.history.append(performance)
        return performance

    def get_performance_history(self) -> pd.DataFrame:
        """获取性能历史"""
        return pd.DataFrame(self.history)

    def compare_with_baseline(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             baseline_auc: float,
                             baseline_ks: float) -> Dict:
        """
        与基准对比

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        baseline_auc : float
            基准AUC
        baseline_ks : float
            基准KS

        Returns:
        --------
        comparison : dict
            对比结果
        """
        from ..evaluation.metrics import calculate_auc, calculate_ks

        current_auc = calculate_auc(y_true, y_pred)
        current_ks = calculate_ks(y_true, y_pred)[0]

        return {
            'current_auc': current_auc,
            'current_ks': current_ks,
            'baseline_auc': baseline_auc,
            'baseline_ks': baseline_ks,
            'auc_change': current_auc - baseline_auc,
            'ks_change': current_ks - baseline_ks,
            'auc_change_pct': (current_auc - baseline_auc) / baseline_auc * 100,
            'ks_change_pct': (current_ks - baseline_ks) / baseline_ks * 100
        }