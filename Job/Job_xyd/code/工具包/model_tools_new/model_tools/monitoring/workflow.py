"""
自动化工作流模块

提供模型监控的自动化工作流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
import threading
import time


class MonitoringWorkflow:
    """
    监控工作流

    集成模型稳定性监控、特征漂移检测和报警系统的自动化工作流
    """

    def __init__(self,
                 model_name: str,
                 stability_monitor=None,
                 drift_detector=None,
                 alert_manager=None):
        """
        初始化监控工作流

        Parameters:
        -----------
        model_name : str
            模型名称
        stability_monitor : ModelStabilityMonitor, optional
            稳定性监控器
        drift_detector : FeatureDriftDetector, optional
            漂移检测器
        alert_manager : AlertManager, optional
            报警管理器
        """
        self.model_name = model_name
        self.stability_monitor = stability_monitor
        self.drift_detector = drift_detector
        self.alert_manager = alert_manager

        self.workflow_log = []
        self.is_running = False
        self.monitoring_thread = None

        self.logger = logging.getLogger(f"MonitoringWorkflow_{model_name}")

    def setup_baseline(self,
                      X_baseline: pd.DataFrame,
                      y_baseline: pd.Series,
                      y_pred_baseline: np.ndarray) -> Dict:
        """
        设置基准数据

        Parameters:
        -----------
        X_baseline : pd.DataFrame
            基准特征数据
        y_baseline : pd.Series
            基准目标变量
        y_pred_baseline : np.ndarray
            基准预测结果

        Returns:
        --------
        setup_result : dict
            设置结果
        """
        setup_result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'components_setup': {}
        }

        # 设置稳定性监控基准
        if self.stability_monitor:
            try:
                baseline_metrics = self.stability_monitor.set_baseline(
                    y_baseline, y_pred_baseline
                )
                setup_result['components_setup']['stability'] = {
                    'status': 'success',
                    'metrics': baseline_metrics
                }
                self.logger.info(f"稳定性监控基准设置成功: AUC={baseline_metrics['auc']:.4f}")
            except Exception as e:
                setup_result['components_setup']['stability'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.logger.error(f"稳定性监控基准设置失败: {e}")

        # 设置漂移检测基准
        if self.drift_detector:
            try:
                self.drift_detector.set_baseline(X_baseline)
                setup_result['components_setup']['drift'] = {
                    'status': 'success',
                    'features_count': len(X_baseline.columns)
                }
                self.logger.info(f"漂移检测基准设置成功: {len(X_baseline.columns)}个特征")
            except Exception as e:
                setup_result['components_setup']['drift'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.logger.error(f"漂移检测基准设置失败: {e}")

        self.workflow_log.append(setup_result)
        return setup_result

    def run_monitoring_cycle(self,
                           X_current: pd.DataFrame,
                           y_current: pd.Series,
                           y_pred_current: np.ndarray,
                           timestamp: Optional[str] = None) -> Dict:
        """
        运行一次监控周期

        Parameters:
        -----------
        X_current : pd.DataFrame
            当前特征数据
        y_current : pd.Series
            当前目标变量
        y_pred_current : np.ndarray
            当前预测结果
        timestamp : str, optional
            时间戳

        Returns:
        --------
        monitoring_result : dict
            监控结果
        """
        cycle_timestamp = timestamp or datetime.now().isoformat()

        monitoring_result = {
            'timestamp': cycle_timestamp,
            'model_name': self.model_name,
            'data_size': len(X_current),
            'results': {},
            'alerts': [],
            'overall_status': 'normal'
        }

        # 运行稳定性监控
        if self.stability_monitor:
            try:
                stability_result = self.stability_monitor.monitor_performance(
                    y_current, y_pred_current, timestamp=cycle_timestamp
                )
                monitoring_result['results']['stability'] = stability_result

                # 收集稳定性报警
                if stability_result['alerts']:
                    monitoring_result['alerts'].extend(stability_result['alerts'])

                self.logger.info(f"稳定性监控完成: AUC={stability_result['metrics']['current_auc']:.4f}")

            except Exception as e:
                monitoring_result['results']['stability'] = {'error': str(e)}
                self.logger.error(f"稳定性监控失败: {e}")

        # 运行漂移检测
        if self.drift_detector:
            try:
                drift_result = self.drift_detector.detect_drift(
                    X_current, timestamp=cycle_timestamp
                )
                monitoring_result['results']['drift'] = drift_result

                # 收集漂移报警
                if drift_result['drift_alerts']:
                    for alert in drift_result['drift_alerts']:
                        monitoring_result['alerts'].append({
                            'type': 'FEATURE_DRIFT',
                            'message': f"特征{alert['feature']}发生漂移: PSI={alert['psi_value']:.3f}",
                            'severity': alert['severity'],
                            'context': alert
                        })

                self.logger.info(f"漂移检测完成: 漂移率={drift_result['drift_rate']:.2%}")

            except Exception as e:
                monitoring_result['results']['drift'] = {'error': str(e)}
                self.logger.error(f"漂移检测失败: {e}")

        # 确定总体状态
        if monitoring_result['alerts']:
            high_severity_alerts = [a for a in monitoring_result['alerts'] if a.get('severity') in ['HIGH', 'CRITICAL']]
            if high_severity_alerts:
                monitoring_result['overall_status'] = 'critical'
            else:
                monitoring_result['overall_status'] = 'warning'

        # 发送报警
        if self.alert_manager and monitoring_result['alerts']:
            for alert in monitoring_result['alerts']:
                try:
                    self.alert_manager.send_alert(
                        alert_type=alert['type'],
                        message=alert['message'],
                        severity=alert['severity'],
                        context=alert.get('context', {})
                    )
                except Exception as e:
                    self.logger.error(f"发送报警失败: {e}")

        self.workflow_log.append(monitoring_result)
        return monitoring_result

    def start_automated_monitoring(self,
                                 data_source: Callable,
                                 interval_minutes: int = 60) -> None:
        """
        启动自动化监控

        Parameters:
        -----------
        data_source : callable
            数据源函数，应返回(X, y, y_pred)元组
        interval_minutes : int, default=60
            监控间隔（分钟）
        """
        if self.is_running:
            self.logger.warning("监控已在运行中")
            return

        self.is_running = True

        def monitoring_loop():
            while self.is_running:
                try:
                    # 获取数据
                    X, y, y_pred = data_source()

                    # 运行监控周期
                    result = self.run_monitoring_cycle(X, y, y_pred)

                    self.logger.info(f"监控周期完成: 状态={result['overall_status']}, 报警数={len(result['alerts'])}")

                    # 等待下一个周期
                    time.sleep(interval_minutes * 60)

                except Exception as e:
                    self.logger.error(f"自动化监控出错: {e}")
                    time.sleep(60)  # 出错后等待1分钟再重试

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info(f"自动化监控已启动，间隔: {interval_minutes}分钟")

    def stop_automated_monitoring(self) -> None:
        """停止自动化监控"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("自动化监控已停止")

    def get_monitoring_summary(self, hours: int = 24) -> Dict:
        """
        获取监控总结

        Parameters:
        -----------
        hours : int, default=24
            总结时间范围（小时）

        Returns:
        --------
        summary : dict
            监控总结
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        recent_logs = [
            log for log in self.workflow_log
            if log.get('timestamp', '') >= cutoff_str and 'results' in log
        ]

        if not recent_logs:
            return {'message': f'最近{hours}小时没有监控记录'}

        # 统计报警
        total_alerts = sum(len(log['alerts']) for log in recent_logs)
        alert_types = {}
        severity_counts = {}

        for log in recent_logs:
            for alert in log['alerts']:
                alert_type = alert['type']
                severity = alert['severity']

                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # 计算稳定性趋势
        stability_metrics = []
        for log in recent_logs:
            stability_result = log['results'].get('stability')
            if stability_result and 'metrics' in stability_result:
                stability_metrics.append(stability_result['metrics'])

        # 计算漂移趋势
        drift_rates = []
        for log in recent_logs:
            drift_result = log['results'].get('drift')
            if drift_result and 'drift_rate' in drift_result:
                drift_rates.append(drift_result['drift_rate'])

        summary = {
            'period_hours': hours,
            'total_monitoring_cycles': len(recent_logs),
            'total_alerts': total_alerts,
            'alert_distribution': {
                'by_type': alert_types,
                'by_severity': severity_counts
            }
        }

        if stability_metrics:
            auc_values = [m['current_auc'] for m in stability_metrics if 'current_auc' in m]
            if auc_values:
                summary['stability_trend'] = {
                    'avg_auc': np.mean(auc_values),
                    'auc_volatility': np.std(auc_values),
                    'latest_auc': auc_values[-1]
                }

        if drift_rates:
            summary['drift_trend'] = {
                'avg_drift_rate': np.mean(drift_rates),
                'max_drift_rate': np.max(drift_rates),
                'latest_drift_rate': drift_rates[-1]
            }

        return summary

    def export_monitoring_report(self, filepath: str) -> None:
        """
        导出监控报告

        Parameters:
        -----------
        filepath : str
            导出文件路径
        """
        # 准备报告数据
        report_data = []

        for log in self.workflow_log:
            if 'results' in log:
                # 基础信息
                base_info = {
                    'timestamp': log['timestamp'],
                    'model_name': log['model_name'],
                    'data_size': log['data_size'],
                    'overall_status': log['overall_status'],
                    'alerts_count': len(log['alerts'])
                }

                # 稳定性指标
                stability_result = log['results'].get('stability', {})
                if 'metrics' in stability_result:
                    metrics = stability_result['metrics']
                    base_info.update({
                        'current_auc': metrics.get('current_auc'),
                        'current_ks': metrics.get('current_ks'),
                        'auc_change': metrics.get('auc_change'),
                        'ks_change': metrics.get('ks_change')
                    })

                # 漂移指标
                drift_result = log['results'].get('drift', {})
                base_info.update({
                    'drifted_features': drift_result.get('drifted_features'),
                    'drift_rate': drift_result.get('drift_rate'),
                    'drift_status': drift_result.get('overall_status')
                })

                report_data.append(base_info)

        # 导出到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 监控总览
            overview_df = pd.DataFrame(report_data)
            overview_df.to_excel(writer, sheet_name='monitoring_overview', index=False)

            # 监控总结
            summary = self.get_monitoring_summary(hours=24*7)  # 一周总结
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='weekly_summary', index=False)

        self.logger.info(f"监控报告已导出到: {filepath}")