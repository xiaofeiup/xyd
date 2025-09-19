"""
模型稳定性监控模块

提供模型性能监控、特征稳定性监控和自动报警功能
"""

import pandas as pd
import numpy as np
import warnings
import json
import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
from .gentools.evaluation import calc_auc, calculate_ks, calculate_lift
from .metric_report.psi import calculate_psi, calculate_multi_feature_psi


class ModelStabilityMonitor:
    """
    模型稳定性监控器

    监控模型性能指标(AUC, KS)和特征稳定性(PSI)的变化，
    当指标出现显著波动时触发报警
    """

    def __init__(self,
                 model_name: str,
                 baseline_metrics: Optional[Dict] = None,
                 thresholds: Optional[Dict] = None,
                 alert_config: Optional[Dict] = None):
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
        alert_config : dict, optional
            报警配置
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics or {}

        # 默认阈值配置
        default_thresholds = {
            'auc_drop_threshold': 0.05,      # AUC下降阈值
            'ks_drop_threshold': 0.1,        # KS下降阈值
            'psi_threshold': 0.25,           # PSI阈值
            'significant_change_threshold': 0.15  # 显著变化阈值
        }
        self.thresholds = {**default_thresholds, **(thresholds or {})}

        # 默认报警配置
        default_alert_config = {
            'enable_email': False,
            'enable_log': True,
            'log_level': 'WARNING',
            'recipients': []
        }
        self.alert_config = {**default_alert_config, **(alert_config or {})}

        # 监控历史
        self.monitoring_history = []

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.alert_config['log_level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"ModelMonitor_{self.model_name}")

    def set_baseline(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     feature_data: Optional[pd.DataFrame] = None,
                     timestamp: Optional[str] = None):
        """
        设置基准指标

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测概率
        feature_data : pd.DataFrame, optional
            特征数据，用于计算基准PSI
        timestamp : str, optional
            时间戳
        """
        try:
            # 计算模型性能指标
            auc = calc_auc(y_true, y_pred)
            ks, _, _ = calculate_ks(y_true, y_pred)

            self.baseline_metrics = {
                'auc': auc,
                'ks': ks,
                'timestamp': timestamp or datetime.datetime.now().isoformat(),
                'sample_size': len(y_true),
                'positive_rate': np.mean(y_true)
            }

            # 保存特征基准数据
            if feature_data is not None:
                self.baseline_features = feature_data.copy()

            self.logger.info(f"基准指标已设置: AUC={auc:.4f}, KS={ks:.4f}")

        except Exception as e:
            self.logger.error(f"设置基准指标失败: {str(e)}")
            raise

    def monitor_model_performance(self,
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

        try:
            # 计算当前指标
            current_auc = calc_auc(y_true, y_pred)
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

            # 构建监控结果
            result = {
                'timestamp': timestamp or datetime.datetime.now().isoformat(),
                'model_name': self.model_name,
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

            # 触发报警
            if alerts:
                self._trigger_alerts(alerts, result)

            self.logger.info(f"模型性能监控完成: AUC={current_auc:.4f}, KS={current_ks:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"模型性能监控失败: {str(e)}")
            raise

    def monitor_feature_stability(self,
                                  current_features: pd.DataFrame,
                                  feature_list: Optional[List[str]] = None,
                                  timestamp: Optional[str] = None) -> Dict:
        """
        监控特征稳定性

        Parameters:
        -----------
        current_features : pd.DataFrame
            当前特征数据
        feature_list : list, optional
            要监控的特征列表
        timestamp : str, optional
            时间戳

        Returns:
        --------
        stability_result : dict
            稳定性监控结果
        """
        if not hasattr(self, 'baseline_features'):
            raise ValueError("请先设置包含特征数据的基准")

        try:
            # 计算PSI
            psi_results = calculate_multi_feature_psi(
                self.baseline_features,
                current_features,
                features=feature_list,
                bins=10
            )

            # 识别不稳定特征
            unstable_features = psi_results[
                psi_results['psi_value'] >= self.thresholds['psi_threshold']
            ]

            # 生成报警
            alerts = []
            for _, feature in unstable_features.iterrows():
                severity = 'HIGH' if feature['psi_value'] > 0.5 else 'MEDIUM'
                alerts.append({
                    'type': 'FEATURE_DRIFT',
                    'feature': feature['feature'],
                    'psi_value': feature['psi_value'],
                    'message': f"特征 {feature['feature']} PSI异常: {feature['psi_value']:.4f} "
                              f"({feature['interpretation']})",
                    'severity': severity
                })

            # 构建结果
            result = {
                'timestamp': timestamp or datetime.datetime.now().isoformat(),
                'model_name': self.model_name,
                'psi_summary': psi_results.to_dict('records'),
                'unstable_features': len(unstable_features),
                'total_features': len(psi_results),
                'stability_rate': (len(psi_results) - len(unstable_features)) / len(psi_results),
                'alerts': alerts
            }

            # 触发报警
            if alerts:
                self._trigger_alerts(alerts, result)

            self.logger.info(f"特征稳定性监控完成: {len(unstable_features)}/{len(psi_results)} 特征不稳定")

            return result

        except Exception as e:
            self.logger.error(f"特征稳定性监控失败: {str(e)}")
            raise

    def _trigger_alerts(self, alerts: List[Dict], context: Dict):
        """触发报警"""
        for alert in alerts:
            if self.alert_config['enable_log']:
                if alert['severity'] == 'HIGH':
                    self.logger.error(f"[{alert['type']}] {alert['message']}")
                else:
                    self.logger.warning(f"[{alert['type']}] {alert['message']}")

            # 这里可以扩展其他报警方式（邮件、短信、webhook等）
            if self.alert_config['enable_email']:
                self._send_email_alert(alert, context)

    def _send_email_alert(self, alert: Dict, context: Dict):
        """发送邮件报警（示例实现）"""
        # 这里是邮件发送的示例框架
        # 实际使用时需要配置SMTP服务器
        self.logger.info(f"邮件报警: {alert['message']}")

    def get_monitoring_summary(self, days: int = 7) -> Dict:
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
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()

        recent_records = [
            record for record in self.monitoring_history
            if record['timestamp'] >= cutoff_str
        ]

        if not recent_records:
            return {'message': f"最近{days}天没有监控记录"}

        # 统计报警
        all_alerts = []
        for record in recent_records:
            all_alerts.extend(record.get('alerts', []))

        alert_summary = {}
        for alert in all_alerts:
            alert_type = alert['type']
            if alert_type not in alert_summary:
                alert_summary[alert_type] = {'count': 0, 'severities': []}
            alert_summary[alert_type]['count'] += 1
            alert_summary[alert_type]['severities'].append(alert['severity'])

        return {
            'period': f"最近{days}天",
            'total_monitoring_records': len(recent_records),
            'total_alerts': len(all_alerts),
            'alert_summary': alert_summary,
            'baseline_metrics': self.baseline_metrics,
            'thresholds': self.thresholds
        }

    def export_monitoring_report(self, filepath: str, format: str = 'json'):
        """
        导出监控报告

        Parameters:
        -----------
        filepath : str
            文件路径
        format : str, default='json'
            导出格式 ('json', 'csv')
        """
        try:
            if format == 'json':
                report = {
                    'model_name': self.model_name,
                    'baseline_metrics': self.baseline_metrics,
                    'thresholds': self.thresholds,
                    'monitoring_history': self.monitoring_history,
                    'export_time': datetime.datetime.now().isoformat()
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

            elif format == 'csv':
                # 将监控历史转换为DataFrame并导出
                if self.monitoring_history:
                    df_records = []
                    for record in self.monitoring_history:
                        flat_record = {
                            'timestamp': record['timestamp'],
                            'model_name': record['model_name'],
                            **record.get('metrics', {}),
                            'alert_count': len(record.get('alerts', []))
                        }
                        df_records.append(flat_record)

                    pd.DataFrame(df_records).to_csv(filepath, index=False, encoding='utf-8-sig')
                else:
                    pd.DataFrame().to_csv(filepath, index=False)

            self.logger.info(f"监控报告已导出: {filepath}")

        except Exception as e:
            self.logger.error(f"导出监控报告失败: {str(e)}")
            raise


def create_monitoring_dashboard(monitor_results: List[Dict]) -> pd.DataFrame:
    """
    创建监控仪表板数据

    Parameters:
    -----------
    monitor_results : list
        监控结果列表

    Returns:
    --------
    dashboard_data : pd.DataFrame
        仪表板数据
    """
    if not monitor_results:
        return pd.DataFrame()

    dashboard_records = []
    for result in monitor_results:
        record = {
            'timestamp': result['timestamp'],
            'model_name': result['model_name'],
            'alert_count': len(result.get('alerts', [])),
            'sample_size': result.get('sample_size', 0)
        }

        # 添加指标数据
        if 'metrics' in result:
            record.update(result['metrics'])

        # 添加稳定性数据
        if 'stability_rate' in result:
            record['stability_rate'] = result['stability_rate']
            record['unstable_features'] = result.get('unstable_features', 0)

        dashboard_records.append(record)

    return pd.DataFrame(dashboard_records)


# 使用示例函数
def example_usage():
    """
    使用示例
    """
    print("模型监控使用示例:")
    print("1. 创建监控器")
    print("   monitor = ModelStabilityMonitor('my_model')")
    print("2. 设置基准")
    print("   monitor.set_baseline(y_true_baseline, y_pred_baseline, features_baseline)")
    print("3. 监控性能")
    print("   result = monitor.monitor_model_performance(y_true_new, y_pred_new)")
    print("4. 监控特征稳定性")
    print("   stability = monitor.monitor_feature_stability(features_new)")
    print("5. 获取汇总报告")
    print("   summary = monitor.get_monitoring_summary(days=30)")


if __name__ == "__main__":
    example_usage()