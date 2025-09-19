"""
自动化监控和报警工作流

提供完整的模型监控工作流，包括定时监控、报警触发、报告生成等功能
"""

import pandas as pd
import numpy as np
import json
import datetime
import time
import schedule
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path

from model_tools.model_monitor import ModelStabilityMonitor


class AutomatedMonitoringWorkflow:
    """
    自动化监控工作流类

    实现定时监控、自动报警、报告生成等功能
    """

    def __init__(self,
                 config_path: str,
                 data_loader_func: Optional[Callable] = None,
                 notification_func: Optional[Callable] = None):
        """
        初始化工作流

        Parameters:
        -----------
        config_path : str
            配置文件路径
        data_loader_func : callable, optional
            数据加载函数
        notification_func : callable, optional
            通知发送函数
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader_func = data_loader_func
        self.notification_func = notification_func

        # 初始化监控器
        self.monitors = {}
        self._init_monitors()

        # 设置日志
        self._setup_logging()

        # 监控状态
        self.is_running = False
        self.monitoring_stats = {
            'total_checks': 0,
            'total_alerts': 0,
            'last_check_time': None,
            'start_time': None
        }

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {str(e)}")

    def _init_monitors(self):
        """初始化模型监控器"""
        for model_config in self.config.get('models', []):
            model_name = model_config['name']
            thresholds = model_config.get('thresholds', {})
            alert_config = model_config.get('alert_config', {})

            monitor = ModelStabilityMonitor(
                model_name=model_name,
                thresholds=thresholds,
                alert_config=alert_config
            )

            # 如果有基准数据配置，加载基准
            if 'baseline_data' in model_config:
                self._load_baseline(monitor, model_config['baseline_data'])

            self.monitors[model_name] = monitor

    def _load_baseline(self, monitor: ModelStabilityMonitor, baseline_config: Dict):
        """加载基准数据"""
        try:
            if self.data_loader_func:
                # 使用自定义数据加载函数
                baseline_data = self.data_loader_func(baseline_config)
                y_true = baseline_data['y_true']
                y_pred = baseline_data['y_pred']
                features = baseline_data.get('features')
            else:
                # 使用默认的文件加载方式
                if 'file_path' in baseline_config:
                    df = pd.read_csv(baseline_config['file_path'])
                    y_true = df[baseline_config['target_column']]
                    y_pred = df[baseline_config['pred_column']]
                    feature_cols = baseline_config.get('feature_columns', [])
                    features = df[feature_cols] if feature_cols else None
                else:
                    raise ValueError("需要提供baseline数据配置")

            monitor.set_baseline(y_true, y_pred, features)
            logging.info(f"已为模型 {monitor.model_name} 设置基准数据")

        except Exception as e:
            logging.error(f"加载基准数据失败: {str(e)}")
            raise

    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'monitoring_workflow.log')

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AutomatedMonitoring")

    def run_monitoring_cycle(self):
        """执行一次监控周期"""
        try:
            self.monitoring_stats['total_checks'] += 1
            self.monitoring_stats['last_check_time'] = datetime.datetime.now().isoformat()

            all_alerts = []

            for model_name, monitor in self.monitors.items():
                try:
                    # 获取模型配置
                    model_config = next(
                        config for config in self.config['models']
                        if config['name'] == model_name
                    )

                    # 加载当前数据
                    current_data = self._load_current_data(model_config)

                    if current_data is None:
                        self.logger.warning(f"模型 {model_name} 无法加载当前数据，跳过监控")
                        continue

                    # 执行性能监控
                    if 'y_true' in current_data and 'y_pred' in current_data:
                        performance_result = monitor.monitor_model_performance(
                            current_data['y_true'],
                            current_data['y_pred']
                        )
                        all_alerts.extend(performance_result.get('alerts', []))

                    # 执行特征稳定性监控
                    if 'features' in current_data:
                        stability_result = monitor.monitor_feature_stability(
                            current_data['features']
                        )
                        all_alerts.extend(stability_result.get('alerts', []))

                except Exception as e:
                    self.logger.error(f"监控模型 {model_name} 时出错: {str(e)}")
                    continue

            # 处理报警
            if all_alerts:
                self.monitoring_stats['total_alerts'] += len(all_alerts)
                self._process_alerts(all_alerts)

            # 生成监控报告
            self._generate_monitoring_report()

            self.logger.info(f"监控周期完成，共检查 {len(self.monitors)} 个模型，产生 {len(all_alerts)} 个报警")

        except Exception as e:
            self.logger.error(f"监控周期执行失败: {str(e)}")
            raise

    def _load_current_data(self, model_config: Dict) -> Optional[Dict]:
        """加载当前数据"""
        try:
            current_data_config = model_config.get('current_data')
            if not current_data_config:
                return None

            if self.data_loader_func:
                return self.data_loader_func(current_data_config)
            else:
                # 默认文件加载方式
                if 'file_path' in current_data_config:
                    df = pd.read_csv(current_data_config['file_path'])
                    result = {}

                    if 'target_column' in current_data_config:
                        result['y_true'] = df[current_data_config['target_column']]
                    if 'pred_column' in current_data_config:
                        result['y_pred'] = df[current_data_config['pred_column']]

                    feature_cols = current_data_config.get('feature_columns', [])
                    if feature_cols:
                        result['features'] = df[feature_cols]

                    return result

            return None

        except Exception as e:
            self.logger.error(f"加载当前数据失败: {str(e)}")
            return None

    def _process_alerts(self, alerts: List[Dict]):
        """处理报警"""
        # 按严重性分组报警
        high_priority_alerts = [alert for alert in alerts if alert.get('severity') == 'HIGH']
        medium_priority_alerts = [alert for alert in alerts if alert.get('severity') == 'MEDIUM']

        # 发送通知
        if self.notification_func:
            try:
                self.notification_func(alerts)
            except Exception as e:
                self.logger.error(f"发送通知失败: {str(e)}")

        # 记录报警到文件
        alert_log_path = self.config.get('alert_log_path', 'alerts.json')
        self._save_alerts_to_file(alerts, alert_log_path)

        # 输出报警摘要
        if high_priority_alerts:
            self.logger.error(f"检测到 {len(high_priority_alerts)} 个高优先级报警")
        if medium_priority_alerts:
            self.logger.warning(f"检测到 {len(medium_priority_alerts)} 个中等优先级报警")

    def _save_alerts_to_file(self, alerts: List[Dict], file_path: str):
        """保存报警到文件"""
        try:
            alert_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'alerts': alerts,
                'alert_count': len(alerts)
            }

            # 读取现有记录
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
            else:
                existing_records = []

            # 添加新记录
            existing_records.append(alert_record)

            # 保持最近100条记录
            if len(existing_records) > 100:
                existing_records = existing_records[-100:]

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"保存报警记录失败: {str(e)}")

    def _generate_monitoring_report(self):
        """生成监控报告"""
        try:
            report_config = self.config.get('reporting', {})
            if not report_config.get('enabled', False):
                return

            report_path = report_config.get('path', 'monitoring_report.json')

            # 收集所有监控器的汇总信息
            report_data = {
                'generation_time': datetime.datetime.now().isoformat(),
                'workflow_stats': self.monitoring_stats,
                'models': {}
            }

            for model_name, monitor in self.monitors.items():
                summary = monitor.get_monitoring_summary(
                    days=report_config.get('summary_days', 7)
                )
                report_data['models'][model_name] = summary

            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"监控报告已生成: {report_path}")

        except Exception as e:
            self.logger.error(f"生成监控报告失败: {str(e)}")

    def start_scheduled_monitoring(self):
        """启动定时监控"""
        schedule_config = self.config.get('schedule', {})
        if not schedule_config.get('enabled', False):
            self.logger.warning("定时监控未启用")
            return

        # 设置定时任务
        interval = schedule_config.get('interval', 'daily')
        time_str = schedule_config.get('time', '09:00')

        if interval == 'hourly':
            schedule.every().hour.do(self.run_monitoring_cycle)
        elif interval == 'daily':
            schedule.every().day.at(time_str).do(self.run_monitoring_cycle)
        elif interval == 'weekly':
            day = schedule_config.get('day', 'monday')
            getattr(schedule.every(), day).at(time_str).do(self.run_monitoring_cycle)

        self.is_running = True
        self.monitoring_stats['start_time'] = datetime.datetime.now().isoformat()

        self.logger.info(f"定时监控已启动，间隔: {interval}，时间: {time_str}")

        # 运行调度器
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            self.stop_monitoring()
        except Exception as e:
            self.logger.error(f"定时监控运行出错: {str(e)}")
            raise

    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        schedule.clear()
        self.logger.info("监控已停止")

    def run_once(self):
        """运行一次监控（非定时）"""
        self.logger.info("开始执行单次监控...")
        if not self.monitoring_stats['start_time']:
            self.monitoring_stats['start_time'] = datetime.datetime.now().isoformat()

        self.run_monitoring_cycle()
        self.logger.info("单次监控完成")

    def get_workflow_status(self) -> Dict:
        """获取工作流状态"""
        return {
            'is_running': self.is_running,
            'model_count': len(self.monitors),
            'config_path': self.config_path,
            'stats': self.monitoring_stats
        }


def create_default_config(config_path: str):
    """
    创建默认配置文件

    Parameters:
    -----------
    config_path : str
        配置文件保存路径
    """
    default_config = {
        "models": [
            {
                "name": "example_model",
                "thresholds": {
                    "auc_drop_threshold": 0.05,
                    "ks_drop_threshold": 0.1,
                    "psi_threshold": 0.25
                },
                "alert_config": {
                    "enable_email": False,
                    "enable_log": True,
                    "log_level": "WARNING"
                },
                "baseline_data": {
                    "file_path": "baseline_data.csv",
                    "target_column": "target",
                    "pred_column": "pred_prob",
                    "feature_columns": ["feature1", "feature2", "feature3"]
                },
                "current_data": {
                    "file_path": "current_data.csv",
                    "target_column": "target",
                    "pred_column": "pred_prob",
                    "feature_columns": ["feature1", "feature2", "feature3"]
                }
            }
        ],
        "schedule": {
            "enabled": True,
            "interval": "daily",
            "time": "09:00"
        },
        "reporting": {
            "enabled": True,
            "path": "monitoring_report.json",
            "summary_days": 7
        },
        "logging": {
            "level": "INFO",
            "file": "monitoring_workflow.log"
        },
        "alert_log_path": "alerts.json"
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)

    print(f"默认配置文件已创建: {config_path}")


def example_notification_function(alerts: List[Dict]):
    """
    示例通知函数

    Parameters:
    -----------
    alerts : list
        报警列表
    """
    print(f"\n=== 模型监控报警 ({datetime.datetime.now()}) ===")
    for alert in alerts:
        print(f"[{alert.get('severity', 'UNKNOWN')}] {alert.get('message', '未知报警')}")
    print("=" * 50)


def example_data_loader(data_config: Dict) -> Dict:
    """
    示例数据加载函数

    Parameters:
    -----------
    data_config : dict
        数据配置

    Returns:
    --------
    data : dict
        加载的数据
    """
    # 这里可以实现从数据库、API等加载数据的逻辑
    # 示例：从文件加载
    if 'file_path' in data_config:
        df = pd.read_csv(data_config['file_path'])
        return {
            'y_true': df[data_config['target_column']],
            'y_pred': df[data_config['pred_column']],
            'features': df[data_config.get('feature_columns', [])]
        }
    return {}


if __name__ == "__main__":
    # 使用示例
    print("自动化监控工作流使用示例:")
    print("1. 创建配置文件:")
    print("   create_default_config('config.json')")
    print("2. 启动工作流:")
    print("   workflow = AutomatedMonitoringWorkflow('config.json')")
    print("   workflow.start_scheduled_monitoring()")
    print("3. 运行单次监控:")
    print("   workflow.run_once()")