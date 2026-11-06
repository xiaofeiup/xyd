"""
报警系统模块

提供模型监控的报警功能，包括多种通知方式
"""

import json
import logging
import smtplib
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertManager:
    """
    报警管理器

    管理各种类型的报警通知
    """

    def __init__(self,
                 alert_config: Optional[Dict] = None,
                 custom_handlers: Optional[Dict[str, Callable]] = None):
        """
        初始化报警管理器

        Parameters:
        -----------
        alert_config : dict, optional
            报警配置
        custom_handlers : dict, optional
            自定义报警处理器
        """
        # 默认配置
        default_config = {
            'enable_email': False,
            'enable_log': True,
            'enable_webhook': False,
            'log_level': 'WARNING',
            'email_config': {},
            'webhook_config': {},
            'alert_templates': {}
        }

        self.config = {**default_config, **(alert_config or {})}
        self.custom_handlers = custom_handlers or {}

        # 设置日志
        self._setup_logging()

        # 报警历史
        self.alert_history = []

    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.config['log_level'].upper())
        self.logger = logging.getLogger("AlertManager")
        self.logger.setLevel(log_level)

    def send_alert(self,
                  alert_type: str,
                  message: str,
                  severity: str = 'MEDIUM',
                  context: Optional[Dict] = None,
                  timestamp: Optional[str] = None) -> bool:
        """
        发送报警

        Parameters:
        -----------
        alert_type : str
            报警类型
        message : str
            报警消息
        severity : str, default='MEDIUM'
            严重程度 ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        context : dict, optional
            报警上下文信息
        timestamp : str, optional
            时间戳

        Returns:
        --------
        success : bool
            是否发送成功
        """
        alert = {
            'type': alert_type,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': timestamp or datetime.now().isoformat(),
            'context': context or {}
        }

        # 记录到历史
        self.alert_history.append(alert)

        success = True

        # 日志报警
        if self.config['enable_log']:
            success &= self._send_log_alert(alert)

        # 邮件报警
        if self.config['enable_email']:
            success &= self._send_email_alert(alert)

        # Webhook报警
        if self.config['enable_webhook']:
            success &= self._send_webhook_alert(alert)

        # 自定义处理器
        for handler_name, handler_func in self.custom_handlers.items():
            try:
                handler_func(alert)
            except Exception as e:
                self.logger.error(f"自定义处理器 {handler_name} 失败: {str(e)}")
                success = False

        return success

    def _send_log_alert(self, alert: Dict) -> bool:
        """发送日志报警"""
        try:
            log_message = f"[{alert['severity']}] {alert['type']}: {alert['message']}"

            if alert['severity'] == 'CRITICAL':
                self.logger.critical(log_message)
            elif alert['severity'] == 'HIGH':
                self.logger.error(log_message)
            elif alert['severity'] == 'MEDIUM':
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)

            return True
        except Exception as e:
            self.logger.error(f"日志报警失败: {str(e)}")
            return False

    def _send_email_alert(self, alert: Dict) -> bool:
        """发送邮件报警"""
        try:
            email_config = self.config.get('email_config', {})

            if not email_config.get('smtp_server'):
                self.logger.warning("邮件配置不完整，跳过邮件报警")
                return False

            # 构建邮件内容
            subject = f"[{alert['severity']}] 模型监控报警: {alert['type']}"
            body = self._format_email_body(alert)

            # 发送邮件
            return self._send_email(
                to_addresses=email_config.get('recipients', []),
                subject=subject,
                body=body,
                smtp_config=email_config
            )

        except Exception as e:
            self.logger.error(f"邮件报警失败: {str(e)}")
            return False

    def _send_webhook_alert(self, alert: Dict) -> bool:
        """发送Webhook报警"""
        try:
            import requests

            webhook_config = self.config.get('webhook_config', {})
            webhook_url = webhook_config.get('url')

            if not webhook_url:
                self.logger.warning("Webhook配置不完整，跳过Webhook报警")
                return False

            # 构建payload
            payload = {
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }

            # 发送请求
            response = requests.post(
                webhook_url,
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=webhook_config.get('timeout', 30)
            )

            response.raise_for_status()
            return True

        except Exception as e:
            self.logger.error(f"Webhook报警失败: {str(e)}")
            return False

    def _format_email_body(self, alert: Dict) -> str:
        """格式化邮件正文"""
        template = self.config.get('alert_templates', {}).get('email', '''
报警详情:
========
类型: {type}
严重程度: {severity}
时间: {timestamp}
消息: {message}

上下文信息:
{context}

此邮件由模型监控系统自动发送。
        ''')

        context_str = json.dumps(alert['context'], indent=2, ensure_ascii=False)

        return template.format(
            type=alert['type'],
            severity=alert['severity'],
            timestamp=alert['timestamp'],
            message=alert['message'],
            context=context_str
        )

    def _send_email(self,
                   to_addresses: List[str],
                   subject: str,
                   body: str,
                   smtp_config: Dict) -> bool:
        """发送邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['sender']
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config.get('port', 587)) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()

                if smtp_config.get('username'):
                    server.login(smtp_config['username'], smtp_config['password'])

                server.send_message(msg)

            return True

        except Exception as e:
            self.logger.error(f"SMTP邮件发送失败: {str(e)}")
            return False

    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        获取报警汇总

        Parameters:
        -----------
        hours : int, default=24
            汇总时间范围（小时）

        Returns:
        --------
        summary : dict
            报警汇总
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_str
        ]

        # 按类型和严重程度统计
        type_counts = {}
        severity_counts = {}

        for alert in recent_alerts:
            alert_type = alert['type']
            severity = alert['severity']

            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alert_types': type_counts,
            'severity_distribution': severity_counts,
            'most_frequent_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            'highest_severity': max(severity_counts.keys(), key=lambda x: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x)) if severity_counts else None
        }

    def clear_alert_history(self, keep_days: int = 0):
        """
        清理报警历史

        Parameters:
        -----------
        keep_days : int, default=30
            保留天数
        """
        from datetime import timedelta

        if keep_days <= 0:
            self.alert_history = []
            return
        cutoff_time = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff_time.isoformat()

        self.alert_history = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_str
        ]

        self.logger.info(f"报警历史已清理，保留最近{keep_days}天的记录")


class AlertRule:
    """
    报警规则

    定义何时触发报警的规则
    """

    def __init__(self,
                 rule_name: str,
                 condition: Callable,
                 alert_type: str,
                 severity: str = 'MEDIUM',
                 message_template: str = "规则 {rule_name} 被触发"):
        """
        初始化报警规则

        Parameters:
        -----------
        rule_name : str
            规则名称
        condition : callable
            条件函数，返回布尔值
        alert_type : str
            报警类型
        severity : str, default='MEDIUM'
            严重程度
        message_template : str
            消息模板
        """
        self.rule_name = rule_name
        self.condition = condition
        self.alert_type = alert_type
        self.severity = severity
        self.message_template = message_template

    def evaluate(self, data: Dict) -> Optional[Dict]:
        """
        评估规则

        Parameters:
        -----------
        data : dict
            输入数据

        Returns:
        --------
        alert : dict or None
            如果条件满足，返回报警信息
        """
        try:
            if self.condition(data):
                return {
                    'type': self.alert_type,
                    'severity': self.severity,
                    'message': self.message_template.format(
                        rule_name=self.rule_name,
                        **data
                    ),
                    'context': {
                        'rule_name': self.rule_name,
                        'trigger_data': data
                    }
                }
        except Exception as e:
            logging.error(f"规则 {self.rule_name} 评估失败: {str(e)}")

        return None


class AlertEngine:
    """
    报警引擎

    集成报警管理器和规则引擎
    """

    def __init__(self, alert_manager: AlertManager):
        """
        初始化报警引擎

        Parameters:
        -----------
        alert_manager : AlertManager
            报警管理器
        """
        self.alert_manager = alert_manager
        self.rules = []

    def add_rule(self, rule: Union[AlertRule, None] = None, **kwargs):
        """
        添加报警规则

        Parameters:
        -----------
        rule : AlertRule, optional
            直接传入规则对象
        **kwargs : dict
            用于创建规则的参数，包括：
            - name : str
            - condition : callable
            - alert_type : str
            - message : str
            - severity : str
        """
        if rule is not None:
            self.rules.append(rule)
        else:
            # 从kwargs创建规则
            rule_name = kwargs.get('name', 'unnamed_rule')
            condition = kwargs.get('condition')
            alert_type = kwargs.get('alert_type', 'UNKNOWN')
            severity = kwargs.get('severity', 'MEDIUM')
            message = kwargs.get('message', f"Rule {rule_name} triggered")

            if condition is None:
                raise ValueError("condition is required")

            alert_rule = AlertRule(
                rule_name=rule_name,
                condition=condition,
                alert_type=alert_type,
                severity=severity,
                message_template=message
            )
            self.rules.append(alert_rule)

    def evaluate_all_rules(self, data: Dict) -> List[Dict]:
        """
        评估所有规则

        Parameters:
        -----------
        data : dict
            输入数据

        Returns:
        --------
        triggered_alerts : list
            触发的报警列表
        """
        triggered_alerts = []

        for rule in self.rules:
            alert = rule.evaluate(data)
            if alert:
                # 调整字段名以匹配send_alert方法
                alert_copy = alert.copy()
                if 'type' in alert_copy:
                    alert_copy['alert_type'] = alert_copy.pop('type')

                # 发送报警
                self.alert_manager.send_alert(**alert_copy)
                alert['alert_type'] = alert.get('type', alert_copy.get('alert_type'))
                triggered_alerts.append(alert)

        return triggered_alerts

    def add_standard_rules(self):
        """添加标准监控规则"""
        # AUC下降规则
        self.add_rule(AlertRule(
            rule_name="AUC_DROP_HIGH",
            condition=lambda data: data.get('auc_change', 0) < -0.1,
            alert_type="PERFORMANCE_DEGRADATION",
            severity="HIGH",
            message_template="AUC显著下降 {auc_change:.4f}"
        ))

        # KS下降规则
        self.add_rule(AlertRule(
            rule_name="KS_DROP_HIGH",
            condition=lambda data: data.get('ks_change', 0) < -0.2,
            alert_type="PERFORMANCE_DEGRADATION",
            severity="HIGH",
            message_template="KS显著下降 {ks_change:.4f}"
        ))

        # PSI漂移规则
        self.add_rule(AlertRule(
            rule_name="FEATURE_DRIFT_HIGH",
            condition=lambda data: data.get('drift_rate', 0) > 0.3,
            alert_type="FEATURE_DRIFT",
            severity="HIGH",
            message_template="特征漂移严重，漂移率: {drift_rate:.2%}"
        ))

        # 样本量异常规则
        self.add_rule(AlertRule(
            rule_name="SAMPLE_SIZE_LOW",
            condition=lambda data: data.get('sample_size', 0) < 1000,
            alert_type="DATA_QUALITY",
            severity="MEDIUM",
            message_template="样本量过少: {sample_size}"
        ))
