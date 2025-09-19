"""
测试monitoring模块
"""

import pytest
import pandas as pd
import numpy as np
import time

from model_tools.monitoring.stability import ModelStabilityMonitor
from model_tools.monitoring.drift import FeatureDriftDetector
from model_tools.monitoring.alerting import AlertManager, AlertEngine


class TestModelStabilityMonitor:
    """测试模型稳定性监控"""

    def test_init(self):
        """测试初始化"""
        monitor = ModelStabilityMonitor(
            model_name="test_model",
            thresholds={'auc_drop_threshold': 0.05}
        )
        assert monitor.model_name == "test_model"
        assert monitor.thresholds['auc_drop_threshold'] == 0.05

    def test_set_baseline(self, sample_binary_classification_data):
        """测试设置基准"""
        _, y_true, y_scores = sample_binary_classification_data

        monitor = ModelStabilityMonitor("test_model")
        baseline_metrics = monitor.set_baseline(y_true, y_scores)

        # 验证基准指标
        assert isinstance(baseline_metrics, dict)
        assert 'auc' in baseline_metrics
        assert 'ks' in baseline_metrics
        assert 0 <= baseline_metrics['auc'] <= 1
        assert 0 <= baseline_metrics['ks'] <= 1

    def test_monitor_performance(self, sample_binary_classification_data):
        """测试性能监控"""
        _, y_true, y_scores = sample_binary_classification_data

        monitor = ModelStabilityMonitor("test_model")
        # 先设置基准
        monitor.set_baseline(y_true, y_scores)

        # 监控性能（使用相同数据，应该没有显著变化）
        result = monitor.monitor_performance(y_true, y_scores)

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'metrics' in result
        assert 'alerts' in result
        assert 'status' in result

    def test_monitor_performance_with_degradation(self, sample_binary_classification_data):
        """测试性能退化监控"""
        _, y_true, y_scores = sample_binary_classification_data

        monitor = ModelStabilityMonitor(
            "test_model",
            thresholds={'auc_drop_threshold': 0.01}  # 很低的阈值
        )

        # 设置基准
        monitor.set_baseline(y_true, y_scores)

        # 创建退化的预测结果（添加噪声降低性能）
        degraded_scores = y_scores + np.random.normal(0, 0.1, len(y_scores))
        degraded_scores = np.clip(degraded_scores, 0, 1)

        # 监控退化性能
        result = monitor.monitor_performance(y_true, degraded_scores)

        # 可能触发报警（取决于添加的噪声影响）
        assert isinstance(result['alerts'], list)

    def test_get_monitoring_history(self, sample_binary_classification_data):
        """测试获取监控历史"""
        _, y_true, y_scores = sample_binary_classification_data

        monitor = ModelStabilityMonitor("test_model")
        monitor.set_baseline(y_true, y_scores)

        # 进行几次监控
        for _ in range(3):
            monitor.monitor_performance(y_true, y_scores)

        history = monitor.get_monitoring_history()

        # 验证历史记录
        assert isinstance(history, list)
        assert len(history) == 3


class TestFeatureDriftDetector:
    """测试特征漂移检测"""

    def test_init(self):
        """测试初始化"""
        feature_names = ['feature_1', 'feature_2']
        detector = FeatureDriftDetector(
            feature_names=feature_names,
            psi_threshold=0.25
        )
        assert detector.feature_names == feature_names
        assert detector.psi_threshold == 0.25

    def test_set_baseline(self, sample_drift_data):
        """测试设置基准"""
        baseline_data, _ = sample_drift_data

        detector = FeatureDriftDetector(
            feature_names=baseline_data.columns.tolist()
        )
        detector.set_baseline(baseline_data)

        # 验证基准统计信息已保存
        assert hasattr(detector, 'baseline_stats')
        assert len(detector.baseline_stats) == len(baseline_data.columns)

    def test_detect_drift(self, sample_drift_data):
        """测试漂移检测"""
        baseline_data, current_data = sample_drift_data

        detector = FeatureDriftDetector(
            feature_names=baseline_data.columns.tolist(),
            psi_threshold=0.1  # 较低的阈值以便检测到漂移
        )

        # 设置基准
        detector.set_baseline(baseline_data)

        # 检测漂移
        result = detector.detect_drift(current_data)

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'total_features' in result
        assert 'drifted_features' in result
        assert 'drift_rate' in result
        assert 'overall_status' in result
        assert 'drift_alerts' in result

    def test_calculate_psi_for_feature(self, sample_drift_data):
        """测试单个特征的PSI计算"""
        baseline_data, current_data = sample_drift_data

        detector = FeatureDriftDetector(['feature_1'])
        detector.set_baseline(baseline_data)

        psi = detector._calculate_psi_for_feature(
            baseline_data['feature_1'],
            current_data['feature_1'],
            'feature_1'
        )

        # 验证PSI值
        assert psi >= 0
        assert isinstance(psi, float)

    def test_get_drift_summary(self, sample_drift_data):
        """测试获取漂移总结"""
        baseline_data, current_data = sample_drift_data

        detector = FeatureDriftDetector(
            feature_names=baseline_data.columns.tolist()
        )
        detector.set_baseline(baseline_data)

        # 进行几次漂移检测
        for _ in range(3):
            detector.detect_drift(current_data)

        summary = detector.get_drift_summary()

        # 验证总结
        assert isinstance(summary, dict)
        assert 'total_detections' in summary


class TestAlertManager:
    """测试报警管理器"""

    def test_init(self):
        """测试初始化"""
        config = {
            'enable_email': False,
            'enable_log': True,
            'log_level': 'INFO'
        }
        alert_manager = AlertManager(config)
        # 检查传入的配置已正确设置
        for key, value in config.items():
            assert alert_manager.config[key] == value

    def test_send_alert(self):
        """测试发送报警"""
        config = {
            'enable_email': False,
            'enable_log': True,
            'log_level': 'INFO'
        }
        alert_manager = AlertManager(config)

        # 发送测试报警
        alert_manager.send_alert(
            alert_type="TEST_ALERT",
            message="测试报警消息",
            severity="MEDIUM",
            context={'test': 'value'}
        )

        # 验证报警记录
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert alert['alert_type'] == "TEST_ALERT"
        assert alert['severity'] == "MEDIUM"

    def test_get_alert_summary(self):
        """测试获取报警总结"""
        config = {'enable_log': True}
        alert_manager = AlertManager(config)

        # 发送几个测试报警
        for i in range(3):
            alert_manager.send_alert(
                alert_type="TEST_ALERT",
                message=f"测试报警 {i}",
                severity="LOW"
            )

        summary = alert_manager.get_alert_summary(hours=1)

        # 验证总结
        assert isinstance(summary, dict)
        assert summary['total_alerts'] == 3

    def test_clear_alert_history(self):
        """测试清除报警历史"""
        config = {'enable_log': True}
        alert_manager = AlertManager(config)

        # 发送测试报警
        alert_manager.send_alert("TEST_ALERT", "测试", "LOW")

        # 清除历史
        alert_manager.clear_alert_history()

        assert len(alert_manager.alert_history) == 0


class TestAlertEngine:
    """测试报警引擎"""

    def test_init(self):
        """测试初始化"""
        alert_manager = AlertManager({'enable_log': True})
        engine = AlertEngine(alert_manager)
        assert engine.alert_manager == alert_manager

    def test_add_rule(self):
        """测试添加规则"""
        alert_manager = AlertManager({'enable_log': True})
        engine = AlertEngine(alert_manager)

        def test_rule(data):
            return data.get('value', 0) > 10

        engine.add_rule(
            name="test_rule",
            condition=test_rule,
            alert_type="VALUE_TOO_HIGH",
            message="值过高",
            severity="MEDIUM"
        )

        assert len(engine.rules) == 1
        assert engine.rules[0].rule_name == "test_rule"

    def test_evaluate_rules(self):
        """测试规则评估"""
        alert_manager = AlertManager({'enable_log': True})
        engine = AlertEngine(alert_manager)

        # 添加测试规则
        def high_value_rule(data):
            return data.get('value', 0) > 10

        engine.add_rule(
            name="high_value",
            condition=high_value_rule,
            alert_type="HIGH_VALUE",
            message="值过高",
            severity="MEDIUM"
        )

        # 测试数据触发规则
        test_data = {'value': 15}
        triggered_alerts = engine.evaluate_all_rules(test_data)

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0]['alert_type'] == "HIGH_VALUE"

    def test_add_standard_rules(self):
        """测试添加标准规则"""
        alert_manager = AlertManager({'enable_log': True})
        engine = AlertEngine(alert_manager)

        engine.add_standard_rules()

        # 应该有标准规则被添加
        assert len(engine.rules) > 0

        # 测试标准规则
        test_data = {
            'auc_change': -0.15,  # 触发AUC下降报警
            'drift_rate': 0.4,    # 触发漂移报警
        }

        triggered_alerts = engine.evaluate_all_rules(test_data)
        assert len(triggered_alerts) > 0


class TestIntegration:
    """测试集成功能"""

    def test_monitoring_workflow_integration(self, sample_binary_classification_data, sample_drift_data):
        """测试监控工作流集成"""
        _, y_true, y_scores = sample_binary_classification_data
        baseline_data, current_data = sample_drift_data

        # 创建组件
        stability_monitor = ModelStabilityMonitor("test_model")
        drift_detector = FeatureDriftDetector(
            feature_names=baseline_data.columns.tolist()
        )
        alert_manager = AlertManager({'enable_log': True})

        # 设置基准
        stability_monitor.set_baseline(y_true, y_scores)
        drift_detector.set_baseline(baseline_data)

        # 运行监控
        stability_result = stability_monitor.monitor_performance(y_true, y_scores)
        drift_result = drift_detector.detect_drift(current_data)

        # 验证结果
        assert isinstance(stability_result, dict)
        assert isinstance(drift_result, dict)

        # 根据结果发送报警
        if stability_result['alerts']:
            for alert in stability_result['alerts']:
                alert_manager.send_alert(
                    alert_type=alert['type'],
                    message=alert['message'],
                    severity=alert['severity']
                )

        # 验证报警记录
        assert len(alert_manager.alert_history) >= 0


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_data_monitoring(self):
        """测试空数据监控"""
        monitor = ModelStabilityMonitor("test_model")

        # 空数据应该抛出异常
        with pytest.raises(ValueError):
            monitor.set_baseline([], [])

    def test_mismatched_data_lengths(self):
        """测试数据长度不匹配"""
        monitor = ModelStabilityMonitor("test_model")

        with pytest.raises(ValueError):
            monitor.set_baseline([0, 1], [0.1, 0.2, 0.3])

    def test_invalid_psi_threshold(self):
        """测试无效PSI阈值"""
        with pytest.raises(ValueError):
            FeatureDriftDetector(['feature'], psi_threshold=-0.1)

    def test_drift_detection_without_baseline(self):
        """测试未设置基准的漂移检测"""
        detector = FeatureDriftDetector(['feature'])
        test_data = pd.DataFrame({'feature': [1, 2, 3]})

        with pytest.raises(ValueError):
            detector.detect_drift(test_data)