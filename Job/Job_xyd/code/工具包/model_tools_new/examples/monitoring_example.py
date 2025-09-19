"""
Model Tools 2.0 监控示例

展示如何使用监控模块进行模型稳定性监控和特征漂移检测
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

# 导入model_tools
import sys
import os
sys.path.append('/Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new')

import model_tools as mt


def generate_sample_data(n_samples=1000, drift_factor=0.0):
    """
    生成示例数据，可以添加漂移

    Parameters:
    -----------
    n_samples : int
        样本数量
    drift_factor : float
        漂移因子，0表示无漂移，>0表示有漂移
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # 添加漂移
    if drift_factor > 0:
        # 对部分特征添加偏移
        X[:, :5] += np.random.normal(0, drift_factor, (n_samples, 5))
        # 改变部分特征的方差
        X[:, 5:8] *= (1 + drift_factor)

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y


def demonstrate_stability_monitoring():
    """演示模型稳定性监控"""
    print("=== 模型稳定性监控演示 ===")

    # 生成基准数据
    print("1. 生成基准数据...")
    X_baseline, y_baseline = generate_sample_data(n_samples=1000, drift_factor=0)

    # 训练模型
    print("2. 训练基准模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_baseline, y_baseline)

    # 获取基准预测
    y_pred_baseline = model.predict_proba(X_baseline)[:, 1]

    # 创建稳定性监控器
    print("3. 创建稳定性监控器...")
    stability_monitor = mt.ModelStabilityMonitor(
        model_name="RandomForest_Demo",
        thresholds={
            'auc_drop_threshold': 0.05,
            'ks_drop_threshold': 0.1,
            'sample_size_threshold': 100
        }
    )

    # 设置基准
    print("4. 设置基准指标...")
    baseline_metrics = stability_monitor.set_baseline(y_baseline, y_pred_baseline)
    print(f"   基准AUC: {baseline_metrics['auc']:.4f}")
    print(f"   基准KS: {baseline_metrics['ks']:.4f}")

    # 模拟多个时期的监控
    print("\n5. 模拟时期监控...")
    for period in range(1, 6):
        print(f"\n   === 第{period}期监控 ===")

        # 生成新数据（逐渐增加漂移）
        drift_factor = period * 0.02  # 逐渐增加漂移
        X_current, y_current = generate_sample_data(
            n_samples=800,
            drift_factor=drift_factor
        )

        # 预测
        y_pred_current = model.predict_proba(X_current)[:, 1]

        # 监控性能
        monitor_result = stability_monitor.monitor_performance(
            y_current,
            y_pred_current,
            timestamp=f"2024-01-{period:02d}T10:00:00"
        )

        # 输出结果
        print(f"   当前AUC: {monitor_result['metrics']['current_auc']:.4f}")
        print(f"   AUC变化: {monitor_result['metrics']['auc_change']:.4f}")
        print(f"   状态: {monitor_result['status']}")

        if monitor_result['alerts']:
            print(f"   ⚠️  触发报警: {len(monitor_result['alerts'])}个")
            for alert in monitor_result['alerts']:
                print(f"      - {alert['type']}: {alert['message']}")
        else:
            print("   ✅ 无报警")

    # 获取监控历史
    print("\n6. 监控历史总结...")
    history = stability_monitor.get_monitoring_history()
    print(f"   总监控次数: {len(history)}")

    return stability_monitor


def demonstrate_drift_detection():
    """演示特征漂移检测"""
    print("\n\n=== 特征漂移检测演示 ===")

    # 生成基准数据
    print("1. 生成基准数据...")
    X_baseline, _ = generate_sample_data(n_samples=1000, drift_factor=0)

    # 创建漂移检测器
    print("2. 创建漂移检测器...")
    drift_detector = mt.FeatureDriftDetector(
        feature_names=X_baseline.columns.tolist(),
        psi_threshold=0.25,
        severe_threshold=0.5
    )

    # 设置基准
    print("3. 设置基准分布...")
    drift_detector.set_baseline(X_baseline)

    # 模拟不同程度的漂移
    print("\n4. 检测不同程度的漂移...")
    drift_levels = [0.0, 0.1, 0.3, 0.5, 0.8]

    for i, drift_factor in enumerate(drift_levels):
        print(f"\n   === 漂移水平 {drift_factor} ===")

        # 生成带漂移的数据
        X_current, _ = generate_sample_data(
            n_samples=800,
            drift_factor=drift_factor
        )

        # 检测漂移
        drift_result = drift_detector.detect_drift(
            X_current,
            timestamp=f"2024-01-{i+1:02d}T12:00:00"
        )

        # 输出结果
        print(f"   总特征数: {drift_result['total_features']}")
        print(f"   漂移特征数: {drift_result['drifted_features']}")
        print(f"   漂移率: {drift_result['drift_rate']:.2%}")
        print(f"   总体状态: {drift_result['overall_status']}")

        if drift_result['drift_alerts']:
            print(f"   ⚠️  漂移报警: {len(drift_result['drift_alerts'])}个")
            # 显示前3个最严重的漂移
            for alert in drift_result['drift_alerts'][:3]:
                print(f"      - {alert['feature']}: PSI={alert['psi_value']:.3f} ({alert['severity']})")
        else:
            print("   ✅ 无漂移报警")

    # 获取漂移总结
    print("\n5. 漂移检测总结...")
    drift_summary = drift_detector.get_drift_summary()
    print(f"   总检测次数: {drift_summary['total_detections']}")

    return drift_detector


def demonstrate_alerting_system():
    """演示报警系统"""
    print("\n\n=== 报警系统演示 ===")

    # 创建报警配置
    print("1. 配置报警系统...")
    alert_config = {
        'enable_email': False,      # 演示中不发送邮件
        'enable_log': True,         # 启用日志报警
        'enable_webhook': False,    # 演示中不发送webhook
        'log_level': 'INFO'
    }

    # 创建报警管理器
    alert_manager = mt.AlertManager(alert_config)

    # 创建报警引擎
    alert_engine = mt.AlertEngine(alert_manager)

    # 添加标准规则
    print("2. 添加标准报警规则...")
    alert_engine.add_standard_rules()
    print(f"   已添加 {len(alert_engine.rules)} 个标准规则")

    # 添加自定义规则
    print("3. 添加自定义报警规则...")
    def high_drift_rate_rule(data):
        return data.get('drift_rate', 0) > 0.4

    alert_engine.add_rule(
        name="high_drift_rate",
        condition=high_drift_rate_rule,
        alert_type="HIGH_DRIFT_RATE",
        message="特征漂移率过高，建议检查数据质量",
        severity="HIGH"
    )

    # 模拟触发报警的数据
    print("\n4. 模拟报警触发...")
    test_scenarios = [
        {
            'name': '正常情况',
            'data': {
                'auc_change': -0.02,
                'ks_change': -0.05,
                'drift_rate': 0.15,
                'sample_size': 1000
            }
        },
        {
            'name': '性能下降',
            'data': {
                'auc_change': -0.12,  # 触发AUC下降报警
                'ks_change': -0.08,
                'drift_rate': 0.20,
                'sample_size': 1000
            }
        },
        {
            'name': '严重漂移',
            'data': {
                'auc_change': -0.05,
                'ks_change': -0.03,
                'drift_rate': 0.45,   # 触发高漂移率报警
                'sample_size': 1000
            }
        },
        {
            'name': '样本不足',
            'data': {
                'auc_change': -0.03,
                'ks_change': -0.02,
                'drift_rate': 0.10,
                'sample_size': 50     # 触发样本量不足报警
            }
        }
    ]

    for scenario in test_scenarios:
        print(f"\n   --- {scenario['name']} ---")
        triggered_alerts = alert_engine.evaluate_all_rules(scenario['data'])

        if triggered_alerts:
            print(f"   触发报警: {len(triggered_alerts)}个")
            for alert in triggered_alerts:
                print(f"   🚨 {alert['alert_type']}: {alert['message']} [{alert['severity']}]")
        else:
            print("   ✅ 无报警触发")

    # 获取报警汇总
    print("\n5. 报警汇总...")
    alert_summary = alert_manager.get_alert_summary(hours=1)
    print(f"   总报警数: {alert_summary['total_alerts']}")
    print(f"   报警类型分布: {alert_summary['alert_distribution']['by_type']}")
    print(f"   严重程度分布: {alert_summary['alert_distribution']['by_severity']}")

    return alert_manager, alert_engine


def demonstrate_monitoring_workflow():
    """演示监控工作流"""
    print("\n\n=== 监控工作流演示 ===")

    # 生成基准数据
    print("1. 准备基准数据...")
    X_baseline, y_baseline = generate_sample_data(n_samples=1000, drift_factor=0)

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_baseline, y_baseline)
    y_pred_baseline = model.predict_proba(X_baseline)[:, 1]

    # 创建监控组件
    print("2. 创建监控组件...")
    stability_monitor = mt.ModelStabilityMonitor("WorkflowDemo")
    drift_detector = mt.FeatureDriftDetector(
        feature_names=X_baseline.columns.tolist()
    )
    alert_manager = mt.AlertManager({'enable_log': True})

    # 创建监控工作流
    print("3. 创建监控工作流...")
    workflow = mt.MonitoringWorkflow(
        model_name="WorkflowDemo",
        stability_monitor=stability_monitor,
        drift_detector=drift_detector,
        alert_manager=alert_manager
    )

    # 设置基准
    print("4. 设置工作流基准...")
    setup_result = workflow.setup_baseline(
        X_baseline, y_baseline, y_pred_baseline
    )
    print(f"   基准设置状态: {setup_result['status']}")

    # 运行监控周期
    print("\n5. 运行监控周期...")
    for cycle in range(1, 4):
        print(f"\n   === 监控周期 {cycle} ===")

        # 生成当前数据
        drift_factor = cycle * 0.1
        X_current, y_current = generate_sample_data(
            n_samples=800,
            drift_factor=drift_factor
        )
        y_pred_current = model.predict_proba(X_current)[:, 1]

        # 运行监控周期
        cycle_result = workflow.run_monitoring_cycle(
            X_current, y_current, y_pred_current,
            timestamp=f"2024-01-{cycle:02d}T14:00:00"
        )

        print(f"   数据量: {cycle_result['data_size']}")
        print(f"   总体状态: {cycle_result['overall_status']}")
        print(f"   报警数量: {len(cycle_result['alerts'])}")

        if cycle_result['alerts']:
            for alert in cycle_result['alerts'][:2]:  # 只显示前2个
                print(f"   🚨 {alert['type']}: {alert['message']}")

    # 获取监控总结
    print("\n6. 获取监控总结...")
    summary = workflow.get_monitoring_summary(hours=24)
    print(f"   监控周期数: {summary['total_monitoring_cycles']}")
    print(f"   总报警数: {summary['total_alerts']}")

    return workflow


def main():
    """主演示函数"""
    print("Model Tools 2.0 监控功能演示")
    print("=" * 50)

    try:
        # 1. 模型稳定性监控
        stability_monitor = demonstrate_stability_monitoring()

        # 2. 特征漂移检测
        drift_detector = demonstrate_drift_detection()

        # 3. 报警系统
        alert_manager, alert_engine = demonstrate_alerting_system()

        # 4. 监控工作流
        workflow = demonstrate_monitoring_workflow()

        print("\n" + "=" * 50)
        print("✅ 监控功能演示完成！")
        print("\n主要功能演示:")
        print("• 模型稳定性监控 - 检测AUC、KS等指标变化")
        print("• 特征漂移检测 - 基于PSI的特征分布变化检测")
        print("• 自动化报警系统 - 规则引擎和多渠道报警")
        print("• 监控工作流 - 集成的端到端监控流程")
        print("\n💡 提示: 在生产环境中，可以:")
        print("1. 设置更严格的阈值")
        print("2. 启用邮件和webhook报警")
        print("3. 使用自动化数据源")
        print("4. 定期生成监控报告")

        return {
            'stability_monitor': stability_monitor,
            'drift_detector': drift_detector,
            'alert_manager': alert_manager,
            'workflow': workflow
        }

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()