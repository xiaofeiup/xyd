"""
Model Tools 基础使用示例

展示如何使用重构后的模型工具包进行特征选择、模型评估和监控
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 导入重构后的模型工具
import sys
import os
sys.path.append('/Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new')

import model_tools as mt

def create_sample_data():
    """创建示例数据"""
    print("=== 创建示例数据 ===")

    # 生成分类数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )

    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # 添加一些无用特征（单一值占比高）
    df['useless_feature_1'] = 1  # 完全单一值
    df['useless_feature_2'] = np.random.choice([0, 1], size=len(df), p=[0.99, 0.01])  # 99%为0

    # 添加一些缺失值
    df.loc[df.sample(frac=0.1).index, 'feature_0'] = np.nan

    print(f"数据形状: {df.shape}")
    print(f"目标变量分布: {df['target'].value_counts().to_dict()}")

    return df

def demonstrate_data_quality_check(df):
    """演示数据质量检查"""
    print("\n=== 数据质量检查 ===")

    # 使用数据质量检查工具
    quality_report = mt.check_data_quality(df, target_col='target')

    print(f"基础信息: {quality_report['basic_info']}")
    print(f"缺失值分析: 有{quality_report['missing_analysis']['columns_with_missing']}列存在缺失值")
    print(f"目标变量分析: {quality_report['target_analysis']['class_balance']}")

    return quality_report

def demonstrate_feature_selection(df):
    """演示特征选择"""
    print("\n=== 特征选择演示 ===")

    # 分离特征和目标变量
    X, y = df.drop('target', axis=1), df['target']

    # 创建特征选择器
    selector = mt.FeatureSelector(
        method='iv',
        single_value_threshold=0.95,  # 单一值占比阈值
        iv_threshold=0.1,             # IV阈值
        k_features=10                 # 最终选择10个特征
    )

    # 拟合并选择特征
    X_selected = selector.fit_transform(X, y, target_col='target')

    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择后特征数量: {X_selected.shape[1]}")

    # 查看选择总结
    summary = selector.get_selection_summary()
    print(f"选择总结: {summary}")

    # 查看IV筛选日志
    iv_log = selector.get_iv_log()
    print(f"\nIV值最高的5个特征:")
    print(iv_log.head()[['feature', 'iv_value', 'interpretation', 'keep_feature']])

    return X_selected, selector

def demonstrate_feature_importance(X, y):
    """演示特征重要性分析"""
    print("\n=== 特征重要性分析 ===")

    # 创建特征重要性分析器
    importance_analyzer = mt.FeatureImportanceAnalyzer(random_state=42)

    # 计算多种重要性
    methods = ['random_forest', 'iv', 'correlation']
    all_importance = importance_analyzer.calculate_all_importance(X, y, methods)

    # 获取综合排名
    consensus_ranking = importance_analyzer.get_consensus_ranking(methods, top_k=10)
    print(f"\n综合重要性排名 (前10):")
    print(consensus_ranking[['feature', 'avg_rank', 'avg_score']].head(10))

    # 获取稳定的重要特征
    stable_features = importance_analyzer.get_stable_features(methods, top_k_per_method=15, min_appearances=2)
    print(f"\n稳定的重要特征 (在至少2个方法中都排前15): {stable_features}")

    return importance_analyzer

def demonstrate_model_evaluation(X_train, X_test, y_train, y_test):
    """演示模型评估"""
    print("\n=== 模型评估演示 ===")

    # 训练一个简单的模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 使用模型评估器
    evaluator = mt.ModelEvaluator(model_name="RandomForest")
    evaluation_result = evaluator.evaluate_binary_classification(y_test, y_pred_proba)

    print(f"基础指标: {evaluation_result['basic_metrics']}")
    print(f"Lift指标: {evaluation_result['lift_metrics']}")
    print(f"样本信息: {evaluation_result['sample_info']}")

    return evaluation_result, y_pred_proba

def demonstrate_monitoring(X_train, X_test, y_train, y_test, y_pred_proba):
    """演示模型监控"""
    print("\n=== 模型监控演示 ===")

    # 1. 模型稳定性监控
    print("\n--- 模型稳定性监控 ---")
    stability_monitor = mt.ModelStabilityMonitor(
        model_name="RandomForest",
        thresholds={'auc_drop_threshold': 0.05, 'ks_drop_threshold': 0.1}
    )

    # 设置基准
    baseline_metrics = stability_monitor.set_baseline(y_train,
                                                     RandomForestClassifier(n_estimators=100, random_state=42)
                                                     .fit(X_train, y_train)
                                                     .predict_proba(X_train)[:, 1])

    # 监控测试集性能
    monitor_result = stability_monitor.monitor_performance(y_test, y_pred_proba)
    print(f"监控结果: {monitor_result['metrics']}")
    if monitor_result['alerts']:
        print(f"触发的报警: {monitor_result['alerts']}")

    # 2. 特征漂移检测
    print("\n--- 特征漂移检测 ---")
    drift_detector = mt.FeatureDriftDetector(
        feature_names=X_train.columns.tolist(),
        psi_threshold=0.25
    )

    # 设置基准
    drift_detector.set_baseline(X_train)

    # 检测漂移
    drift_result = drift_detector.detect_drift(X_test)
    print(f"漂移检测结果: 总特征{drift_result['total_features']}, 漂移特征{drift_result['drifted_features']}")
    print(f"漂移率: {drift_result['drift_rate']:.2%}")
    print(f"总体状态: {drift_result['overall_status']}")

    if drift_result['drift_alerts']:
        print("漂移报警:")
        for alert in drift_result['drift_alerts']:
            print(f"  - {alert['feature']}: PSI={alert['psi_value']:.3f}, {alert['interpretation']}")

    return stability_monitor, drift_detector

def demonstrate_alerting():
    """演示报警系统"""
    print("\n=== 报警系统演示 ===")

    # 创建报警管理器
    alert_config = {
        'enable_email': False,
        'enable_log': True,
        'enable_webhook': False,
        'log_level': 'INFO'
    }

    alert_manager = mt.AlertManager(alert_config)

    # 发送报警
    alert_manager.send_alert(
        alert_type="PERFORMANCE_DEGRADATION",
        message="模型AUC显著下降",
        severity="HIGH",
        context={'auc_change': -0.08, 'model': 'RandomForest'}
    )

    # 创建报警引擎
    alert_engine = mt.monitoring.AlertEngine(alert_manager)

    # 添加标准规则
    alert_engine.add_standard_rules()

    # 模拟数据触发规则
    test_data = {
        'auc_change': -0.12,  # 触发AUC下降报警
        'ks_change': -0.25,   # 触发KS下降报警
        'drift_rate': 0.35,   # 触发特征漂移报警
        'sample_size': 500    # 触发样本量不足报警
    }

    triggered_alerts = alert_engine.evaluate_all_rules(test_data)
    print(f"触发的报警数量: {len(triggered_alerts)}")

    # 获取报警汇总
    alert_summary = alert_manager.get_alert_summary(hours=1)
    print(f"报警汇总: {alert_summary}")

    return alert_manager, alert_engine

def main():
    """主函数"""
    print("Model Tools 2.0 使用示例")
    print("=" * 50)

    # 1. 创建示例数据
    df = create_sample_data()

    # 2. 数据质量检查
    quality_report = demonstrate_data_quality_check(df)

    # 3. 特征选择
    X_selected, selector = demonstrate_feature_selection(df)
    X, y = df.drop('target', axis=1), df['target']

    # 4. 特征重要性分析
    importance_analyzer = demonstrate_feature_importance(X_selected, y)

    # 5. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )

    # 6. 模型评估
    evaluation_result, y_pred_proba = demonstrate_model_evaluation(X_train, X_test, y_train, y_test)

    # 7. 模型监控
    stability_monitor, drift_detector = demonstrate_monitoring(X_train, X_test, y_train, y_test, y_pred_proba)

    # 8. 报警系统
    alert_manager, alert_engine = demonstrate_alerting()

    print("\n" + "=" * 50)
    print("示例演示完成！")
    print("\n主要功能演示:")
    print("✓ 数据质量检查")
    print("✓ 特征选择（单一值筛选 + IV筛选）")
    print("✓ 特征重要性分析")
    print("✓ 模型评估（AUC、KS、Lift等）")
    print("✓ 模型稳定性监控")
    print("✓ 特征漂移检测")
    print("✓ 报警系统")

    return {
        'data': df,
        'selector': selector,
        'importance_analyzer': importance_analyzer,
        'evaluation_result': evaluation_result,
        'stability_monitor': stability_monitor,
        'drift_detector': drift_detector,
        'alert_manager': alert_manager
    }

if __name__ == "__main__":
    results = main()