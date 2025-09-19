#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型监控系统简单演示

演示PSI计算和模型稳定性监控的基本功能
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def demo_psi_calculation():
    """演示PSI计算"""
    print("=== PSI计算演示 ===")

    # 创建示例数据
    np.random.seed(42)

    # 基准数据（训练时）
    baseline_data = np.random.normal(0, 1, 1000)
    print(f"基准数据统计: 均值={baseline_data.mean():.3f}, 标准差={baseline_data.std():.3f}")

    # 当前数据（有轻微漂移）
    current_data = np.random.normal(0.3, 1.2, 800)
    print(f"当前数据统计: 均值={current_data.mean():.3f}, 标准差={current_data.std():.3f}")

    # 简化的PSI计算
    def simple_psi(base, test, bins=10):
        """简化的PSI计算函数"""
        # 基于基准数据确定分箱边界
        _, bin_edges = np.histogram(base, bins=bins)

        # 计算各分箱的频率
        base_counts, _ = np.histogram(base, bins=bin_edges)
        test_counts, _ = np.histogram(test, bins=bin_edges)

        # 计算分布比例
        base_dist = base_counts / len(base)
        test_dist = test_counts / len(test)

        # 避免除零
        base_dist = np.where(base_dist == 0, 1e-6, base_dist)
        test_dist = np.where(test_dist == 0, 1e-6, test_dist)

        # 计算PSI
        psi_components = (test_dist - base_dist) * np.log(test_dist / base_dist)
        psi = np.sum(psi_components)

        return psi

    psi_value = simple_psi(baseline_data, current_data)
    print(f"PSI值: {psi_value:.4f}")

    # PSI解释
    if psi_value < 0.1:
        interpretation = "稳定 (变化很小)"
    elif psi_value < 0.2:
        interpretation = "轻微变化"
    elif psi_value < 0.25:
        interpretation = "中等变化 (需要关注)"
    else:
        interpretation = "显著变化 (需要重新建模)"

    print(f"PSI解释: {interpretation}")
    return psi_value


def demo_performance_monitoring():
    """演示模型性能监控"""
    print("\n=== 模型性能监控演示 ===")

    # 创建示例数据
    np.random.seed(42)

    # 基准期数据
    n_baseline = 1000
    baseline_features = np.random.normal(0, 1, (n_baseline, 3))
    baseline_prob = 1 / (1 + np.exp(-(
        0.5 * baseline_features[:, 0] +
        0.3 * baseline_features[:, 1] -
        0.2 * baseline_features[:, 2]
    )))
    baseline_y_true = np.random.binomial(1, baseline_prob, n_baseline)
    baseline_y_pred = baseline_prob + np.random.normal(0, 0.05, n_baseline)
    baseline_y_pred = np.clip(baseline_y_pred, 0, 1)

    # 计算基准AUC和KS
    def calculate_auc(y_true, y_pred):
        """计算AUC"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)

    def calculate_ks(y_true, y_pred):
        """计算KS值"""
        # 按预测概率排序
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df = df.sort_values('y_pred', ascending=False)

        # 计算累积比例
        df['bad'] = df['y_true']
        df['good'] = 1 - df['y_true']

        total_good = df['good'].sum()
        total_bad = df['bad'].sum()

        df['cum_good_rate'] = df['good'].cumsum() / total_good
        df['cum_bad_rate'] = df['bad'].cumsum() / total_bad

        # KS值
        ks = (df['cum_bad_rate'] - df['cum_good_rate']).max()
        return ks

    try:
        baseline_auc = calculate_auc(baseline_y_true, baseline_y_pred)
        baseline_ks = calculate_ks(baseline_y_true, baseline_y_pred)
        print(f"基准AUC: {baseline_auc:.4f}")
        print(f"基准KS: {baseline_ks:.4f}")
    except ImportError:
        print("scikit-learn未安装，使用模拟指标")
        baseline_auc = 0.75
        baseline_ks = 0.35
        print(f"基准AUC: {baseline_auc:.4f} (模拟)")
        print(f"基准KS: {baseline_ks:.4f} (模拟)")

    # 当前期数据（性能下降）
    n_current = 800
    current_features = np.random.normal(0.3, 1.2, (n_current, 3))
    current_prob = 1 / (1 + np.exp(-(
        0.4 * current_features[:, 0] +  # 系数变化
        0.2 * current_features[:, 1] -
        0.1 * current_features[:, 2] +
        np.random.normal(0, 0.2, n_current)  # 增加噪音
    )))
    current_y_true = np.random.binomial(1, current_prob, n_current)
    current_y_pred = current_prob + np.random.normal(0, 0.1, n_current)
    current_y_pred = np.clip(current_y_pred, 0, 1)

    try:
        current_auc = calculate_auc(current_y_true, current_y_pred)
        current_ks = calculate_ks(current_y_true, current_y_pred)
        print(f"当前AUC: {current_auc:.4f}")
        print(f"当前KS: {current_ks:.4f}")
    except ImportError:
        current_auc = 0.68
        current_ks = 0.28
        print(f"当前AUC: {current_auc:.4f} (模拟)")
        print(f"当前KS: {current_ks:.4f} (模拟)")

    # 计算变化
    auc_change = current_auc - baseline_auc
    ks_change = current_ks - baseline_ks

    print(f"AUC变化: {auc_change:.4f}")
    print(f"KS变化: {ks_change:.4f}")

    # 报警判断
    alerts = []
    if auc_change < -0.05:
        severity = 'HIGH' if abs(auc_change) > 0.1 else 'MEDIUM'
        alerts.append(f"[{severity}] AUC显著下降: {abs(auc_change):.4f}")

    if ks_change < -0.1:
        severity = 'HIGH' if abs(ks_change) > 0.2 else 'MEDIUM'
        alerts.append(f"[{severity}] KS显著下降: {abs(ks_change):.4f}")

    if alerts:
        print("⚠️  检测到以下报警:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("✅ 模型性能稳定")

    return len(alerts)


def demo_feature_stability():
    """演示特征稳定性监控"""
    print("\n=== 特征稳定性监控演示 ===")

    np.random.seed(42)

    # 基准特征
    baseline_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.uniform(-2, 2, 1000),
        'feature3': np.random.exponential(1, 1000)
    })

    # 当前特征（有不同程度的漂移）
    current_features = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1.05, 800),  # 轻微漂移
        'feature2': np.random.uniform(-1.8, 2.3, 800), # 轻微漂移
        'feature3': np.random.exponential(1.8, 800)     # 显著漂移
    })

    print("特征统计对比:")
    print("基准特征:")
    print(baseline_features.describe().round(3))
    print("\n当前特征:")
    print(current_features.describe().round(3))

    # 简化的PSI计算
    def simple_psi(base, test, bins=10):
        """简化的PSI计算"""
        _, bin_edges = np.histogram(base, bins=bins)
        base_counts, _ = np.histogram(base, bins=bin_edges)
        test_counts, _ = np.histogram(test, bins=bin_edges)

        base_dist = base_counts / len(base)
        test_dist = test_counts / len(test)

        base_dist = np.where(base_dist == 0, 1e-6, base_dist)
        test_dist = np.where(test_dist == 0, 1e-6, test_dist)

        psi = np.sum((test_dist - base_dist) * np.log(test_dist / base_dist))
        return psi

    # 计算各特征PSI
    feature_psi = {}
    unstable_features = []

    for feature in baseline_features.columns:
        psi = simple_psi(baseline_features[feature], current_features[feature])
        feature_psi[feature] = psi

        if psi >= 0.25:
            unstable_features.append(feature)

    print(f"\n特征PSI结果:")
    for feature, psi in feature_psi.items():
        status = "❌ 不稳定" if psi >= 0.25 else "✅ 稳定"
        print(f"{feature}: {psi:.4f} {status}")

    stability_rate = (len(feature_psi) - len(unstable_features)) / len(feature_psi)
    print(f"\n特征稳定率: {stability_rate:.2%}")

    if unstable_features:
        print(f"⚠️  不稳定特征: {', '.join(unstable_features)}")
        return len(unstable_features)
    else:
        print("✅ 所有特征都稳定")
        return 0


def main():
    """主函数"""
    print("模型监控系统演示")
    print("=" * 50)

    # 运行各个演示
    psi_value = demo_psi_calculation()
    alert_count = demo_performance_monitoring()
    unstable_feature_count = demo_feature_stability()

    # 总结
    print("\n" + "=" * 50)
    print("监控总结")
    print("=" * 50)
    print(f"PSI值: {psi_value:.4f}")
    print(f"性能报警数: {alert_count}")
    print(f"不稳定特征数: {unstable_feature_count}")

    if psi_value < 0.25 and alert_count == 0 and unstable_feature_count == 0:
        print("🎉 模型整体稳定，可以继续使用")
    elif psi_value < 0.5 and alert_count <= 1 and unstable_feature_count <= 2:
        print("⚠️  模型有轻微波动，建议密切监控")
    else:
        print("🚨 模型存在显著问题，建议重新训练")

    print("\n监控系统功能验证完成！")
    print("主要功能包括:")
    print("1. ✅ PSI计算和特征稳定性监控")
    print("2. ✅ 模型性能监控（AUC/KS）")
    print("3. ✅ 自动报警机制")
    print("4. ✅ 综合稳定性评估")


if __name__ == "__main__":
    main()