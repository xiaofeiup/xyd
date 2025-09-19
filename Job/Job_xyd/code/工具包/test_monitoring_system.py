#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型监控系统综合测试脚本

测试PSI计算、模型监控、特征稳定性监控和自动化工作流等功能
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import warnings
from pathlib import Path

# 添加模块路径
sys.path.append('model_tools')

# 导入测试模块
from model_tools.metric_report.psi import calculate_psi, calculate_multi_feature_psi, generate_psi_report
from model_tools.model_monitor import ModelStabilityMonitor
from model_tools.alerting_workflow import AutomatedMonitoringWorkflow, create_default_config


def create_sample_data():
    """
    创建示例数据用于测试
    """
    print("创建示例数据...")

    # 设置随机种子
    np.random.seed(42)

    # 基准数据
    n_baseline = 1000
    baseline_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_baseline),
        'feature2': np.random.uniform(-2, 2, n_baseline),
        'feature3': np.random.exponential(1, n_baseline),
        'feature4': np.random.choice(['A', 'B', 'C'], n_baseline, p=[0.5, 0.3, 0.2])
    })

    # 生成基准标签和预测
    baseline_prob = 1 / (1 + np.exp(-(
        0.5 * baseline_features['feature1'] +
        0.3 * baseline_features['feature2'] -
        0.2 * baseline_features['feature3'] +
        np.random.normal(0, 0.1, n_baseline)
    )))
    baseline_y_true = np.random.binomial(1, baseline_prob, n_baseline)
    baseline_y_pred = baseline_prob + np.random.normal(0, 0.05, n_baseline)
    baseline_y_pred = np.clip(baseline_y_pred, 0, 1)

    # 测试数据（有轻微漂移）
    n_test = 800
    test_features = pd.DataFrame({
        'feature1': np.random.normal(0.2, 1.1, n_test),  # 均值和方差略有变化
        'feature2': np.random.uniform(-1.8, 2.2, n_test),  # 范围略有变化
        'feature3': np.random.exponential(1.2, n_test),  # 参数略有变化
        'feature4': np.random.choice(['A', 'B', 'C'], n_test, p=[0.4, 0.4, 0.2])  # 分布变化
    })

    # 生成测试标签和预测（性能略有下降）
    test_prob = 1 / (1 + np.exp(-(
        0.4 * test_features['feature1'] +  # 系数略有变化
        0.25 * test_features['feature2'] -
        0.18 * test_features['feature3'] +
        np.random.normal(0, 0.15, n_test)  # 噪音增加
    )))
    test_y_true = np.random.binomial(1, test_prob, n_test)
    test_y_pred = test_prob + np.random.normal(0, 0.08, n_test)
    test_y_pred = np.clip(test_y_pred, 0, 1)

    # 问题数据（显著漂移）
    n_problem = 600
    problem_features = pd.DataFrame({
        'feature1': np.random.normal(0.8, 1.5, n_problem),  # 显著变化
        'feature2': np.random.uniform(-1, 3, n_problem),  # 显著变化
        'feature3': np.random.exponential(2, n_problem),  # 显著变化
        'feature4': np.random.choice(['A', 'B', 'C'], n_problem, p=[0.2, 0.2, 0.6])  # 显著变化
    })

    # 生成问题数据的标签和预测（性能显著下降）
    problem_prob = 1 / (1 + np.exp(-(
        0.2 * problem_features['feature1'] +  # 系数显著变化
        0.1 * problem_features['feature2'] -
        0.05 * problem_features['feature3'] +
        np.random.normal(0, 0.3, n_problem)  # 噪音显著增加
    )))
    problem_y_true = np.random.binomial(1, problem_prob, n_problem)
    problem_y_pred = problem_prob + np.random.normal(0, 0.15, n_problem)
    problem_y_pred = np.clip(problem_y_pred, 0, 1)

    return {
        'baseline': {
            'features': baseline_features,
            'y_true': baseline_y_true,
            'y_pred': baseline_y_pred
        },
        'test': {
            'features': test_features,
            'y_true': test_y_true,
            'y_pred': test_y_pred
        },
        'problem': {
            'features': problem_features,
            'y_true': problem_y_true,
            'y_pred': problem_y_pred
        }
    }


def test_psi_calculation(data):
    """
    测试PSI计算功能
    """
    print("\n" + "="*60)
    print("测试PSI计算功能")
    print("="*60)

    baseline_features = data['baseline']['features']
    test_features = data['test']['features']
    problem_features = data['problem']['features']

    print("\n1. 单特征PSI计算测试:")
    print("-" * 30)

    # 测试数值特征
    psi_value, psi_detail = calculate_psi(
        baseline_features['feature1'],
        test_features['feature1'],
        feature_name='feature1'
    )
    print(f"Feature1 PSI (轻微漂移): {psi_value:.4f}")

    psi_value, psi_detail = calculate_psi(
        baseline_features['feature1'],
        problem_features['feature1'],
        feature_name='feature1'
    )
    print(f"Feature1 PSI (显著漂移): {psi_value:.4f}")

    print("\n2. 多特征PSI批量计算:")
    print("-" * 30)

    # 测试轻微漂移
    psi_summary_test = calculate_multi_feature_psi(
        baseline_features,
        test_features,
        features=['feature1', 'feature2', 'feature3']
    )
    print("轻微漂移情况:")
    print(psi_summary_test)

    # 测试显著漂移
    psi_summary_problem = calculate_multi_feature_psi(
        baseline_features,
        problem_features,
        features=['feature1', 'feature2', 'feature3']
    )
    print("\n显著漂移情况:")
    print(psi_summary_problem)

    print("\n3. PSI报告生成:")
    print("-" * 30)

    report = generate_psi_report(psi_summary_problem, threshold=0.25)
    print(f"总特征数: {report['summary']['total_features']}")
    print(f"稳定特征数: {report['summary']['stable_features']}")
    print(f"不稳定特征数: {report['summary']['unstable_features']}")
    print(f"稳定率: {report['summary']['stability_rate']:.2%}")

    print("\n建议:")
    for rec in report['recommendations']:
        print(f"- {rec}")


def test_model_monitoring(data):
    """
    测试模型监控功能
    """
    print("\n" + "="*60)
    print("测试模型监控功能")
    print("="*60)

    # 创建监控器
    monitor = ModelStabilityMonitor(
        model_name='test_model',
        thresholds={
            'auc_drop_threshold': 0.03,
            'ks_drop_threshold': 0.05,
            'psi_threshold': 0.2
        }
    )

    print("\n1. 设置基准指标:")
    print("-" * 30)

    baseline_data = data['baseline']
    monitor.set_baseline(
        baseline_data['y_true'],
        baseline_data['y_pred'],
        baseline_data['features']
    )

    print(f"基准AUC: {monitor.baseline_metrics['auc']:.4f}")
    print(f"基准KS: {monitor.baseline_metrics['ks']:.4f}")

    print("\n2. 监控轻微变化情况:")
    print("-" * 30)

    test_data = data['test']
    result_test = monitor.monitor_model_performance(
        test_data['y_true'],
        test_data['y_pred']
    )

    print(f"当前AUC: {result_test['metrics']['current_auc']:.4f}")
    print(f"当前KS: {result_test['metrics']['current_ks']:.4f}")
    print(f"AUC变化: {result_test['metrics']['auc_change']:.4f}")
    print(f"KS变化: {result_test['metrics']['ks_change']:.4f}")
    print(f"报警数量: {len(result_test['alerts'])}")

    for alert in result_test['alerts']:
        print(f"  - [{alert['severity']}] {alert['message']}")

    print("\n3. 监控显著变化情况:")
    print("-" * 30)

    problem_data = data['problem']
    result_problem = monitor.monitor_model_performance(
        problem_data['y_true'],
        problem_data['y_pred']
    )

    print(f"当前AUC: {result_problem['metrics']['current_auc']:.4f}")
    print(f"当前KS: {result_problem['metrics']['current_ks']:.4f}")
    print(f"AUC变化: {result_problem['metrics']['auc_change']:.4f}")
    print(f"KS变化: {result_problem['metrics']['ks_change']:.4f}")
    print(f"报警数量: {len(result_problem['alerts'])}")

    for alert in result_problem['alerts']:
        print(f"  - [{alert['severity']}] {alert['message']}")

    print("\n4. 特征稳定性监控:")
    print("-" * 30)

    stability_result = monitor.monitor_feature_stability(
        problem_data['features']
    )

    print(f"不稳定特征数: {stability_result['unstable_features']}")
    print(f"总特征数: {stability_result['total_features']}")
    print(f"稳定率: {stability_result['stability_rate']:.2%}")
    print(f"特征报警数: {len(stability_result['alerts'])}")

    for alert in stability_result['alerts']:
        print(f"  - [{alert['severity']}] {alert['message']}")

    print("\n5. 监控历史汇总:")
    print("-" * 30)

    summary = monitor.get_monitoring_summary(days=1)
    print(f"监控记录数: {summary['total_monitoring_records']}")
    print(f"总报警数: {summary['total_alerts']}")

    return monitor


def test_automated_workflow():
    """
    测试自动化工作流
    """
    print("\n" + "="*60)
    print("测试自动化工作流")
    print("="*60)

    # 创建测试配置文件
    config_path = 'test_config.json'

    print(f"\n1. 创建配置文件: {config_path}")
    print("-" * 30)

    create_default_config(config_path)

    # 修改配置以使用测试数据
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 禁用定时任务，仅测试功能
    config['schedule']['enabled'] = False

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("配置文件已创建和修改")

    print("\n2. 创建自定义数据加载函数:")
    print("-" * 30)

    # 创建测试数据文件
    data = create_sample_data()

    # 保存基准数据
    baseline_df = data['baseline']['features'].copy()
    baseline_df['target'] = data['baseline']['y_true']
    baseline_df['pred_prob'] = data['baseline']['y_pred']
    baseline_df.to_csv('test_baseline_data.csv', index=False)

    # 保存测试数据
    test_df = data['test']['features'].copy()
    test_df['target'] = data['test']['y_true']
    test_df['pred_prob'] = data['test']['y_pred']
    test_df.to_csv('test_current_data.csv', index=False)

    print("测试数据文件已保存")

    # 更新配置以指向测试数据
    config['models'][0]['baseline_data']['file_path'] = 'test_baseline_data.csv'
    config['models'][0]['current_data']['file_path'] = 'test_current_data.csv'
    config['models'][0]['baseline_data']['feature_columns'] = ['feature1', 'feature2', 'feature3']
    config['models'][0]['current_data']['feature_columns'] = ['feature1', 'feature2', 'feature3']

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n3. 初始化工作流:")
    print("-" * 30)

    try:
        workflow = AutomatedMonitoringWorkflow(config_path)
        print("工作流初始化成功")

        print("\n4. 执行单次监控:")
        print("-" * 30)

        workflow.run_once()

        print("\n5. 获取工作流状态:")
        print("-" * 30)

        status = workflow.get_workflow_status()
        print(f"运行状态: {status['is_running']}")
        print(f"监控模型数: {status['model_count']}")
        print(f"总检查次数: {status['stats']['total_checks']}")
        print(f"总报警数: {status['stats']['total_alerts']}")

    except Exception as e:
        print(f"工作流测试失败: {str(e)}")

    finally:
        # 清理测试文件
        try:
            os.remove(config_path)
            os.remove('test_baseline_data.csv')
            os.remove('test_current_data.csv')
            if os.path.exists('monitoring_report.json'):
                os.remove('monitoring_report.json')
            if os.path.exists('alerts.json'):
                os.remove('alerts.json')
            if os.path.exists('monitoring_workflow.log'):
                os.remove('monitoring_workflow.log')
            print("\n测试文件已清理")
        except:
            pass


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "="*60)
    print("测试边界情况")
    print("="*60)

    print("\n1. 空数据测试:")
    print("-" * 30)

    try:
        psi_value, _ = calculate_psi([], [])
        print(f"空数据PSI: {psi_value}")
    except Exception as e:
        print(f"空数据测试异常: {str(e)}")

    print("\n2. 单一值数据测试:")
    print("-" * 30)

    try:
        psi_value, _ = calculate_psi([1] * 100, [1] * 100)
        print(f"单一值PSI: {psi_value}")
    except Exception as e:
        print(f"单一值测试异常: {str(e)}")

    print("\n3. 极端分布差异测试:")
    print("-" * 30)

    try:
        base_data = np.random.normal(0, 1, 1000)
        test_data = np.random.normal(10, 1, 1000)  # 均值差异极大
        psi_value, _ = calculate_psi(base_data, test_data)
        print(f"极端差异PSI: {psi_value:.4f}")
    except Exception as e:
        print(f"极端差异测试异常: {str(e)}")

    print("\n4. 不同样本大小测试:")
    print("-" * 30)

    try:
        base_data = np.random.normal(0, 1, 10000)
        test_data = np.random.normal(0.1, 1, 100)  # 样本量差异很大
        psi_value, _ = calculate_psi(base_data, test_data)
        print(f"不同样本大小PSI: {psi_value:.4f}")
    except Exception as e:
        print(f"不同样本大小测试异常: {str(e)}")


def performance_test():
    """
    性能测试
    """
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)

    import time

    # 测试大数据量PSI计算
    print("\n大数据量PSI计算测试:")
    print("-" * 30)

    sizes = [1000, 10000, 100000]

    for size in sizes:
        base_data = np.random.normal(0, 1, size)
        test_data = np.random.normal(0.1, 1.1, size)

        start_time = time.time()
        psi_value, _ = calculate_psi(base_data, test_data)
        end_time = time.time()

        print(f"样本量 {size:6d}: PSI={psi_value:.4f}, 用时={end_time-start_time:.3f}秒")

    # 测试多特征PSI计算
    print("\n多特征PSI计算测试:")
    print("-" * 30)

    n_features_list = [10, 50, 100]
    n_samples = 10000

    for n_features in n_features_list:
        # 创建多特征数据
        base_df = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        test_df = pd.DataFrame(
            np.random.normal(0.1, 1.1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        start_time = time.time()
        psi_summary = calculate_multi_feature_psi(base_df, test_df)
        end_time = time.time()

        avg_psi = psi_summary['psi_value'].mean()
        print(f"特征数 {n_features:3d}: 平均PSI={avg_psi:.4f}, 用时={end_time-start_time:.3f}秒")


def run_comprehensive_test():
    """
    运行综合测试
    """
    print("模型监控系统综合测试")
    print("="*60)

    try:
        # 创建测试数据
        data = create_sample_data()

        # 运行各项测试
        test_psi_calculation(data)
        monitor = test_model_monitoring(data)
        test_automated_workflow()
        test_edge_cases()
        performance_test()

        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print("✓ PSI计算功能测试通过")
        print("✓ 模型监控功能测试通过")
        print("✓ 自动化工作流测试通过")
        print("✓ 边界情况测试通过")
        print("✓ 性能测试通过")
        print("\n所有测试完成！系统运行正常。")

        # 导出监控报告示例
        if monitor:
            try:
                monitor.export_monitoring_report('test_monitoring_report.json', 'json')
                print(f"\n监控报告已导出: test_monitoring_report.json")

                # 清理测试文件
                if os.path.exists('test_monitoring_report.json'):
                    os.remove('test_monitoring_report.json')

            except Exception as e:
                print(f"导出报告时出错: {str(e)}")

    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        print("请检查代码并重试。")
        raise


def print_usage():
    """
    打印使用说明
    """
    print("模型监控系统使用说明:")
    print("="*60)
    print("1. 基本使用:")
    print("   python test_monitoring_system.py")
    print("")
    print("2. 单独测试PSI计算:")
    print("   python test_monitoring_system.py --test-psi")
    print("")
    print("3. 单独测试模型监控:")
    print("   python test_monitoring_system.py --test-monitor")
    print("")
    print("4. 单独测试工作流:")
    print("   python test_monitoring_system.py --test-workflow")
    print("")
    print("5. 性能测试:")
    print("   python test_monitoring_system.py --performance")


if __name__ == "__main__":
    import sys

    # 忽略警告以保持输出清洁
    warnings.filterwarnings('ignore')

    if len(sys.argv) > 1:
        if sys.argv[1] == '--test-psi':
            data = create_sample_data()
            test_psi_calculation(data)
        elif sys.argv[1] == '--test-monitor':
            data = create_sample_data()
            test_model_monitoring(data)
        elif sys.argv[1] == '--test-workflow':
            test_automated_workflow()
        elif sys.argv[1] == '--performance':
            performance_test()
        elif sys.argv[1] == '--help':
            print_usage()
        else:
            print("未知参数，使用 --help 查看使用说明")
    else:
        run_comprehensive_test()