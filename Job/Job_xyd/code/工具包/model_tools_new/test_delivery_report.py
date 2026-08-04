"""
测试模型交付报告生成器
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加模块路径
sys.path.append('/Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new')

from model_tools.evaluation.delivery_report import ModelDeliveryReport


def create_realistic_test_data(n_samples=20000):
    """创建更真实的测试数据"""
    np.random.seed(42)

    # 生成时间范围 (2023年1月到2024年6月)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)

    # 生成18个月的数据
    months = pd.date_range(start='2023-01-01', end='2024-06-01', freq='MS')

    data = []

    # 按月生成数据，确保每月都有足够样本
    samples_per_month = n_samples // len(months)

    for month_idx, month in enumerate(months):
        # 为每个月生成数据
        for i in range(samples_per_month):
            # 随机选择具体日期
            next_month = month + pd.DateOffset(months=1)
            days_in_month = (next_month - month).days
            random_day = np.random.randint(0, days_in_month)
            sample_date = month + pd.Timedelta(days=random_day)

            # 样本类型分配：前12个月为train/test，后6个月为oot
            if month_idx < 12:
                sample_type = np.random.choice(['train', 'test'], p=[0.7, 0.3])
            else:
                sample_type = 'oot'

            # 生成特征变量 - 模拟风控特征
            age = np.random.normal(35, 10)  # 年龄
            income = np.random.lognormal(10, 0.5)  # 收入
            credit_score = np.random.normal(650, 100)  # 信用分
            debt_ratio = np.random.beta(2, 5)  # 负债率
            work_years = np.random.gamma(2, 2)  # 工作年限

            # 历史逾期次数 - 泊松分布
            past_due_count = np.random.poisson(0.5)

            # 查询次数
            inquiry_count = np.random.poisson(2)

            # 银行产品数量
            bank_product_count = np.random.poisson(1.5)

            # 月收入稳定性评分
            income_stability = np.random.normal(70, 15)

            # 社交网络评分
            social_score = np.random.normal(60, 20)

            # 添加时间趋势和季节性影响
            time_trend = month_idx * 0.01  # 轻微时间趋势
            seasonal_effect = 0.1 * np.sin(2 * np.pi * month_idx / 12)  # 季节性

            # 生成目标变量 - 基于逻辑回归模型，调整参数以获得合理的违约率
            logit = (
                -1.5 +  # 截距，调高以增加违约概率
                -0.01 * age +  # 年龄越大违约概率越低
                -0.00005 * income +  # 收入越高违约概率越低
                -0.003 * credit_score +  # 信用分越高违约概率越低
                3.0 * debt_ratio +  # 负债率越高违约概率越高
                -0.05 * work_years +  # 工作年限越长违约概率越低
                0.5 * past_due_count +  # 历史逾期次数越多违约概率越高
                0.2 * inquiry_count +  # 查询次数越多违约概率越高
                -0.02 * bank_product_count +  # 银行产品越多违约概率越低
                -0.005 * income_stability +  # 收入稳定性越高违约概率越低
                -0.002 * social_score +  # 社交评分越高违约概率越低
                time_trend +  # 时间趋势
                seasonal_effect +  # 季节性效应
                np.random.normal(0, 0.3)  # 随机噪声
            )

            # 转换为概率
            prob = 1 / (1 + np.exp(-logit))
            target = np.random.binomial(1, prob)

            # 生成模型分数 - 基于特征的线性组合 + 噪声
            model_score = (
                0.5 * (credit_score - 650) / 100 +
                0.3 * (income - 20000) / 50000 +
                0.2 * (age - 35) / 10 +
                -0.4 * debt_ratio +
                0.1 * work_years +
                -0.2 * past_due_count +
                -0.1 * inquiry_count +
                0.05 * bank_product_count +
                0.1 * income_stability / 100 +
                0.05 * social_score / 100 +
                np.random.normal(0, 0.1)
            )

            data.append({
                'date': sample_date,
                'sample_type': sample_type,
                'target': target,
                'score': model_score,
                'age': age,
                'income': income,
                'credit_score': credit_score,
                'debt_ratio': debt_ratio,
                'work_years': work_years,
                'past_due_count': past_due_count,
                'inquiry_count': inquiry_count,
                'bank_product_count': bank_product_count,
                'income_stability': income_stability,
                'social_score': social_score
            })

    # 确保剩余样本被分配
    remaining_samples = n_samples - len(data)
    if remaining_samples > 0:
        for i in range(remaining_samples):
            # 随机选择一个月
            month = np.random.choice(months)
            month_idx = list(months).index(month)

            next_month = month + pd.DateOffset(months=1)
            days_in_month = (next_month - month).days
            random_day = np.random.randint(0, days_in_month)
            sample_date = month + pd.Timedelta(days=random_day)

            if month_idx < 12:
                sample_type = np.random.choice(['train', 'test'], p=[0.7, 0.3])
            else:
                sample_type = 'oot'

            # 简化特征生成
            age = np.random.normal(35, 10)
            income = np.random.lognormal(10, 0.5)
            credit_score = np.random.normal(650, 100)
            debt_ratio = np.random.beta(2, 5)
            work_years = np.random.gamma(2, 2)
            past_due_count = np.random.poisson(0.5)
            inquiry_count = np.random.poisson(2)
            bank_product_count = np.random.poisson(1.5)
            income_stability = np.random.normal(70, 15)
            social_score = np.random.normal(60, 20)

            # 简化目标变量生成
            logit = -1.0 + np.random.normal(0, 1)
            prob = 1 / (1 + np.exp(-logit))
            target = np.random.binomial(1, prob)

            model_score = np.random.normal(0, 1)

            data.append({
                'date': sample_date,
                'sample_type': sample_type,
                'target': target,
                'score': model_score,
                'age': age,
                'income': income,
                'credit_score': credit_score,
                'debt_ratio': debt_ratio,
                'work_years': work_years,
                'past_due_count': past_due_count,
                'inquiry_count': inquiry_count,
                'bank_product_count': bank_product_count,
                'income_stability': income_stability,
                'social_score': social_score
            })

    df = pd.DataFrame(data)

    # 确保数据类型正确
    df['date'] = pd.to_datetime(df['date'])
    df['target'] = df['target'].astype(int)
    df['age'] = df['age'].round(0).astype(int)
    df['income'] = df['income'].round(0).astype(int)
    df['credit_score'] = df['credit_score'].round(0).astype(int)
    df['work_years'] = df['work_years'].round(1)
    df['past_due_count'] = df['past_due_count'].astype(int)
    df['inquiry_count'] = df['inquiry_count'].astype(int)
    df['bank_product_count'] = df['bank_product_count'].astype(int)
    df['income_stability'] = df['income_stability'].round(1)
    df['social_score'] = df['social_score'].round(1)

    # 添加一些衍生特征
    df['income_to_age_ratio'] = (df['income'] / df['age']).round(2)
    df['credit_utilization'] = (df['debt_ratio'] * 100).round(2)
    df['experience_score'] = (df['work_years'] * 10 + df['age'] * 0.5).round(1)

    return df


def test_delivery_report():
    """测试模型交付报告生成"""
    print("正在生成测试数据...")

    # 生成测试数据
    test_data = create_realistic_test_data(n_samples=20000)

    print(f"测试数据生成完成！")
    print(f"数据形状: {test_data.shape}")
    print(f"时间范围: {test_data['date'].min()} 到 {test_data['date'].max()}")
    print(f"样本类型分布:\n{test_data['sample_type'].value_counts()}")
    print(f"目标变量分布:\n{test_data['target'].value_counts()}")
    print(f"违约率: {test_data['target'].mean():.4f}")

    # 初始化报告生成器
    print("\n正在初始化报告生成器...")
    reporter = ModelDeliveryReport(
        data=test_data,
        target_col='target',
        score_col='score',
        date_col='date',
        sample_type_col='sample_type'
    )

    # 定义特征列表和中文名称
    features = [
        'age', 'income', 'credit_score', 'debt_ratio', 'work_years',
        'past_due_count', 'inquiry_count', 'bank_product_count',
        'income_stability', 'social_score', 'income_to_age_ratio',
        'credit_utilization', 'experience_score'
    ]

    feature_names = {
        'age': '年龄',
        'income': '月收入',
        'credit_score': '信用评分',
        'debt_ratio': '负债率',
        'work_years': '工作年限',
        'past_due_count': '历史逾期次数',
        'inquiry_count': '征信查询次数',
        'bank_product_count': '银行产品数量',
        'income_stability': '收入稳定性评分',
        'social_score': '社交网络评分',
        'income_to_age_ratio': '收入年龄比',
        'credit_utilization': '信用利用率',
        'experience_score': '经验评分'
    }

    # 生成完整报告
    print("\n正在生成完整的模型交付报告...")
    output_path = '/Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new/model_delivery_report_test.xlsx'

    try:
        report = reporter.generate_full_report(
            features=features,
            feature_names=feature_names,
            save_path=output_path
        )

        print(f"\n✓ 报告生成成功！")
        print(f"✓ Excel文件已保存至: {output_path}")

        # 显示各表格的基本信息
        print(f"\n报告包含以下表格:")
        for sheet_name, df in report.items():
            if not df.empty:
                print(f"  - {sheet_name}: {len(df)} 行 × {len(df.columns)} 列")
            else:
                print(f"  - {sheet_name}: 空表格")

        # 显示一些关键统计信息
        print(f"\n关键统计信息:")
        if not report['样本情况'].empty:
            total_samples = report['样本情况'].iloc[-1]['Total']
            total_bad_rate = report['样本情况'].iloc[-1]['Bad_Rate']
            print(f"  - 总样本数: {total_samples}")
            print(f"  - 总体违约率: {total_bad_rate}")

        if not report['模型效果'].empty:
            all_performance = report['模型效果'][report['模型效果']['样本类型'] == 'all']
            if not all_performance.empty:
                ks = all_performance.iloc[0]['KS']
                auc = all_performance.iloc[0]['AUC']
                print(f"  - 全样本KS: {ks}")
                print(f"  - 全样本AUC: {auc}")

        return report

    except Exception as e:
        print(f"✗ 报告生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 运行测试
    report = test_delivery_report()

    if report is not None:
        print("\n🎉 测试完成！请查看生成的Excel文件。")
    else:
        print("\n❌ 测试失败，请检查错误信息。")