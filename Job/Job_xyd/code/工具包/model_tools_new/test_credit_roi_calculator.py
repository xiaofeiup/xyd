"""
测试信贷ROI计算器
展示简化版和完整版的差异
"""

from model_tools.utils.credit_roi_calculator_v2 import (
    CreditROICalculator,
    BasicCreditParams,
    ComprehensiveCreditParams
)
import pandas as pd

def test_roi_calculator():
    """测试ROI计算器功能"""
    print("🏦 信贷ROI计算器测试")
    print("=" * 60)

    calculator = CreditROICalculator()

    # 1. 基于表格数据的场景分析
    print("\n📊 表格场景分析（简化版）:")
    table_analysis = calculator.calculate_table_scenario()

    base = table_analysis['base_scenario']
    new_data = table_analysis['new_data_scenario']
    changes = table_analysis['summary']

    print(f"现有CPS成本:")
    print(f"  - 数据成本: ¥{base['total_data_cost']:,}")
    print(f"  - 放款金额: ¥{base['total_loan_amount']:,}")
    print(f"  - 净利润: ¥{base['net_profit']:,}")
    print(f"  - CPS成本率: {base['cps_cost_rate']:.3f}")
    print(f"  - 万元收益: ¥{base['unit_profit_per_10k']:.0f}")

    print(f"\n新增数据源后:")
    print(f"  - 数据成本: ¥{new_data['total_data_cost']:,}")
    print(f"  - 放款金额: ¥{new_data['total_loan_amount']:,}")
    print(f"  - 净利润: ¥{new_data['net_profit']:,}")
    print(f"  - CPS成本率: {new_data['cps_cost_rate']:.3f}")
    print(f"  - 万元收益: ¥{new_data['unit_profit_per_10k']:.0f}")

    print(f"\n📈 变化分析:")
    print(f"  - 数据成本变化: {changes['data_cost_increase']:+.2%}")
    print(f"  - CPS成本率变化: {changes['cps_rate_change']:+.2%}")
    print(f"  - 利润变化: {changes['profit_change']:+.2%}")
    print(f"  - 万元收益变化: {changes['unit_profit_change']:+.2%}")
    print(f"  - 放款量变化: {changes['loan_volume_change']:+.2%}")

    # 2. 简化版 vs 完整版对比
    print(f"\n🔍 简化版 vs 完整版对比:")

    # 使用相同参数
    basic_params = BasicCreditParams(
        entry_volume=100000,
        unit_data_cost=8.0,
        pass_rate=0.10,
        avg_loan_amount=7000,
        withdrawal_rate=0.80,
        risk_rate=0.065,
        pricing_rate=0.24
    )

    comprehensive_params = ComprehensiveCreditParams(
        # 继承基础参数
        entry_volume=100000,
        unit_data_cost=8.0,
        pass_rate=0.10,
        avg_loan_amount=7000,
        withdrawal_rate=0.80,
        risk_rate=0.065,
        pricing_rate=0.24,
        # 额外的完整版参数
        funding_cost_rate=0.045,
        reserve_ratio=0.12,
        operation_cost_rate=0.02,
        early_repay_rate=0.15,
        avg_holding_period=8.0
    )

    basic_result = calculator.calculate_basic_roi(basic_params)
    comprehensive_result = calculator.calculate_comprehensive_roi(comprehensive_params)

    print(f"简化版结果:")
    print(f"  - 总成本: ¥{basic_result['total_cost']:,}")
    print(f"  - 净利润: ¥{basic_result['net_profit']:,}")
    print(f"  - ROI: {basic_result['roi_on_data_cost']:.2%}")
    print(f"  - 利润率: {basic_result['profit_margin']:.2%}")

    print(f"\n完整版结果:")
    print(f"  - 总成本: ¥{comprehensive_result['total_comprehensive_cost']:,}")
    print(f"  - 净利润: ¥{comprehensive_result['comprehensive_net_profit']:,}")
    print(f"  - ROI: {comprehensive_result['comprehensive_roi']:.2%}")
    print(f"  - 利润率: {comprehensive_result['comprehensive_profit_margin']:.2%}")

    print(f"\n💰 完整版详细成本结构:")
    cost_items = [
        ('数据成本', comprehensive_result['total_data_cost']),
        ('坏账成本', comprehensive_result['credit_loss']),
        ('资金成本', comprehensive_result['funding_cost']),
        ('保证金成本', comprehensive_result['reserve_cost']),
        ('运营成本', comprehensive_result['operation_cost']),
        ('流量成本', comprehensive_result['traffic_cost']),
        ('提前还款损失', comprehensive_result['early_repay_loss'])
    ]

    total_cost = comprehensive_result['total_comprehensive_cost']
    for name, cost in cost_items:
        ratio = cost / total_cost if total_cost > 0 else 0
        print(f"  - {name}: ¥{cost:,.0f} ({ratio:.1%})")

    # 3. 多场景对比
    print(f"\n📊 多场景对比:")
    scenarios = {
        '基准场景': BasicCreditParams(),
        '优化场景1_降成本': BasicCreditParams(unit_data_cost=6.0),
        '优化场景2_提通过率': BasicCreditParams(pass_rate=0.12),
        '优化场景3_降风险': BasicCreditParams(risk_rate=0.055),
        '综合优化': BasicCreditParams(unit_data_cost=6.5, pass_rate=0.12, risk_rate=0.055)
    }

    comparison = calculator.scenario_comparison(scenarios)

    # 格式化显示
    print(f"{'场景名称':15} {'净利润(万元)':12} {'ROI':8} {'万元收益':10} {'CPS成本率':10}")
    print("-" * 65)
    for _, row in comparison.iterrows():
        print(f"{row['scenario']:15} {row['net_profit']/10000:8.0f} {row['roi']:8.1%} {row['unit_profit_per_10k']:6.0f} {row['cps_cost_rate']:8.3f}")

    print(f"\n✅ 测试完成！")

    return calculator, table_analysis

if __name__ == "__main__":
    calculator, analysis = test_roi_calculator()