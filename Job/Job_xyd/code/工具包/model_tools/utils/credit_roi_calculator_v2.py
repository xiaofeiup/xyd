"""
信贷业务ROI计算器

基于实际信贷业务ROI测算表格的计算函数，提供两个版本：
1. 简化版：基础CPS成本和利润计算
2. 完整版：考虑资金成本、保证金、流量、运营、提前还款等全要素
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class BasicCreditParams:
    """基础信贷参数（简化版）"""
    entry_volume: int = 100000          # 进件量
    unit_data_cost: float = 8.0         # 单人数据成本,元
    pass_rate: float = 0.10             # 通过率
    avg_loan_amount: float = 7000       # 户均额度
    withdrawal_rate: float = 0.80       # 提现率
    risk_rate: float = 0.065            # 风险率
    pricing_rate: float = 0.24          # 定价（年化利率）
    risk_free_rate: float = 0.07        # 无风险利润率


@dataclass
class ComprehensiveCreditParams(BasicCreditParams):
    """全面信贷参数（完整版）"""
    # 资金成本相关
    funding_cost_rate: float = 0.045    # 资金成本率
    reserve_ratio: float = 0.12         # 保证金比例

    # 运营成本相关
    operation_cost_rate: float = 0.02   # 运营成本率
    collection_cost_rate: float = 0.01  # 催收成本率

    # 流量和获客
    traffic_cost_per_entry: float = 5.0 # 流量成本/进件
    marketing_cost_rate: float = 0.003  # 营销成本率

    # 提前还款影响
    early_repay_rate: float = 0.15      # 提前还款率
    avg_holding_period: float = 8.0     # 平均持有期(月)

    # 其他费用
    system_cost_rate: float = 0.001     # 系统成本率
    compliance_cost_rate: float = 0.002 # 合规成本率


class CreditROICalculator:
    """信贷ROI计算器"""

    def __init__(self):
        pass

    def calculate_basic_roi(self, params: BasicCreditParams) -> Dict[str, Union[float, int]]:
        """
        简化版ROI计算（对应表格基础版本）

        Args:
            params: 基础信贷参数

        Returns:
            基础ROI计算结果
        """
        # 1. 基础指标计算
        total_data_cost = params.entry_volume * params.unit_data_cost
        passed_customers = int(params.entry_volume * params.pass_rate)
        actual_loan_customers = int(passed_customers * params.withdrawal_rate)
        total_loan_amount = actual_loan_customers * params.avg_loan_amount

        # 2. CPS指标计算
        avg_acquisition_cost = total_data_cost / actual_loan_customers if actual_loan_customers > 0 else 0
        cps_cost_rate = total_data_cost / total_loan_amount if total_loan_amount > 0 else 0

        # 3. 收入计算
        gross_revenue = total_loan_amount * params.pricing_rate

        # 4. 成本计算
        credit_loss = total_loan_amount * params.risk_rate
        total_cost = total_data_cost + credit_loss

        # 5. 利润计算
        net_profit = gross_revenue - total_cost

        # 6. ROI相关指标
        roi_on_data_cost = net_profit / total_data_cost if total_data_cost > 0 else 0
        profit_margin = net_profit / total_loan_amount if total_loan_amount > 0 else 0
        unit_profit_per_10k = (net_profit / total_loan_amount * 10000) if total_loan_amount > 0 else 0

        # 7. 无风险利润对比
        risk_free_profit = total_loan_amount * params.risk_free_rate
        risk_premium = net_profit - risk_free_profit

        return {
            # 基础指标
            'entry_volume': params.entry_volume,
            'unit_data_cost': params.unit_data_cost,
            'total_data_cost': total_data_cost,
            'pass_rate': params.pass_rate,
            'passed_customers': passed_customers,
            'withdrawal_rate': params.withdrawal_rate,
            'actual_loan_customers': actual_loan_customers,
            'avg_loan_amount': params.avg_loan_amount,
            'total_loan_amount': total_loan_amount,

            # CPS指标
            'avg_acquisition_cost': avg_acquisition_cost,
            'cps_cost_rate': cps_cost_rate,

            # 收入成本
            'gross_revenue': gross_revenue,
            'credit_loss': credit_loss,
            'total_cost': total_cost,
            'net_profit': net_profit,

            # ROI指标
            'roi_on_data_cost': roi_on_data_cost,
            'profit_margin': profit_margin,
            'unit_profit_per_10k': unit_profit_per_10k,

            # 风险对比
            'risk_free_profit': risk_free_profit,
            'risk_premium': risk_premium,
            'risk_adjusted_return': net_profit / total_loan_amount if total_loan_amount > 0 else 0
        }

    def calculate_comprehensive_roi(self, params: ComprehensiveCreditParams) -> Dict[str, Union[float, int]]:
        """
        完整版ROI计算（考虑所有业务因素）

        Args:
            params: 全面信贷参数

        Returns:
            完整ROI计算结果
        """
        # 1. 基础计算（继承简化版）
        basic_result = self.calculate_basic_roi(params)

        total_loan_amount = basic_result['total_loan_amount']
        actual_loan_customers = basic_result['actual_loan_customers']

        # 2. 资金成本计算
        funding_cost = total_loan_amount * params.funding_cost_rate * (params.avg_holding_period / 12)
        reserve_cost = total_loan_amount * params.reserve_ratio * params.funding_cost_rate * (params.avg_holding_period / 12)

        # 3. 运营成本计算
        operation_cost = total_loan_amount * params.operation_cost_rate
        collection_cost = basic_result['credit_loss'] * params.collection_cost_rate  # 催收成本基于坏账
        system_cost = total_loan_amount * params.system_cost_rate
        compliance_cost = total_loan_amount * params.compliance_cost_rate

        # 4. 流量和营销成本
        traffic_cost = params.entry_volume * params.traffic_cost_per_entry
        marketing_cost = total_loan_amount * params.marketing_cost_rate

        # 5. 提前还款影响
        early_repay_loss = total_loan_amount * params.early_repay_rate * (
            params.pricing_rate * (12 - params.avg_holding_period) / 12
        )  # 提前还款导致的利息损失

        # 6. 收入调整（考虑实际持有期）
        actual_revenue = total_loan_amount * params.pricing_rate * (params.avg_holding_period / 12)

        # 7. 总成本汇总
        comprehensive_costs = {
            'data_cost': basic_result['total_data_cost'],
            'credit_loss': basic_result['credit_loss'],
            'funding_cost': funding_cost,
            'reserve_cost': reserve_cost,
            'operation_cost': operation_cost,
            'collection_cost': collection_cost,
            'system_cost': system_cost,
            'compliance_cost': compliance_cost,
            'traffic_cost': traffic_cost,
            'marketing_cost': marketing_cost,
            'early_repay_loss': early_repay_loss
        }

        total_comprehensive_cost = sum(comprehensive_costs.values())

        # 8. 净利润重新计算
        comprehensive_net_profit = actual_revenue - total_comprehensive_cost

        # 9. 全面ROI指标
        comprehensive_roi = comprehensive_net_profit / basic_result['total_data_cost'] if basic_result['total_data_cost'] > 0 else 0
        comprehensive_profit_margin = comprehensive_net_profit / total_loan_amount if total_loan_amount > 0 else 0
        comprehensive_unit_profit = (comprehensive_net_profit / total_loan_amount * 10000) if total_loan_amount > 0 else 0

        # 10. 成本结构分析
        cost_structure = {k: v/total_comprehensive_cost for k, v in comprehensive_costs.items() if total_comprehensive_cost > 0}

        # 11. 效率指标
        customer_acquisition_efficiency = comprehensive_net_profit / actual_loan_customers if actual_loan_customers > 0 else 0
        capital_efficiency = comprehensive_net_profit / (total_loan_amount + total_loan_amount * params.reserve_ratio) if total_loan_amount > 0 else 0

        # 合并基础结果和全面结果
        comprehensive_result = basic_result.copy()
        comprehensive_result.update({
            # 调整后的收入和利润
            'actual_revenue': actual_revenue,
            'comprehensive_net_profit': comprehensive_net_profit,
            'profit_difference': comprehensive_net_profit - basic_result['net_profit'],

            # 详细成本结构
            'funding_cost': funding_cost,
            'reserve_cost': reserve_cost,
            'operation_cost': operation_cost,
            'collection_cost': collection_cost,
            'system_cost': system_cost,
            'compliance_cost': compliance_cost,
            'traffic_cost': traffic_cost,
            'marketing_cost': marketing_cost,
            'early_repay_loss': early_repay_loss,
            'total_comprehensive_cost': total_comprehensive_cost,

            # 全面ROI指标
            'comprehensive_roi': comprehensive_roi,
            'comprehensive_profit_margin': comprehensive_profit_margin,
            'comprehensive_unit_profit': comprehensive_unit_profit,

            # 成本结构占比
            'cost_structure': cost_structure,

            # 效率指标
            'customer_acquisition_efficiency': customer_acquisition_efficiency,
            'capital_efficiency': capital_efficiency,

            # 业务参数
            'avg_holding_period': params.avg_holding_period,
            'early_repay_rate': params.early_repay_rate,
            'funding_cost_rate': params.funding_cost_rate
        })

        return comprehensive_result

    def scenario_comparison(
        self,
        scenarios: Dict[str, Union[BasicCreditParams, ComprehensiveCreditParams]],
        use_comprehensive: bool = False
    ) -> pd.DataFrame:
        """
        多场景对比分析

        Args:
            scenarios: 场景字典 {场景名: 参数}
            use_comprehensive: 是否使用全面计算

        Returns:
            场景对比结果DataFrame
        """
        results = []

        for scenario_name, params in scenarios.items():
            if use_comprehensive:
                result = self.calculate_comprehensive_roi(params)
                key_metrics = {
                    'scenario': scenario_name,
                    'net_profit': result['comprehensive_net_profit'],
                    'roi': result['comprehensive_roi'],
                    'profit_margin': result['comprehensive_profit_margin'],
                    'unit_profit_per_10k': result['comprehensive_unit_profit'],
                    'cps_cost_rate': result['cps_cost_rate'],
                    'total_cost': result['total_comprehensive_cost']
                }
            else:
                result = self.calculate_basic_roi(params)
                key_metrics = {
                    'scenario': scenario_name,
                    'net_profit': result['net_profit'],
                    'roi': result['roi_on_data_cost'],
                    'profit_margin': result['profit_margin'],
                    'unit_profit_per_10k': result['unit_profit_per_10k'],
                    'cps_cost_rate': result['cps_cost_rate'],
                    'total_cost': result['total_cost']
                }

            results.append(key_metrics)

        return pd.DataFrame(results)

    def calculate_table_scenario(self) -> Dict[str, any]:
        """
        根据提供的表格数据计算ROI变化场景

        Returns:
            表格场景分析结果
        """
        # 场景1：现有CPS成本（基准）
        base_params = BasicCreditParams(
            entry_volume=100000,
            unit_data_cost=8.0,
            pass_rate=0.10,
            avg_loan_amount=7000,
            withdrawal_rate=0.80,
            risk_rate=0.065,
            pricing_rate=0.24,
            risk_free_rate=0.07
        )

        # 场景2：新增数据源后CPS变化（单价+3.75%，通过率+10%，风险-3.08%）
        new_data_params = BasicCreditParams(
            entry_volume=100000,
            unit_data_cost=8.3,  # +3.75%
            pass_rate=0.11,      # +10%
            avg_loan_amount=7000,
            withdrawal_rate=0.80,
            risk_rate=0.063,     # -3.08%
            pricing_rate=0.24,
            risk_free_rate=0.07
        )

        # 计算两个场景
        base_result = self.calculate_basic_roi(base_params)
        new_result = self.calculate_basic_roi(new_data_params)

        # 计算变化
        changes = {}
        for key in ['total_data_cost', 'cps_cost_rate', 'net_profit', 'unit_profit_per_10k', 'total_loan_amount']:
            if key in base_result and key in new_result:
                base_val = base_result[key]
                new_val = new_result[key]
                change_rate = (new_val - base_val) / base_val if base_val != 0 else 0
                changes[f'{key}_change'] = change_rate

        return {
            'base_scenario': base_result,
            'new_data_scenario': new_result,
            'changes': changes,
            'summary': {
                'data_cost_increase': changes.get('total_data_cost_change', 0),
                'cps_rate_change': changes.get('cps_cost_rate_change', 0),
                'profit_change': changes.get('net_profit_change', 0),
                'unit_profit_change': changes.get('unit_profit_per_10k_change', 0),
                'loan_volume_change': changes.get('total_loan_amount_change', 0)
            }
        }


def create_roi_calculator_example():
    """创建ROI计算器使用示例"""
    calculator = CreditROICalculator()

    print("🏦 信贷业务ROI计算示例")
    print("=" * 60)

    # 1. 表格场景分析
    print("\n📊 根据表格数据的场景分析:")
    table_analysis = calculator.calculate_table_scenario()

    base = table_analysis['base_scenario']
    new_data = table_analysis['new_data_scenario']
    summary = table_analysis['summary']

    print(f"基准场景 - 净利润: ¥{base['net_profit']:,.0f}, 万元收益: ¥{base['unit_profit_per_10k']:.0f}")
    print(f"新数据源 - 净利润: ¥{new_data['net_profit']:,.0f}, 万元收益: ¥{new_data['unit_profit_per_10k']:.0f}")
    print(f"利润变化: {summary['profit_change']:+.2%}, 万元收益变化: {summary['unit_profit_change']:+.2%}")

    # 2. 简化版vs完整版对比
    print("\n🔍 简化版vs完整版对比:")

    simple_params = BasicCreditParams()
    comprehensive_params = ComprehensiveCreditParams()

    simple_result = calculator.calculate_basic_roi(simple_params)
    comprehensive_result = calculator.calculate_comprehensive_roi(comprehensive_params)

    print(f"简化版净利润: ¥{simple_result['net_profit']:,.0f}")
    print(f"完整版净利润: ¥{comprehensive_result['comprehensive_net_profit']:,.0f}")
    print(f"利润差异: ¥{comprehensive_result['profit_difference']:,.0f}")

    # 3. 成本结构分析
    print(f"\n💰 完整版成本结构:")
    cost_structure = comprehensive_result['cost_structure']
    for cost_type, ratio in cost_structure.items():
        print(f"   {cost_type}: {ratio:.2%}")

    return calculator, table_analysis


# 使用示例
if __name__ == "__main__":
    calculator, analysis = create_roi_calculator_example()

    # 多场景对比
    scenarios = {
        '基准场景': BasicCreditParams(),
        '低成本场景': BasicCreditParams(unit_data_cost=6.0, risk_rate=0.055),
        '高通过率场景': BasicCreditParams(pass_rate=0.15, risk_rate=0.075)
    }

    comparison = calculator.scenario_comparison(scenarios)
    print(f"\n📈 多场景对比:")
    print(comparison.round(4))