"""
信贷业务ROI计算工具

本模块提供信贷领域的投资回报率(ROI)计算功能，包括：
- 基础ROI计算
- 客户生命周期价值(CLV)计算
- 风险调整后的收益计算
- 营销成本效益分析
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class CreditMetrics:
    """信贷业务指标数据结构"""
    principal: float  # 本金
    interest_rate: float  # 年化利率
    term_months: int  # 期限(月)
    default_rate: float  # 违约率
    recovery_rate: float = 0.0  # 回收率
    operational_cost: float = 0.0  # 运营成本
    acquisition_cost: float = 0.0  # 获客成本


class CreditROICalculator:
    """信贷业务ROI计算器"""

    def __init__(self):
        self.default_recovery_rate = 0.3  # 默认回收率30%
        self.default_discount_rate = 0.12  # 默认折现率12%

    def calculate_basic_roi(
        self,
        revenue: float,
        cost: float,
        investment: float = None
    ) -> float:
        """
        计算基础ROI

        Args:
            revenue: 总收入
            cost: 总成本
            investment: 投资额(如未提供则使用cost)

        Returns:
            ROI百分比
        """
        if investment is None:
            investment = cost

        if investment == 0:
            return float('inf') if revenue > cost else 0.0

        roi = (revenue - cost) / investment
        return roi * 100

    def calculate_loan_roi(self, metrics: CreditMetrics) -> Dict[str, float]:
        """
        计算单笔贷款ROI

        Args:
            metrics: 信贷业务指标

        Returns:
            包含各种ROI计算结果的字典
        """
        # 计算利息收入
        monthly_rate = metrics.interest_rate / 12
        total_interest = metrics.principal * monthly_rate * metrics.term_months

        # 计算预期损失
        expected_loss = metrics.principal * metrics.default_rate * (1 - metrics.recovery_rate)

        # 计算净收入
        net_revenue = total_interest - expected_loss - metrics.operational_cost

        # 计算总投资(本金 + 获客成本)
        total_investment = metrics.principal + metrics.acquisition_cost

        # 基础ROI
        basic_roi = self.calculate_basic_roi(
            revenue=total_interest,
            cost=metrics.operational_cost + expected_loss,
            investment=total_investment
        )

        # 风险调整ROI
        risk_adjusted_roi = (net_revenue / total_investment) * 100

        # 年化ROI
        annualized_roi = (risk_adjusted_roi / metrics.term_months) * 12

        return {
            'basic_roi': basic_roi,
            'risk_adjusted_roi': risk_adjusted_roi,
            'annualized_roi': annualized_roi,
            'net_revenue': net_revenue,
            'expected_loss': expected_loss,
            'total_interest': total_interest
        }

    def calculate_portfolio_roi(
        self,
        loan_data: pd.DataFrame,
        principal_col: str = 'principal',
        rate_col: str = 'interest_rate',
        term_col: str = 'term_months',
        default_col: str = 'default_rate',
        recovery_col: str = 'recovery_rate',
        cost_col: str = 'operational_cost'
    ) -> Dict[str, float]:
        """
        计算贷款组合ROI

        Args:
            loan_data: 贷款数据DataFrame
            *_col: 各列名称

        Returns:
            组合ROI指标字典
        """
        results = []

        for _, row in loan_data.iterrows():
            metrics = CreditMetrics(
                principal=row[principal_col],
                interest_rate=row[rate_col],
                term_months=row[term_col],
                default_rate=row[default_col],
                recovery_rate=row.get(recovery_col, self.default_recovery_rate),
                operational_cost=row.get(cost_col, 0)
            )

            loan_roi = self.calculate_loan_roi(metrics)
            loan_roi['weight'] = metrics.principal
            results.append(loan_roi)

        df_results = pd.DataFrame(results)
        total_weight = df_results['weight'].sum()

        # 加权平均ROI
        weighted_roi = (df_results['risk_adjusted_roi'] * df_results['weight']).sum() / total_weight
        weighted_annualized = (df_results['annualized_roi'] * df_results['weight']).sum() / total_weight

        # 总收益指标
        total_revenue = df_results['total_interest'].sum()
        total_loss = df_results['expected_loss'].sum()
        total_net = df_results['net_revenue'].sum()

        return {
            'portfolio_roi': weighted_roi,
            'portfolio_annualized_roi': weighted_annualized,
            'total_revenue': total_revenue,
            'total_expected_loss': total_loss,
            'total_net_revenue': total_net,
            'portfolio_size': len(loan_data),
            'total_principal': total_weight
        }

    def calculate_clv_roi(
        self,
        monthly_revenue: float,
        monthly_cost: float,
        churn_rate: float,
        acquisition_cost: float,
        discount_rate: float = None
    ) -> Dict[str, float]:
        """
        计算客户生命周期价值ROI

        Args:
            monthly_revenue: 月均收入
            monthly_cost: 月均成本
            churn_rate: 月流失率
            acquisition_cost: 获客成本
            discount_rate: 折现率

        Returns:
            CLV相关ROI指标
        """
        if discount_rate is None:
            discount_rate = self.default_discount_rate / 12  # 转换为月率

        # 客户平均生命周期(月)
        avg_lifetime = 1 / churn_rate if churn_rate > 0 else float('inf')

        # 月净利润
        monthly_profit = monthly_revenue - monthly_cost

        # CLV计算(考虑折现)
        if churn_rate > 0 and discount_rate > 0:
            clv = monthly_profit / (churn_rate + discount_rate)
        else:
            clv = monthly_profit * avg_lifetime

        # CLV ROI
        clv_roi = ((clv - acquisition_cost) / acquisition_cost) * 100 if acquisition_cost > 0 else 0

        # 回本周期
        payback_period = acquisition_cost / monthly_profit if monthly_profit > 0 else float('inf')

        return {
            'clv': clv,
            'clv_roi': clv_roi,
            'avg_lifetime_months': avg_lifetime,
            'payback_period_months': payback_period,
            'monthly_profit': monthly_profit
        }

    def calculate_campaign_roi(
        self,
        campaign_cost: float,
        conversions: int,
        avg_loan_amount: float,
        avg_interest_rate: float,
        avg_term_months: int,
        default_rate: float,
        recovery_rate: float = None
    ) -> Dict[str, float]:
        """
        计算营销活动ROI

        Args:
            campaign_cost: 活动总成本
            conversions: 转化客户数
            avg_loan_amount: 平均贷款金额
            avg_interest_rate: 平均利率
            avg_term_months: 平均期限
            default_rate: 违约率
            recovery_rate: 回收率

        Returns:
            营销活动ROI指标
        """
        if recovery_rate is None:
            recovery_rate = self.default_recovery_rate

        if conversions == 0:
            return {
                'campaign_roi': -100.0,
                'cost_per_acquisition': float('inf'),
                'total_revenue': 0.0,
                'total_profit': -campaign_cost
            }

        # 单客获客成本
        cost_per_acquisition = campaign_cost / conversions

        # 计算单笔贷款收益
        metrics = CreditMetrics(
            principal=avg_loan_amount,
            interest_rate=avg_interest_rate,
            term_months=avg_term_months,
            default_rate=default_rate,
            recovery_rate=recovery_rate,
            acquisition_cost=cost_per_acquisition
        )

        single_loan_roi = self.calculate_loan_roi(metrics)

        # 总收益计算
        total_revenue = single_loan_roi['total_interest'] * conversions
        total_expected_loss = single_loan_roi['expected_loss'] * conversions
        total_profit = total_revenue - total_expected_loss - campaign_cost

        # 活动ROI
        campaign_roi = (total_profit / campaign_cost) * 100

        return {
            'campaign_roi': campaign_roi,
            'cost_per_acquisition': cost_per_acquisition,
            'total_revenue': total_revenue,
            'total_expected_loss': total_expected_loss,
            'total_profit': total_profit,
            'conversion_count': conversions,
            'profit_per_customer': total_profit / conversions
        }

    def calculate_risk_adjusted_roi(
        self,
        base_roi: float,
        volatility: float,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        计算风险调整后的ROI

        Args:
            base_roi: 基础ROI
            volatility: 收益波动率
            confidence_level: 置信水平

        Returns:
            风险调整指标
        """
        # VaR计算(正态分布假设)
        z_score = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}.get(confidence_level, 1.65)
        var = base_roi - z_score * volatility

        # 夏普比率(假设无风险利率为3%)
        risk_free_rate = 3.0
        sharpe_ratio = (base_roi - risk_free_rate) / volatility if volatility > 0 else 0

        # 风险调整ROI
        risk_adjusted_roi = base_roi - (volatility * 0.5)  # 简单风险惩罚

        return {
            'risk_adjusted_roi': risk_adjusted_roi,
            'value_at_risk': var,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }


def batch_roi_analysis(
    data: pd.DataFrame,
    groupby_cols: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    批量ROI分析

    Args:
        data: 数据DataFrame
        groupby_cols: 分组列
        **kwargs: 传递给calculate_portfolio_roi的参数

    Returns:
        ROI分析结果DataFrame
    """
    calculator = CreditROICalculator()

    if groupby_cols:
        results = []
        for name, group in data.groupby(groupby_cols):
            roi_metrics = calculator.calculate_portfolio_roi(group, **kwargs)
            roi_metrics['group'] = name if isinstance(name, str) else '_'.join(map(str, name))
            results.append(roi_metrics)
        return pd.DataFrame(results)
    else:
        roi_metrics = calculator.calculate_portfolio_roi(data, **kwargs)
        return pd.DataFrame([roi_metrics])


# 使用示例
if __name__ == "__main__":
    # 示例：单笔贷款ROI计算
    calculator = CreditROICalculator()

    loan_metrics = CreditMetrics(
        principal=100000,
        interest_rate=0.12,
        term_months=12,
        default_rate=0.05,
        recovery_rate=0.3,
        operational_cost=2000,
        acquisition_cost=1000
    )

    roi_result = calculator.calculate_loan_roi(loan_metrics)
    print("单笔贷款ROI分析:")
    for key, value in roi_result.items():
        print(f"{key}: {value:.2f}")