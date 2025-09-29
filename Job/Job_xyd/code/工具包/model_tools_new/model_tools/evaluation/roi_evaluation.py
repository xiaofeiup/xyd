"""
基于ROI的模型评估模块

提供信贷模型的ROI评估功能，包括：
- 模型业务价值评估
- 不同策略的ROI对比
- 模型经济效益量化
- 决策阈值优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings

from ..utils.roi_calculator import CreditROICalculator, CreditMetrics


class ModelROIEvaluator:
    """模型ROI评估器"""

    def __init__(self, roi_calculator: CreditROICalculator = None):
        self.roi_calculator = roi_calculator or CreditROICalculator()
        self.default_params = {
            'interest_rate': 0.15,
            'term_months': 12,
            'recovery_rate': 0.3,
            'operational_cost_rate': 0.02,  # 2%的运营成本率
            'acquisition_cost': 500
        }

    def evaluate_model_roi(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        threshold: float = 0.5,
        **business_params
    ) -> Dict[str, float]:
        """
        评估模型在特定阈值下的ROI表现

        Args:
            y_true: 真实标签(1=违约, 0=正常)
            y_prob: 预测概率
            loan_amounts: 贷款金额
            threshold: 决策阈值
            **business_params: 业务参数

        Returns:
            ROI评估结果字典
        """
        params = {**self.default_params, **business_params}

        # 基于阈值的决策
        y_pred = (y_prob >= threshold).astype(int)

        # 计算混淆矩阵
        tp = np.sum((y_pred == 1) & (y_true == 1))  # 正确拒绝
        tn = np.sum((y_pred == 0) & (y_true == 0))  # 正确批准
        fp = np.sum((y_pred == 1) & (y_true == 0))  # 错误拒绝
        fn = np.sum((y_pred == 0) & (y_true == 1))  # 错误批准

        # 批准的贷款(预测为好客户)
        approved_mask = (y_pred == 0)
        approved_amounts = loan_amounts[approved_mask]
        approved_true_labels = y_true[approved_mask]

        if len(approved_amounts) == 0:
            return {
                'total_roi': -100.0,
                'approved_count': 0,
                'approval_rate': 0.0,
                'bad_rate_approved': 0.0,
                'total_revenue': 0.0,
                'total_loss': 0.0,
                'net_profit': -params['acquisition_cost'] * len(y_true)
            }

        # 计算批准客户的违约率
        bad_rate_approved = np.mean(approved_true_labels)

        # 计算收益
        total_principal = np.sum(approved_amounts)
        total_interest = total_principal * params['interest_rate']
        total_operational_cost = total_principal * params['operational_cost_rate']
        total_acquisition_cost = params['acquisition_cost'] * len(approved_amounts)

        # 计算损失
        expected_loss = np.sum(approved_amounts * approved_true_labels * (1 - params['recovery_rate']))

        # 净收益
        net_profit = total_interest - expected_loss - total_operational_cost - total_acquisition_cost

        # ROI计算
        total_investment = total_principal + total_acquisition_cost
        total_roi = (net_profit / total_investment * 100) if total_investment > 0 else -100.0

        return {
            'total_roi': total_roi,
            'approved_count': len(approved_amounts),
            'approval_rate': len(approved_amounts) / len(y_true),
            'bad_rate_approved': bad_rate_approved,
            'total_revenue': total_interest,
            'total_loss': expected_loss,
            'net_profit': net_profit,
            'total_principal': total_principal,
            'precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'recall': tn / (tn + fp) if (tn + fp) > 0 else 0
        }

    def threshold_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        thresholds: np.ndarray = None,
        **business_params
    ) -> pd.DataFrame:
        """
        阈值分析：评估不同阈值下的ROI表现

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            loan_amounts: 贷款金额
            thresholds: 阈值数组
            **business_params: 业务参数

        Returns:
            阈值分析结果DataFrame
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)

        results = []
        for threshold in thresholds:
            roi_metrics = self.evaluate_model_roi(
                y_true, y_prob, loan_amounts, threshold, **business_params
            )
            roi_metrics['threshold'] = threshold
            results.append(roi_metrics)

        return pd.DataFrame(results)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        objective: str = 'total_roi',
        **business_params
    ) -> Tuple[float, Dict[str, float]]:
        """
        寻找最优决策阈值

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            loan_amounts: 贷款金额
            objective: 优化目标('total_roi', 'net_profit', etc.)
            **business_params: 业务参数

        Returns:
            (最优阈值, 最优结果字典)
        """
        threshold_results = self.threshold_analysis(
            y_true, y_prob, loan_amounts, **business_params
        )

        optimal_idx = threshold_results[objective].idxmax()
        optimal_threshold = threshold_results.loc[optimal_idx, 'threshold']
        optimal_result = threshold_results.loc[optimal_idx].to_dict()

        return optimal_threshold, optimal_result

    def compare_models(
        self,
        models_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        loan_amounts: np.ndarray,
        threshold: float = 0.5,
        **business_params
    ) -> pd.DataFrame:
        """
        比较多个模型的ROI表现

        Args:
            models_results: 模型结果字典 {模型名: (y_prob, threshold)}
            y_true: 真实标签
            loan_amounts: 贷款金额
            threshold: 统一使用的阈值
            **business_params: 业务参数

        Returns:
            模型比较结果DataFrame
        """
        comparison_results = []

        for model_name, (y_prob, _) in models_results.items():
            roi_metrics = self.evaluate_model_roi(
                y_true, y_prob, loan_amounts, threshold, **business_params
            )
            roi_metrics['model'] = model_name
            comparison_results.append(roi_metrics)

        return pd.DataFrame(comparison_results)

    def segment_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        segments: np.ndarray,
        threshold: float = 0.5,
        **business_params
    ) -> pd.DataFrame:
        """
        分段分析：评估不同客群的ROI表现

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            loan_amounts: 贷款金额
            segments: 客群标识
            threshold: 决策阈值
            **business_params: 业务参数

        Returns:
            分段分析结果DataFrame
        """
        results = []
        unique_segments = np.unique(segments)

        for segment in unique_segments:
            segment_mask = (segments == segment)
            segment_y_true = y_true[segment_mask]
            segment_y_prob = y_prob[segment_mask]
            segment_amounts = loan_amounts[segment_mask]

            if len(segment_y_true) == 0:
                continue

            roi_metrics = self.evaluate_model_roi(
                segment_y_true, segment_y_prob, segment_amounts,
                threshold, **business_params
            )
            roi_metrics['segment'] = segment
            roi_metrics['sample_size'] = len(segment_y_true)
            results.append(roi_metrics)

        return pd.DataFrame(results)

    def profit_curve_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        **business_params
    ) -> Dict[str, np.ndarray]:
        """
        利润曲线分析

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            loan_amounts: 贷款金额
            **business_params: 业务参数

        Returns:
            利润曲线数据字典
        """
        params = {**self.default_params, **business_params}

        # 按预测概率排序
        sorted_indices = np.argsort(y_prob)[::-1]  # 降序排列
        sorted_y_true = y_true[sorted_indices]
        sorted_amounts = loan_amounts[sorted_indices]

        cumulative_profit = []
        cumulative_revenue = []
        cumulative_loss = []

        cumulative_principal = 0
        cumulative_cost = 0
        cumulative_expected_loss = 0

        for i in range(len(sorted_y_true)):
            amount = sorted_amounts[i]
            is_bad = sorted_y_true[i]

            # 累计本金
            cumulative_principal += amount

            # 累计收入
            interest = amount * params['interest_rate']
            operational_cost = amount * params['operational_cost_rate']
            acquisition_cost = params['acquisition_cost']

            cumulative_cost += operational_cost + acquisition_cost

            # 累计损失
            if is_bad:
                loss = amount * (1 - params['recovery_rate'])
                cumulative_expected_loss += loss

            # 累计收益
            total_interest = cumulative_principal * params['interest_rate']
            net_profit = total_interest - cumulative_expected_loss - cumulative_cost

            cumulative_profit.append(net_profit)
            cumulative_revenue.append(total_interest)
            cumulative_loss.append(cumulative_expected_loss)

        return {
            'cumulative_profit': np.array(cumulative_profit),
            'cumulative_revenue': np.array(cumulative_revenue),
            'cumulative_loss': np.array(cumulative_loss),
            'cumulative_customers': np.arange(1, len(cumulative_profit) + 1)
        }

    def plot_roi_analysis(
        self,
        threshold_results: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        绘制ROI分析图表

        Args:
            threshold_results: 阈值分析结果
            figsize: 图表大小

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model ROI Analysis', fontsize=16)

        # 1. ROI vs Threshold
        axes[0, 0].plot(threshold_results['threshold'], threshold_results['total_roi'])
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Total ROI (%)')
        axes[0, 0].set_title('ROI vs Threshold')
        axes[0, 0].grid(True)

        # 2. Approval Rate vs Threshold
        axes[0, 1].plot(threshold_results['threshold'], threshold_results['approval_rate'])
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Approval Rate')
        axes[0, 1].set_title('Approval Rate vs Threshold')
        axes[0, 1].grid(True)

        # 3. Bad Rate vs Threshold
        axes[0, 2].plot(threshold_results['threshold'], threshold_results['bad_rate_approved'])
        axes[0, 2].set_xlabel('Threshold')
        axes[0, 2].set_ylabel('Bad Rate (Approved)')
        axes[0, 2].set_title('Bad Rate vs Threshold')
        axes[0, 2].grid(True)

        # 4. Net Profit vs Threshold
        axes[1, 0].plot(threshold_results['threshold'], threshold_results['net_profit'])
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Net Profit')
        axes[1, 0].set_title('Net Profit vs Threshold')
        axes[1, 0].grid(True)

        # 5. ROI vs Approval Rate
        axes[1, 1].scatter(threshold_results['approval_rate'], threshold_results['total_roi'])
        axes[1, 1].set_xlabel('Approval Rate')
        axes[1, 1].set_ylabel('Total ROI (%)')
        axes[1, 1].set_title('ROI vs Approval Rate')
        axes[1, 1].grid(True)

        # 6. Risk-Return Trade-off
        axes[1, 2].scatter(threshold_results['bad_rate_approved'], threshold_results['total_roi'])
        axes[1, 2].set_xlabel('Bad Rate (Approved)')
        axes[1, 2].set_ylabel('Total ROI (%)')
        axes[1, 2].set_title('Risk-Return Trade-off')
        axes[1, 2].grid(True)

        plt.tight_layout()
        return fig

    def plot_profit_curve(
        self,
        profit_data: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        绘制利润曲线

        Args:
            profit_data: 利润曲线数据
            figsize: 图表大小

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        customers = profit_data['cumulative_customers']

        # 累计利润曲线
        axes[0].plot(customers, profit_data['cumulative_profit'], label='Cumulative Profit', linewidth=2)
        axes[0].plot(customers, profit_data['cumulative_revenue'], label='Cumulative Revenue', linestyle='--')
        axes[0].plot(customers, -profit_data['cumulative_loss'], label='Cumulative Loss', linestyle=':')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_xlabel('Number of Customers (Ranked by Score)')
        axes[0].set_ylabel('Amount')
        axes[0].set_title('Cumulative Profit Curve')
        axes[0].legend()
        axes[0].grid(True)

        # 利润率曲线
        profit_rate = profit_data['cumulative_profit'] / (customers * 1000)  # 假设平均每客户1000投资
        axes[1].plot(customers, profit_rate * 100)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('Number of Customers (Ranked by Score)')
        axes[1].set_ylabel('Profit Rate (%)')
        axes[1].set_title('Profit Rate Curve')
        axes[1].grid(True)

        plt.tight_layout()
        return fig

    def generate_roi_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        loan_amounts: np.ndarray,
        model_name: str = "Model",
        **business_params
    ) -> Dict[str, any]:
        """
        生成完整的ROI评估报告

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            loan_amounts: 贷款金额
            model_name: 模型名称
            **business_params: 业务参数

        Returns:
            完整的评估报告字典
        """
        # 阈值分析
        threshold_results = self.threshold_analysis(y_true, y_prob, loan_amounts, **business_params)

        # 最优阈值
        optimal_threshold, optimal_result = self.find_optimal_threshold(
            y_true, y_prob, loan_amounts, **business_params
        )

        # 利润曲线
        profit_data = self.profit_curve_analysis(y_true, y_prob, loan_amounts, **business_params)

        # 生成图表
        roi_fig = self.plot_roi_analysis(threshold_results)
        profit_fig = self.plot_profit_curve(profit_data)

        return {
            'model_name': model_name,
            'threshold_analysis': threshold_results,
            'optimal_threshold': optimal_threshold,
            'optimal_result': optimal_result,
            'profit_curve_data': profit_data,
            'roi_analysis_plot': roi_fig,
            'profit_curve_plot': profit_fig,
            'business_params': {**self.default_params, **business_params}
        }


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 生成模拟数据
    y_true = np.random.binomial(1, 0.15, n_samples)  # 15%违约率
    y_prob = np.random.beta(2, 8, n_samples)  # 模拟预测概率
    loan_amounts = np.random.normal(50000, 20000, n_samples)  # 贷款金额

    # 创建评估器
    evaluator = ModelROIEvaluator()

    # 运行完整分析
    report = evaluator.generate_roi_report(
        y_true, y_prob, loan_amounts,
        model_name="XGBoost Model",
        interest_rate=0.18,
        term_months=24
    )

    print(f"最优阈值: {report['optimal_threshold']:.3f}")
    print(f"最优ROI: {report['optimal_result']['total_roi']:.2f}%")
    print(f"批准率: {report['optimal_result']['approval_rate']:.2f}")