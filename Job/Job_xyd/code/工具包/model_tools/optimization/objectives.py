"""
自定义目标函数

提供多种优化目标函数，包括：
- 业务指标优化（ROI、利润最大化）
- 多目标优化（准确率+稳定性）
- 信贷特定指标（KS、AUC、Lift等）
- 自定义评估函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Union, Optional, Any
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import warnings


class ObjectiveFunction:
    """目标函数基类"""

    def __init__(self, direction: str = 'maximize'):
        """
        初始化目标函数

        Args:
            direction: 优化方向 ('maximize', 'minimize')
        """
        self.direction = direction

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算目标函数值"""
        raise NotImplementedError


class AUCObjective(ObjectiveFunction):
    """AUC目标函数"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError as e:
            warnings.warn(f"AUC计算失败: {e}")
            return 0.0


class KSObjective(ObjectiveFunction):
    """KS目标函数"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算KS值"""
        try:
            # 创建DataFrame并排序
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            df = df.sort_values('y_pred', ascending=False)

            # 计算累积统计
            df['bad'] = df['y_true']
            df['good'] = 1 - df['y_true']

            total_bad = df['bad'].sum()
            total_good = df['good'].sum()

            if total_bad == 0 or total_good == 0:
                return 0.0

            df['cum_bad_rate'] = df['bad'].cumsum() / total_bad
            df['cum_good_rate'] = df['good'].cumsum() / total_good

            # 计算KS值
            ks = (df['cum_bad_rate'] - df['cum_good_rate']).max()
            return ks

        except Exception as e:
            warnings.warn(f"KS计算失败: {e}")
            return 0.0


class LiftObjective(ObjectiveFunction):
    """Lift目标函数（Top 10%的Lift值）"""

    def __init__(self, top_percentile: float = 0.1, direction: str = 'maximize'):
        """
        初始化Lift目标函数

        Args:
            top_percentile: 顶部百分比（默认10%）
            direction: 优化方向
        """
        super().__init__(direction)
        self.top_percentile = top_percentile

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算Top percentile的Lift值"""
        try:
            # 整体bad rate
            overall_bad_rate = np.mean(y_true)
            if overall_bad_rate == 0:
                return 0.0

            # 按预测概率排序
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            df = df.sort_values('y_pred', ascending=False)

            # 计算top percentile的bad rate
            top_n = int(len(df) * self.top_percentile)
            if top_n == 0:
                top_n = 1

            top_bad_rate = df.head(top_n)['y_true'].mean()

            # 计算Lift
            lift = top_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0
            return lift

        except Exception as e:
            warnings.warn(f"Lift计算失败: {e}")
            return 0.0


class F1Objective(ObjectiveFunction):
    """F1分数目标函数"""

    def __init__(self, threshold: float = 0.5, direction: str = 'maximize'):
        """
        初始化F1目标函数

        Args:
            threshold: 分类阈值
            direction: 优化方向
        """
        super().__init__(direction)
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            y_pred_binary = (y_pred >= self.threshold).astype(int)
            return f1_score(y_true, y_pred_binary)
        except Exception as e:
            warnings.warn(f"F1计算失败: {e}")
            return 0.0


class PrecisionRecallObjective(ObjectiveFunction):
    """精确率-召回率平衡目标函数"""

    def __init__(self, precision_weight: float = 0.5, threshold: float = 0.5, direction: str = 'maximize'):
        """
        初始化精确率-召回率目标函数

        Args:
            precision_weight: 精确率权重（0-1），召回率权重为1-precision_weight
            threshold: 分类阈值
            direction: 优化方向
        """
        super().__init__(direction)
        self.precision_weight = precision_weight
        self.recall_weight = 1 - precision_weight
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            y_pred_binary = (y_pred >= self.threshold).astype(int)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)

            return self.precision_weight * precision + self.recall_weight * recall

        except Exception as e:
            warnings.warn(f"精确率-召回率计算失败: {e}")
            return 0.0


class BusinessROIObjective(ObjectiveFunction):
    """业务ROI目标函数"""

    def __init__(
        self,
        loan_amounts: np.ndarray,
        interest_rate: float = 0.15,
        operational_cost_rate: float = 0.02,
        recovery_rate: float = 0.3,
        threshold: float = 0.5,
        direction: str = 'maximize'
    ):
        """
        初始化业务ROI目标函数

        Args:
            loan_amounts: 贷款金额数组
            interest_rate: 年利率
            operational_cost_rate: 运营成本率
            recovery_rate: 回收率
            threshold: 决策阈值
            direction: 优化方向
        """
        super().__init__(direction)
        self.loan_amounts = loan_amounts
        self.interest_rate = interest_rate
        self.operational_cost_rate = operational_cost_rate
        self.recovery_rate = recovery_rate
        self.threshold = threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算ROI"""
        try:
            # 批准决策（预测为好客户）
            approved_mask = (y_pred >= self.threshold)
            if not np.any(approved_mask):
                return -1.0  # 如果不批准任何客户，ROI为负

            approved_amounts = self.loan_amounts[approved_mask]
            approved_labels = y_true[approved_mask]

            # 计算收入
            total_principal = np.sum(approved_amounts)
            total_interest = total_principal * self.interest_rate

            # 计算成本
            operational_cost = total_principal * self.operational_cost_rate
            expected_loss = np.sum(approved_amounts * approved_labels * (1 - self.recovery_rate))

            # 计算ROI
            net_profit = total_interest - operational_cost - expected_loss
            roi = net_profit / total_principal if total_principal > 0 else -1.0

            return roi

        except Exception as e:
            warnings.warn(f"ROI计算失败: {e}")
            return -1.0


class StabilityAwareObjective(ObjectiveFunction):
    """稳定性感知目标函数（AUC + 稳定性惩罚）"""

    def __init__(
        self,
        base_objective: ObjectiveFunction,
        stability_weight: float = 0.1,
        direction: str = 'maximize'
    ):
        """
        初始化稳定性感知目标函数

        Args:
            base_objective: 基础目标函数
            stability_weight: 稳定性权重
            direction: 优化方向
        """
        super().__init__(direction)
        self.base_objective = base_objective
        self.stability_weight = stability_weight

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算稳定性感知的目标函数值"""
        try:
            # 基础目标值
            base_score = self.base_objective(y_true, y_pred, **kwargs)

            # 稳定性评估（预测分布的稳定性）
            pred_std = np.std(y_pred)
            # 稳定性惩罚：预测方差过大时给予惩罚
            stability_penalty = min(pred_std / 0.2, 1.0)  # 标准化到0-1

            # 综合得分
            final_score = base_score - self.stability_weight * stability_penalty

            return final_score

        except Exception as e:
            warnings.warn(f"稳定性感知目标计算失败: {e}")
            return self.base_objective(y_true, y_pred, **kwargs)


class MultiObjective(ObjectiveFunction):
    """多目标优化函数"""

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        weights: List[float],
        direction: str = 'maximize'
    ):
        """
        初始化多目标函数

        Args:
            objectives: 目标函数列表
            weights: 权重列表
            direction: 优化方向
        """
        super().__init__(direction)
        if len(objectives) != len(weights):
            raise ValueError("目标函数数量与权重数量不匹配")

        self.objectives = objectives
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算加权多目标值"""
        try:
            scores = []
            for objective in self.objectives:
                score = objective(y_true, y_pred, **kwargs)
                scores.append(score)

            weighted_score = np.sum(np.array(scores) * self.weights)
            return weighted_score

        except Exception as e:
            warnings.warn(f"多目标计算失败: {e}")
            return 0.0


class CustomObjective(ObjectiveFunction):
    """自定义目标函数包装器"""

    def __init__(
        self,
        custom_func: Callable[[np.ndarray, np.ndarray], float],
        direction: str = 'maximize'
    ):
        """
        初始化自定义目标函数

        Args:
            custom_func: 自定义函数，接受(y_true, y_pred)返回分数
            direction: 优化方向
        """
        super().__init__(direction)
        self.custom_func = custom_func

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """调用自定义函数"""
        try:
            return self.custom_func(y_true, y_pred)
        except Exception as e:
            warnings.warn(f"自定义目标函数计算失败: {e}")
            return 0.0


# 回归任务目标函数
class RegressionObjective(ObjectiveFunction):
    """回归目标函数基类"""

    def __init__(self, direction: str = 'minimize'):
        super().__init__(direction)


class RMSEObjective(RegressionObjective):
    """RMSE目标函数"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            warnings.warn(f"RMSE计算失败: {e}")
            return float('inf')


class MAEObjective(RegressionObjective):
    """MAE目标函数"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            return mean_absolute_error(y_true, y_pred)
        except Exception as e:
            warnings.warn(f"MAE计算失败: {e}")
            return float('inf')


class R2Objective(RegressionObjective):
    """R²目标函数"""

    def __init__(self, direction: str = 'maximize'):
        super().__init__(direction)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        try:
            return r2_score(y_true, y_pred)
        except Exception as e:
            warnings.warn(f"R²计算失败: {e}")
            return -float('inf')


# 预定义目标函数
OBJECTIVE_FUNCTIONS = {
    'auc': AUCObjective(),
    'ks': KSObjective(),
    'lift': LiftObjective(),
    'f1': F1Objective(),
    'rmse': RMSEObjective(),
    'mae': MAEObjective(),
    'r2': R2Objective()
}


def get_objective_function(name: str) -> ObjectiveFunction:
    """
    获取预定义的目标函数

    Args:
        name: 目标函数名称

    Returns:
        目标函数对象
    """
    if name not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"不支持的目标函数: {name}")

    return OBJECTIVE_FUNCTIONS[name]


def create_credit_scoring_objective(
    loan_amounts: np.ndarray,
    primary_metric: str = 'auc',
    business_weight: float = 0.3,
    **business_params
) -> MultiObjective:
    """
    创建信贷评分专用的多目标函数

    Args:
        loan_amounts: 贷款金额
        primary_metric: 主要指标 ('auc', 'ks', 'lift')
        business_weight: 业务指标权重
        **business_params: 业务参数

    Returns:
        多目标函数
    """
    # 主要模型指标
    primary_obj = get_objective_function(primary_metric)

    # 业务ROI目标
    roi_obj = BusinessROIObjective(loan_amounts, **business_params)

    # 组合目标
    multi_obj = MultiObjective(
        objectives=[primary_obj, roi_obj],
        weights=[1 - business_weight, business_weight]
    )

    return multi_obj


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.2, 1000)
    y_pred = np.random.random(1000)
    loan_amounts = np.random.normal(50000, 20000, 1000)

    # 测试不同目标函数
    auc_obj = AUCObjective()
    print(f"AUC: {auc_obj(y_true, y_pred):.4f}")

    ks_obj = KSObjective()
    print(f"KS: {ks_obj(y_true, y_pred):.4f}")

    lift_obj = LiftObjective()
    print(f"Lift: {lift_obj(y_true, y_pred):.4f}")

    # 业务ROI目标
    roi_obj = BusinessROIObjective(loan_amounts)
    print(f"ROI: {roi_obj(y_true, y_pred):.4f}")

    # 多目标
    multi_obj = create_credit_scoring_objective(loan_amounts)
    print(f"Multi-objective: {multi_obj(y_true, y_pred):.4f}")