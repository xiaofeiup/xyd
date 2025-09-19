"""
特征漂移检测模块

提供特征分布漂移检测功能，主要基于PSI指标
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class FeatureDriftDetector:
    """
    特征漂移检测器

    使用PSI等统计指标检测特征分布的变化
    """

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 psi_threshold: float = 0.25,
                 bins: int = 10,
                 min_sample: int = 10):
        """
        初始化漂移检测器

        Parameters:
        -----------
        feature_names : list, optional
            要监控的特征名称列表
        psi_threshold : float, default=0.25
            PSI阈值
        bins : int, default=10
            分箱数量
        min_sample : int, default=10
            每个分箱最小样本数
        """
        self.feature_names = feature_names
        self.psi_threshold = psi_threshold
        self.bins = bins
        self.min_sample = min_sample
        self.baseline_data = None
        self.drift_history = []

    def set_baseline(self, baseline_data: pd.DataFrame) -> None:
        """
        设置基准数据

        Parameters:
        -----------
        baseline_data : pd.DataFrame
            基准数据
        """
        if self.feature_names is None:
            self.feature_names = baseline_data.select_dtypes(include=[np.number]).columns.tolist()

        self.baseline_data = baseline_data[self.feature_names].copy()

    def detect_drift(self,
                    current_data: pd.DataFrame,
                    timestamp: Optional[str] = None) -> Dict:
        """
        检测特征漂移

        Parameters:
        -----------
        current_data : pd.DataFrame
            当前数据
        timestamp : str, optional
            时间戳

        Returns:
        --------
        drift_result : dict
            漂移检测结果
        """
        if self.baseline_data is None:
            raise ValueError("请先设置基准数据")

        from ..evaluation.metrics import calculate_psi, calculate_multi_feature_psi

        # 计算PSI
        psi_results = calculate_multi_feature_psi(
            self.baseline_data,
            current_data[self.feature_names],
            features=self.feature_names,
            bins=self.bins,
            min_sample=self.min_sample
        )

        # 识别漂移特征
        drifted_features = psi_results[
            psi_results['psi_value'] >= self.psi_threshold
        ]

        # 生成漂移报告
        drift_alerts = []
        for _, feature in drifted_features.iterrows():
            severity = 'HIGH' if feature['psi_value'] > 0.5 else 'MEDIUM'
            drift_alerts.append({
                'feature': feature['feature'],
                'psi_value': feature['psi_value'],
                'interpretation': feature['interpretation'],
                'severity': severity
            })

        # 构建结果
        result = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'total_features': len(psi_results),
            'drifted_features': len(drifted_features),
            'drift_rate': len(drifted_features) / len(psi_results),
            'psi_summary': psi_results.to_dict('records'),
            'drift_alerts': drift_alerts,
            'overall_status': self._get_overall_status(len(drifted_features), len(psi_results))
        }

        # 记录历史
        self.drift_history.append(result)

        return result

    def _get_overall_status(self, drifted_count: int, total_count: int) -> str:
        """
        获取总体状态

        Parameters:
        -----------
        drifted_count : int
            漂移特征数量
        total_count : int
            总特征数量

        Returns:
        --------
        status : str
            状态描述
        """
        drift_rate = drifted_count / total_count if total_count > 0 else 0

        if drift_rate == 0:
            return "STABLE"
        elif drift_rate <= 0.1:
            return "MINOR_DRIFT"
        elif drift_rate <= 0.3:
            return "MODERATE_DRIFT"
        else:
            return "SEVERE_DRIFT"

    def get_drift_trend(self, days: int = 30) -> pd.DataFrame:
        """
        获取漂移趋势

        Parameters:
        -----------
        days : int, default=30
            趋势天数

        Returns:
        --------
        trend_df : pd.DataFrame
            漂移趋势数据
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()

        recent_records = [
            record for record in self.drift_history
            if record['timestamp'] >= cutoff_str
        ]

        if not recent_records:
            return pd.DataFrame()

        trend_data = []
        for record in recent_records:
            trend_data.append({
                'timestamp': record['timestamp'],
                'total_features': record['total_features'],
                'drifted_features': record['drifted_features'],
                'drift_rate': record['drift_rate'],
                'status': record['overall_status']
            })

        return pd.DataFrame(trend_data)

    def get_feature_stability_report(self) -> pd.DataFrame:
        """
        获取特征稳定性报告

        Returns:
        --------
        stability_report : pd.DataFrame
            特征稳定性报告
        """
        if not self.drift_history:
            return pd.DataFrame()

        # 聚合所有特征的PSI历史
        all_psi_data = []
        for record in self.drift_history:
            for feature_psi in record['psi_summary']:
                all_psi_data.append({
                    'timestamp': record['timestamp'],
                    'feature': feature_psi['feature'],
                    'psi_value': feature_psi['psi_value'],
                    'interpretation': feature_psi['interpretation']
                })

        if not all_psi_data:
            return pd.DataFrame()

        psi_df = pd.DataFrame(all_psi_data)

        # 计算每个特征的稳定性统计
        stability_stats = []
        for feature in psi_df['feature'].unique():
            feature_data = psi_df[psi_df['feature'] == feature]

            stability_stats.append({
                'feature': feature,
                'avg_psi': feature_data['psi_value'].mean(),
                'max_psi': feature_data['psi_value'].max(),
                'min_psi': feature_data['psi_value'].min(),
                'psi_std': feature_data['psi_value'].std(),
                'drift_episodes': (feature_data['psi_value'] >= self.psi_threshold).sum(),
                'stability_score': self._calculate_stability_score(feature_data['psi_value'])
            })

        return pd.DataFrame(stability_stats).sort_values('avg_psi', ascending=False)

    def _calculate_stability_score(self, psi_values: pd.Series) -> float:
        """
        计算稳定性评分

        Parameters:
        -----------
        psi_values : pd.Series
            PSI值序列

        Returns:
        --------
        score : float
            稳定性评分 (0-100)
        """
        avg_psi = psi_values.mean()
        psi_volatility = psi_values.std()

        # 基础评分：PSI越低越好
        base_score = max(0, 100 - avg_psi * 200)

        # 波动性惩罚：波动越大扣分越多
        volatility_penalty = min(50, psi_volatility * 100)

        return max(0, base_score - volatility_penalty)


class SingleFeatureDriftMonitor:
    """
    单特征漂移监控器

    专门监控单个特征的分布变化
    """

    def __init__(self,
                 feature_name: str,
                 bins: int = 10,
                 min_sample: int = 10):
        """
        初始化单特征监控器

        Parameters:
        -----------
        feature_name : str
            特征名称
        bins : int, default=10
            分箱数量
        min_sample : int, default=10
            最小样本数
        """
        self.feature_name = feature_name
        self.bins = bins
        self.min_sample = min_sample
        self.baseline_data = None
        self.monitoring_history = []

    def set_baseline(self, baseline_values: np.ndarray) -> None:
        """
        设置基准分布

        Parameters:
        -----------
        baseline_values : array-like
            基准特征值
        """
        self.baseline_data = np.array(baseline_values)

    def monitor_drift(self,
                     current_values: np.ndarray,
                     timestamp: Optional[str] = None) -> Dict:
        """
        监控特征漂移

        Parameters:
        -----------
        current_values : array-like
            当前特征值
        timestamp : str, optional
            时间戳

        Returns:
        --------
        monitor_result : dict
            监控结果
        """
        if self.baseline_data is None:
            raise ValueError("请先设置基准数据")

        from ..evaluation.metrics import calculate_psi

        psi_value, psi_detail = calculate_psi(
            self.baseline_data,
            current_values,
            bins=self.bins,
            min_sample=self.min_sample,
            feature_name=self.feature_name
        )

        # 生成监控结果
        result = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'feature_name': self.feature_name,
            'psi_value': psi_value,
            'interpretation': self._interpret_psi(psi_value),
            'baseline_stats': {
                'mean': np.mean(self.baseline_data),
                'std': np.std(self.baseline_data),
                'size': len(self.baseline_data)
            },
            'current_stats': {
                'mean': np.mean(current_values),
                'std': np.std(current_values),
                'size': len(current_values)
            },
            'psi_detail': psi_detail.to_dict('records') if psi_detail is not None else None
        }

        self.monitoring_history.append(result)
        return result

    def _interpret_psi(self, psi_value: float) -> str:
        """
        解释PSI值

        Parameters:
        -----------
        psi_value : float
            PSI值

        Returns:
        --------
        interpretation : str
            解释
        """
        if pd.isna(psi_value):
            return "无效值"
        elif psi_value < 0.1:
            return "稳定 (变化很小)"
        elif psi_value < 0.2:
            return "轻微变化"
        elif psi_value < 0.25:
            return "中等变化 (需要关注)"
        else:
            return "显著变化 (需要重新建模)"

    def get_drift_history(self) -> pd.DataFrame:
        """
        获取漂移历史

        Returns:
        --------
        history_df : pd.DataFrame
            漂移历史数据
        """
        if not self.monitoring_history:
            return pd.DataFrame()

        history_data = []
        for record in self.monitoring_history:
            history_data.append({
                'timestamp': record['timestamp'],
                'psi_value': record['psi_value'],
                'interpretation': record['interpretation'],
                'baseline_mean': record['baseline_stats']['mean'],
                'current_mean': record['current_stats']['mean'],
                'baseline_std': record['baseline_stats']['std'],
                'current_std': record['current_stats']['std']
            })

        return pd.DataFrame(history_data)