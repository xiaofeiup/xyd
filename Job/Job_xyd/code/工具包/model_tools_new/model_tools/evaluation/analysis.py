"""
深度分析模块

提供模型的深度分析和诊断功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from scipy import stats
from datetime import datetime, timedelta
import logging


class ModelAnalyzer:
    """
    模型深度分析器

    提供模型性能的深度分析和诊断功能
    """

    def __init__(self, model_name: str = "model"):
        """
        初始化模型分析器

        Parameters:
        -----------
        model_name : str, default="model"
            模型名称
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"ModelAnalyzer_{model_name}")

    def analyze_model_performance(self,
                                y_true: np.ndarray,
                                y_scores: np.ndarray,
                                sample_weights: Optional[np.ndarray] = None) -> Dict:
        """
        分析模型性能

        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_scores : np.ndarray
            预测概率
        sample_weights : np.ndarray, optional
            样本权重

        Returns:
        --------
        analysis_result : dict
            性能分析结果
        """
        analysis_result = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'basic_analysis': self._analyze_basic_performance(y_true, y_scores),
            'distribution_analysis': self._analyze_score_distribution(y_true, y_scores),
            'stability_analysis': self._analyze_score_stability(y_true, y_scores),
            'segment_analysis': self._analyze_performance_by_segments(y_true, y_scores),
            'outlier_analysis': self._analyze_outliers(y_true, y_scores),
            'calibration_analysis': self._analyze_calibration(y_true, y_scores)
        }

        return analysis_result

    def analyze_feature_impact(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             y_scores: np.ndarray,
                             top_k: int = 10) -> Dict:
        """
        分析特征影响

        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        y_scores : np.ndarray
            预测概率
        top_k : int, default=10
            分析前k个重要特征

        Returns:
        --------
        feature_analysis : dict
            特征影响分析结果
        """
        feature_analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(X.columns),
            'analyzed_features': min(top_k, len(X.columns)),
            'feature_impact': {},
            'feature_interactions': {},
            'feature_stability': {}
        }

        # 计算每个特征的影响
        for feature in X.columns[:top_k]:
            feature_analysis['feature_impact'][feature] = self._analyze_single_feature_impact(
                X[feature], y, y_scores
            )

        # 分析特征交互
        if len(X.columns) >= 2:
            feature_analysis['feature_interactions'] = self._analyze_feature_interactions(
                X.iloc[:, :min(5, len(X.columns))], y
            )

        # 分析特征稳定性
        feature_analysis['feature_stability'] = self._analyze_feature_stability(X)

        return feature_analysis

    def analyze_model_degradation(self,
                                historical_performance: List[Dict],
                                current_performance: Dict,
                                lookback_periods: int = 10) -> Dict:
        """
        分析模型退化

        Parameters:
        -----------
        historical_performance : list
            历史性能数据
        current_performance : dict
            当前性能数据
        lookback_periods : int, default=10
            回看期数

        Returns:
        --------
        degradation_analysis : dict
            退化分析结果
        """
        degradation_analysis = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': lookback_periods,
            'degradation_status': 'unknown',
            'performance_trends': {},
            'anomaly_detection': {},
            'recommendations': []
        }

        if len(historical_performance) < 2:
            degradation_analysis['degradation_status'] = 'insufficient_data'
            return degradation_analysis

        # 分析性能趋势
        degradation_analysis['performance_trends'] = self._analyze_performance_trends(
            historical_performance[-lookback_periods:], current_performance
        )

        # 异常检测
        degradation_analysis['anomaly_detection'] = self._detect_performance_anomalies(
            historical_performance, current_performance
        )

        # 评估整体退化状态
        degradation_analysis['degradation_status'] = self._assess_degradation_status(
            degradation_analysis['performance_trends'],
            degradation_analysis['anomaly_detection']
        )

        # 生成建议
        degradation_analysis['recommendations'] = self._generate_degradation_recommendations(
            degradation_analysis
        )

        return degradation_analysis

    def analyze_data_drift_impact(self,
                                baseline_data: pd.DataFrame,
                                current_data: pd.DataFrame,
                                target_col: str,
                                score_col: str) -> Dict:
        """
        分析数据漂移对模型的影响

        Parameters:
        -----------
        baseline_data : pd.DataFrame
            基准数据
        current_data : pd.DataFrame
            当前数据
        target_col : str
            目标变量列名
        score_col : str
            预测得分列名

        Returns:
        --------
        drift_impact_analysis : dict
            漂移影响分析结果
        """
        drift_impact_analysis = {
            'timestamp': datetime.now().isoformat(),
            'baseline_size': len(baseline_data),
            'current_size': len(current_data),
            'population_drift': {},
            'performance_impact': {},
            'feature_drift_impact': {},
            'risk_assessment': {}
        }

        # 分析总体分布漂移
        drift_impact_analysis['population_drift'] = self._analyze_population_drift(
            baseline_data, current_data, target_col, score_col
        )

        # 分析性能影响
        if target_col in current_data.columns and score_col in current_data.columns:
            drift_impact_analysis['performance_impact'] = self._analyze_drift_performance_impact(
                baseline_data[target_col], baseline_data[score_col],
                current_data[target_col], current_data[score_col]
            )

        # 分析特征级别漂移影响
        feature_cols = [col for col in baseline_data.columns if col not in [target_col, score_col]]
        drift_impact_analysis['feature_drift_impact'] = self._analyze_feature_drift_impact(
            baseline_data[feature_cols], current_data[feature_cols]
        )

        # 风险评估
        drift_impact_analysis['risk_assessment'] = self._assess_drift_risk(
            drift_impact_analysis
        )

        return drift_impact_analysis

    def generate_model_health_report(self,
                                   performance_analysis: Dict,
                                   feature_analysis: Optional[Dict] = None,
                                   degradation_analysis: Optional[Dict] = None,
                                   drift_analysis: Optional[Dict] = None) -> Dict:
        """
        生成模型健康报告

        Parameters:
        -----------
        performance_analysis : dict
            性能分析结果
        feature_analysis : dict, optional
            特征分析结果
        degradation_analysis : dict, optional
            退化分析结果
        drift_analysis : dict, optional
            漂移分析结果

        Returns:
        --------
        health_report : dict
            模型健康报告
        """
        health_report = {
            'model_name': self.model_name,
            'report_timestamp': datetime.now().isoformat(),
            'overall_health_score': 0,
            'health_dimensions': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'detailed_analysis': {
                'performance': performance_analysis,
                'features': feature_analysis,
                'degradation': degradation_analysis,
                'drift': drift_analysis
            }
        }

        # 计算各维度健康分数
        health_scores = {}

        # 性能维度
        health_scores['performance'] = self._calculate_performance_health_score(performance_analysis)

        # 稳定性维度
        if degradation_analysis:
            health_scores['stability'] = self._calculate_stability_health_score(degradation_analysis)

        # 特征维度
        if feature_analysis:
            health_scores['features'] = self._calculate_feature_health_score(feature_analysis)

        # 数据质量维度
        if drift_analysis:
            health_scores['data_quality'] = self._calculate_data_quality_health_score(drift_analysis)

        health_report['health_dimensions'] = health_scores

        # 计算总体健康分数
        if health_scores:
            health_report['overall_health_score'] = np.mean(list(health_scores.values()))

        # 识别关键问题和警告
        health_report['critical_issues'], health_report['warnings'] = self._identify_issues_and_warnings(
            health_report['detailed_analysis']
        )

        # 生成建议
        health_report['recommendations'] = self._generate_health_recommendations(
            health_report
        )

        return health_report

    def _analyze_basic_performance(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """分析基础性能指标"""
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        # 计算基础指标
        auc_score = roc_auc_score(y_true, y_scores)

        # 计算最优阈值
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_idx] if optimal_threshold_idx < len(thresholds) else 0.5

        # 使用最优阈值计算其他指标
        y_pred = (y_scores >= optimal_threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        return {
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'pr_auc': auc(recall, precision)
        }

    def _analyze_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """分析评分分布"""
        good_scores = y_scores[y_true == 0]
        bad_scores = y_scores[y_true == 1]

        # 计算KS统计量
        ks_statistic, ks_p_value = stats.ks_2samp(good_scores, bad_scores)

        return {
            'good_samples': {
                'count': len(good_scores),
                'mean': np.mean(good_scores),
                'std': np.std(good_scores),
                'median': np.median(good_scores),
                'q25': np.percentile(good_scores, 25),
                'q75': np.percentile(good_scores, 75)
            },
            'bad_samples': {
                'count': len(bad_scores),
                'mean': np.mean(bad_scores),
                'std': np.std(bad_scores),
                'median': np.median(bad_scores),
                'q25': np.percentile(bad_scores, 25),
                'q75': np.percentile(bad_scores, 75)
            },
            'separation_metrics': {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mean_difference': np.mean(bad_scores) - np.mean(good_scores),
                'separation_power': 'good' if ks_statistic > 0.2 else 'poor'
            }
        }

    def _analyze_score_stability(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """分析评分稳定性"""
        # 将数据分成时间段进行稳定性分析（假设数据按时间排序）
        n_segments = min(5, len(y_scores) // 100)  # 至少100个样本一段
        segment_size = len(y_scores) // n_segments

        segment_aucs = []
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < n_segments - 1 else len(y_scores)

            segment_y_true = y_true[start_idx:end_idx]
            segment_y_scores = y_scores[start_idx:end_idx]

            if len(np.unique(segment_y_true)) > 1:  # 确保有正负样本
                from sklearn.metrics import roc_auc_score
                segment_auc = roc_auc_score(segment_y_true, segment_y_scores)
                segment_aucs.append(segment_auc)

        stability_metrics = {
            'n_segments': len(segment_aucs),
            'auc_stability': {
                'mean': np.mean(segment_aucs) if segment_aucs else 0,
                'std': np.std(segment_aucs) if segment_aucs else 0,
                'min': np.min(segment_aucs) if segment_aucs else 0,
                'max': np.max(segment_aucs) if segment_aucs else 0,
                'coefficient_of_variation': np.std(segment_aucs) / np.mean(segment_aucs) if segment_aucs and np.mean(segment_aucs) > 0 else 0
            }
        }

        # 评估稳定性
        cv = stability_metrics['auc_stability']['coefficient_of_variation']
        if cv < 0.05:
            stability_metrics['stability_assessment'] = 'very_stable'
        elif cv < 0.1:
            stability_metrics['stability_assessment'] = 'stable'
        elif cv < 0.2:
            stability_metrics['stability_assessment'] = 'moderately_stable'
        else:
            stability_metrics['stability_assessment'] = 'unstable'

        return stability_metrics

    def _analyze_performance_by_segments(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """按分段分析性能"""
        # 按评分分段分析
        segments = []
        n_segments = 10

        for i in range(n_segments):
            lower_percentile = i * 10
            upper_percentile = (i + 1) * 10

            lower_threshold = np.percentile(y_scores, lower_percentile)
            upper_threshold = np.percentile(y_scores, upper_percentile)

            mask = (y_scores >= lower_threshold) & (y_scores < upper_threshold)
            if i == n_segments - 1:  # 最后一段包含上边界
                mask = (y_scores >= lower_threshold) & (y_scores <= upper_threshold)

            segment_y_true = y_true[mask]
            segment_y_scores = y_scores[mask]

            if len(segment_y_true) > 0:
                segment_info = {
                    'segment': f'{lower_percentile}-{upper_percentile}%',
                    'sample_count': len(segment_y_true),
                    'positive_rate': np.mean(segment_y_true),
                    'mean_score': np.mean(segment_y_scores),
                    'score_range': [lower_threshold, upper_threshold]
                }

                # 计算lift
                overall_positive_rate = np.mean(y_true)
                segment_info['lift'] = segment_info['positive_rate'] / overall_positive_rate if overall_positive_rate > 0 else 0

                segments.append(segment_info)

        return {
            'segments': segments,
            'segment_analysis': {
                'consistent_lift_trend': self._check_lift_trend_consistency(segments),
                'high_performing_segments': [s for s in segments if s['lift'] > 1.5],
                'low_performing_segments': [s for s in segments if s['lift'] < 0.5]
            }
        }

    def _analyze_outliers(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """分析异常值"""
        # 识别评分异常值
        q1, q3 = np.percentile(y_scores, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (y_scores < lower_bound) | (y_scores > upper_bound)
        outlier_scores = y_scores[outlier_mask]
        outlier_labels = y_true[outlier_mask]

        return {
            'outlier_detection': {
                'method': 'IQR',
                'outlier_count': np.sum(outlier_mask),
                'outlier_percentage': np.sum(outlier_mask) / len(y_scores) * 100,
                'outlier_bounds': [lower_bound, upper_bound]
            },
            'outlier_analysis': {
                'outlier_positive_rate': np.mean(outlier_labels) if len(outlier_labels) > 0 else 0,
                'normal_positive_rate': np.mean(y_true[~outlier_mask]) if np.sum(~outlier_mask) > 0 else 0,
                'outlier_score_stats': {
                    'mean': np.mean(outlier_scores) if len(outlier_scores) > 0 else 0,
                    'std': np.std(outlier_scores) if len(outlier_scores) > 0 else 0
                }
            }
        }

    def _analyze_calibration(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """分析模型校准度"""
        # 分箱分析校准度
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        bin_calibration = []
        for i in range(n_bins):
            mask = (y_scores >= bin_boundaries[i]) & (y_scores < bin_boundaries[i + 1])
            if i == n_bins - 1:  # 最后一个箱子包含上边界
                mask = (y_scores >= bin_boundaries[i]) & (y_scores <= bin_boundaries[i + 1])

            if np.sum(mask) > 0:
                bin_true_rate = np.mean(y_true[mask])
                bin_predicted_rate = np.mean(y_scores[mask])
                bin_count = np.sum(mask)

                bin_calibration.append({
                    'bin': i + 1,
                    'bin_range': [bin_boundaries[i], bin_boundaries[i + 1]],
                    'sample_count': bin_count,
                    'predicted_rate': bin_predicted_rate,
                    'observed_rate': bin_true_rate,
                    'calibration_error': abs(bin_predicted_rate - bin_true_rate)
                })

        # 计算总体校准误差
        if bin_calibration:
            weighted_calibration_error = np.average(
                [bin_info['calibration_error'] for bin_info in bin_calibration],
                weights=[bin_info['sample_count'] for bin_info in bin_calibration]
            )
        else:
            weighted_calibration_error = 0

        return {
            'bin_calibration': bin_calibration,
            'overall_calibration': {
                'weighted_calibration_error': weighted_calibration_error,
                'calibration_assessment': 'well_calibrated' if weighted_calibration_error < 0.1 else 'poorly_calibrated'
            }
        }

    def _analyze_single_feature_impact(self, feature_values: pd.Series, y: pd.Series, y_scores: np.ndarray) -> Dict:
        """分析单个特征的影响"""
        # 计算特征与目标变量的相关性
        correlation_with_target = feature_values.corr(y)

        # 计算特征与预测概率的相关性
        correlation_with_score = feature_values.corr(pd.Series(y_scores))

        # 分析特征分布
        feature_stats = {
            'mean': feature_values.mean(),
            'std': feature_values.std(),
            'min': feature_values.min(),
            'max': feature_values.max(),
            'missing_rate': feature_values.isnull().mean(),
            'unique_values': feature_values.nunique()
        }

        # 特征重要性评估（基于相关性）
        importance_score = abs(correlation_with_target) * abs(correlation_with_score)

        return {
            'feature_stats': feature_stats,
            'correlations': {
                'with_target': correlation_with_target,
                'with_prediction': correlation_with_score
            },
            'importance_score': importance_score,
            'feature_type': 'categorical' if feature_values.nunique() < 20 else 'numerical'
        }

    def _analyze_feature_interactions(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """分析特征交互"""
        feature_interactions = {}
        features = X.columns.tolist()

        # 分析两两特征间的相关性
        correlation_matrix = X.corr()

        # 找出高度相关的特征对
        high_correlation_pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # 高相关性阈值
                    high_correlation_pairs.append({
                        'feature1': features[i],
                        'feature2': features[j],
                        'correlation': corr
                    })

        feature_interactions['high_correlation_pairs'] = high_correlation_pairs
        feature_interactions['max_correlation'] = correlation_matrix.abs().max().max()
        feature_interactions['mean_correlation'] = correlation_matrix.abs().mean().mean()

        return feature_interactions

    def _analyze_feature_stability(self, X: pd.DataFrame) -> Dict:
        """分析特征稳定性"""
        stability_analysis = {}

        for feature in X.columns:
            feature_data = X[feature].dropna()

            if len(feature_data) > 0:
                # 计算变异系数
                cv = feature_data.std() / feature_data.mean() if feature_data.mean() != 0 else np.inf

                # 计算单一值占比
                value_counts = feature_data.value_counts()
                max_value_ratio = value_counts.iloc[0] / len(feature_data) if len(value_counts) > 0 else 0

                stability_analysis[feature] = {
                    'coefficient_of_variation': cv,
                    'max_value_ratio': max_value_ratio,
                    'unique_ratio': feature_data.nunique() / len(feature_data),
                    'stability_score': 1 - max_value_ratio,  # 简单的稳定性评分
                    'stability_level': self._assess_feature_stability_level(max_value_ratio)
                }

        return stability_analysis

    def _analyze_performance_trends(self, historical_data: List[Dict], current_data: Dict) -> Dict:
        """分析性能趋势"""
        if not historical_data:
            return {'status': 'no_historical_data'}

        # 提取关键指标的时间序列
        metrics = ['auc', 'ks', 'precision', 'recall']
        trends = {}

        for metric in metrics:
            historical_values = []
            for record in historical_data:
                if metric in record.get('basic_metrics', {}):
                    historical_values.append(record['basic_metrics'][metric])

            if historical_values and metric in current_data.get('basic_metrics', {}):
                current_value = current_data['basic_metrics'][metric]
                historical_mean = np.mean(historical_values)
                historical_std = np.std(historical_values)

                # 计算趋势
                if len(historical_values) >= 3:
                    x = np.arange(len(historical_values))
                    slope, _, _, _, _ = stats.linregress(x, historical_values)
                    trend_direction = 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
                else:
                    trend_direction = 'insufficient_data'

                # 计算当前值相对于历史的偏差
                z_score = (current_value - historical_mean) / historical_std if historical_std > 0 else 0

                trends[metric] = {
                    'current_value': current_value,
                    'historical_mean': historical_mean,
                    'historical_std': historical_std,
                    'z_score': z_score,
                    'trend_direction': trend_direction,
                    'is_anomaly': abs(z_score) > 2  # 2 sigma rule
                }

        return trends

    def _detect_performance_anomalies(self, historical_data: List[Dict], current_data: Dict) -> Dict:
        """检测性能异常"""
        anomalies = {}

        # 使用简单的统计方法检测异常
        if len(historical_data) >= 5:  # 需要足够的历史数据
            for metric in ['auc', 'ks']:
                historical_values = []
                for record in historical_data[-10:]:  # 使用最近10个记录
                    if metric in record.get('basic_metrics', {}):
                        historical_values.append(record['basic_metrics'][metric])

                if historical_values and metric in current_data.get('basic_metrics', {}):
                    current_value = current_data['basic_metrics'][metric]
                    q1, q3 = np.percentile(historical_values, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    is_anomaly = current_value < lower_bound or current_value > upper_bound

                    anomalies[metric] = {
                        'current_value': current_value,
                        'normal_range': [lower_bound, upper_bound],
                        'is_anomaly': is_anomaly,
                        'anomaly_type': 'low' if current_value < lower_bound else 'high' if current_value > upper_bound else 'normal'
                    }

        return anomalies

    def _assess_degradation_status(self, trends: Dict, anomalies: Dict) -> str:
        """评估退化状态"""
        if not trends and not anomalies:
            return 'insufficient_data'

        # 检查是否有显著的性能下降
        critical_degradation = False
        moderate_degradation = False

        for metric, trend_info in trends.items():
            if metric in ['auc', 'ks']:  # 关键指标
                if trend_info.get('trend_direction') == 'declining' and trend_info.get('z_score', 0) < -2:
                    critical_degradation = True
                elif trend_info.get('trend_direction') == 'declining' and trend_info.get('z_score', 0) < -1:
                    moderate_degradation = True

        # 检查异常
        for metric, anomaly_info in anomalies.items():
            if anomaly_info.get('is_anomaly') and anomaly_info.get('anomaly_type') == 'low':
                if metric in ['auc', 'ks']:
                    critical_degradation = True

        if critical_degradation:
            return 'critical_degradation'
        elif moderate_degradation:
            return 'moderate_degradation'
        else:
            return 'stable'

    def _generate_degradation_recommendations(self, analysis: Dict) -> List[str]:
        """生成退化分析建议"""
        recommendations = []
        status = analysis.get('degradation_status', 'unknown')

        if status == 'critical_degradation':
            recommendations.extend([
                "模型性能显著下降，建议立即停止使用",
                "检查数据质量和特征稳定性",
                "考虑重新训练模型或使用备用模型"
            ])
        elif status == 'moderate_degradation':
            recommendations.extend([
                "模型性能有所下降，建议密切监控",
                "分析数据漂移和特征变化",
                "准备模型重训练计划"
            ])
        elif status == 'stable':
            recommendations.append("模型性能稳定，继续监控")

        return recommendations

    def _analyze_population_drift(self, baseline_data: pd.DataFrame, current_data: pd.DataFrame,
                                target_col: str, score_col: str) -> Dict:
        """分析总体分布漂移"""
        population_drift = {}

        # 目标变量分布变化
        if target_col in baseline_data.columns and target_col in current_data.columns:
            baseline_positive_rate = baseline_data[target_col].mean()
            current_positive_rate = current_data[target_col].mean()

            population_drift['target_distribution'] = {
                'baseline_positive_rate': baseline_positive_rate,
                'current_positive_rate': current_positive_rate,
                'change': current_positive_rate - baseline_positive_rate,
                'relative_change': (current_positive_rate - baseline_positive_rate) / baseline_positive_rate if baseline_positive_rate > 0 else 0
            }

        # 评分分布变化
        if score_col in baseline_data.columns and score_col in current_data.columns:
            baseline_scores = baseline_data[score_col].dropna()
            current_scores = current_data[score_col].dropna()

            # KS test for distribution comparison
            ks_stat, ks_p_value = stats.ks_2samp(baseline_scores, current_scores)

            population_drift['score_distribution'] = {
                'baseline_mean': baseline_scores.mean(),
                'current_mean': current_scores.mean(),
                'baseline_std': baseline_scores.std(),
                'current_std': current_scores.std(),
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'significant_drift': ks_p_value < 0.05
            }

        return population_drift

    def _analyze_drift_performance_impact(self, baseline_y: pd.Series, baseline_scores: pd.Series,
                                        current_y: pd.Series, current_scores: pd.Series) -> Dict:
        """分析漂移对性能的影响"""
        from sklearn.metrics import roc_auc_score

        # 计算基准和当前的AUC
        baseline_auc = roc_auc_score(baseline_y, baseline_scores)
        current_auc = roc_auc_score(current_y, current_scores)

        return {
            'baseline_auc': baseline_auc,
            'current_auc': current_auc,
            'auc_change': current_auc - baseline_auc,
            'performance_degradation': current_auc < baseline_auc,
            'significant_change': abs(current_auc - baseline_auc) > 0.05
        }

    def _analyze_feature_drift_impact(self, baseline_features: pd.DataFrame, current_features: pd.DataFrame) -> Dict:
        """分析特征漂移影响"""
        feature_drift_impact = {}

        common_features = list(set(baseline_features.columns) & set(current_features.columns))

        for feature in common_features:
            baseline_feature = baseline_features[feature].dropna()
            current_feature = current_features[feature].dropna()

            if len(baseline_feature) > 0 and len(current_feature) > 0:
                # KS test
                ks_stat, ks_p_value = stats.ks_2samp(baseline_feature, current_feature)

                # Mean shift
                mean_shift = current_feature.mean() - baseline_feature.mean()
                relative_mean_shift = mean_shift / baseline_feature.std() if baseline_feature.std() > 0 else 0

                feature_drift_impact[feature] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'significant_drift': ks_p_value < 0.05,
                    'mean_shift': mean_shift,
                    'relative_mean_shift': relative_mean_shift,
                    'drift_severity': 'high' if ks_stat > 0.3 else 'medium' if ks_stat > 0.1 else 'low'
                }

        return feature_drift_impact

    def _assess_drift_risk(self, drift_analysis: Dict) -> Dict:
        """评估漂移风险"""
        risk_factors = []
        risk_score = 0

        # 检查目标分布变化
        target_drift = drift_analysis.get('population_drift', {}).get('target_distribution', {})
        if abs(target_drift.get('relative_change', 0)) > 0.2:
            risk_factors.append('significant_target_distribution_change')
            risk_score += 3

        # 检查评分分布变化
        score_drift = drift_analysis.get('population_drift', {}).get('score_distribution', {})
        if score_drift.get('significant_drift', False):
            risk_factors.append('significant_score_distribution_drift')
            risk_score += 2

        # 检查特征漂移
        feature_drift = drift_analysis.get('feature_drift_impact', {})
        high_drift_features = [f for f, info in feature_drift.items() if info.get('drift_severity') == 'high']
        if len(high_drift_features) > len(feature_drift) * 0.3:  # 超过30%的特征高度漂移
            risk_factors.append('widespread_feature_drift')
            risk_score += 4

        # 评估总体风险等级
        if risk_score >= 6:
            risk_level = 'critical'
        elif risk_score >= 3:
            risk_level = 'high'
        elif risk_score >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'high_drift_features': high_drift_features if 'high_drift_features' in locals() else []
        }

    def _calculate_performance_health_score(self, performance_analysis: Dict) -> float:
        """计算性能健康分数"""
        basic_metrics = performance_analysis.get('basic_analysis', {})
        auc = basic_metrics.get('auc', 0)

        # 基于AUC的简单评分
        if auc >= 0.8:
            return 1.0
        elif auc >= 0.7:
            return 0.8
        elif auc >= 0.6:
            return 0.6
        else:
            return 0.3

    def _calculate_stability_health_score(self, degradation_analysis: Dict) -> float:
        """计算稳定性健康分数"""
        status = degradation_analysis.get('degradation_status', 'unknown')

        status_scores = {
            'stable': 1.0,
            'moderate_degradation': 0.6,
            'critical_degradation': 0.2,
            'insufficient_data': 0.5,
            'unknown': 0.5
        }

        return status_scores.get(status, 0.5)

    def _calculate_feature_health_score(self, feature_analysis: Dict) -> float:
        """计算特征健康分数"""
        # 基于特征稳定性的简单评分
        stability_info = feature_analysis.get('feature_stability', {})

        if not stability_info:
            return 0.5

        stability_scores = []
        for feature_info in stability_info.values():
            stability_scores.append(feature_info.get('stability_score', 0.5))

        return np.mean(stability_scores) if stability_scores else 0.5

    def _calculate_data_quality_health_score(self, drift_analysis: Dict) -> float:
        """计算数据质量健康分数"""
        risk_assessment = drift_analysis.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'medium')

        risk_scores = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4,
            'critical': 0.1
        }

        return risk_scores.get(risk_level, 0.5)

    def _identify_issues_and_warnings(self, detailed_analysis: Dict) -> Tuple[List[str], List[str]]:
        """识别关键问题和警告"""
        critical_issues = []
        warnings = []

        # 检查性能问题
        performance = detailed_analysis.get('performance', {})
        if performance:
            basic_analysis = performance.get('basic_analysis', {})
            auc = basic_analysis.get('auc', 0)

            if auc < 0.6:
                critical_issues.append(f"模型AUC过低 ({auc:.3f})")
            elif auc < 0.7:
                warnings.append(f"模型AUC较低 ({auc:.3f})")

        # 检查退化问题
        degradation = detailed_analysis.get('degradation', {})
        if degradation:
            status = degradation.get('degradation_status')
            if status == 'critical_degradation':
                critical_issues.append("模型性能严重退化")
            elif status == 'moderate_degradation':
                warnings.append("模型性能轻度退化")

        # 检查漂移问题
        drift = detailed_analysis.get('drift', {})
        if drift:
            risk_level = drift.get('risk_assessment', {}).get('risk_level')
            if risk_level == 'critical':
                critical_issues.append("数据漂移风险严重")
            elif risk_level == 'high':
                warnings.append("数据漂移风险较高")

        return critical_issues, warnings

    def _generate_health_recommendations(self, health_report: Dict) -> List[str]:
        """生成健康建议"""
        recommendations = []
        overall_score = health_report.get('overall_health_score', 0)

        if overall_score < 0.5:
            recommendations.extend([
                "模型整体健康状况较差，建议立即检查",
                "考虑停止使用当前模型",
                "准备启用备用模型或重新训练"
            ])
        elif overall_score < 0.7:
            recommendations.extend([
                "模型健康状况一般，需要密切监控",
                "检查具体问题并制定改进计划"
            ])
        else:
            recommendations.append("模型健康状况良好，继续正常监控")

        # 基于具体问题添加建议
        for issue in health_report.get('critical_issues', []):
            if 'AUC' in issue:
                recommendations.append("重新评估特征选择和模型算法")
            elif '退化' in issue:
                recommendations.append("分析数据变化并考虑模型更新")
            elif '漂移' in issue:
                recommendations.append("检查数据来源和特征工程流程")

        return list(set(recommendations))  # 去重

    def _check_lift_trend_consistency(self, segments: List[Dict]) -> bool:
        """检查lift趋势的一致性"""
        if len(segments) < 3:
            return True

        lifts = [seg['lift'] for seg in segments]
        # 检查是否大致呈递减趋势（高分段应该有更高的lift）
        decreasing_count = 0
        for i in range(len(lifts) - 1):
            if lifts[i] >= lifts[i + 1]:
                decreasing_count += 1

        return decreasing_count / (len(lifts) - 1) >= 0.6  # 60%以上符合递减趋势

    def _assess_feature_stability_level(self, max_value_ratio: float) -> str:
        """评估特征稳定性水平"""
        if max_value_ratio >= 0.95:
            return 'very_unstable'
        elif max_value_ratio >= 0.8:
            return 'unstable'
        elif max_value_ratio >= 0.6:
            return 'moderately_stable'
        else:
            return 'stable'


def analyze_model_comprehensive(model_name: str,
                              y_true: np.ndarray,
                              y_scores: np.ndarray,
                              X: Optional[pd.DataFrame] = None,
                              historical_performance: Optional[List[Dict]] = None) -> Dict:
    """
    快捷函数：执行模型的综合分析

    Parameters:
    -----------
    model_name : str
        模型名称
    y_true : np.ndarray
        真实标签
    y_scores : np.ndarray
        预测概率
    X : pd.DataFrame, optional
        特征数据
    historical_performance : list, optional
        历史性能数据

    Returns:
    --------
    comprehensive_analysis : dict
        综合分析结果
    """
    analyzer = ModelAnalyzer(model_name)

    # 性能分析
    performance_analysis = analyzer.analyze_model_performance(y_true, y_scores)

    # 特征分析（如果提供了特征数据）
    feature_analysis = None
    if X is not None:
        feature_analysis = analyzer.analyze_feature_impact(X, pd.Series(y_true), y_scores)

    # 退化分析（如果提供了历史数据）
    degradation_analysis = None
    if historical_performance:
        current_performance = {'basic_metrics': performance_analysis['basic_analysis']}
        degradation_analysis = analyzer.analyze_model_degradation(
            historical_performance, current_performance
        )

    # 生成健康报告
    health_report = analyzer.generate_model_health_report(
        performance_analysis, feature_analysis, degradation_analysis
    )

    return {
        'model_name': model_name,
        'analysis_timestamp': datetime.now().isoformat(),
        'performance_analysis': performance_analysis,
        'feature_analysis': feature_analysis,
        'degradation_analysis': degradation_analysis,
        'health_report': health_report
    }