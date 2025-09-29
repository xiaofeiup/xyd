"""
特征验证模块

提供特征质量检查和验证功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import warnings
from scipy import stats


class FeatureValidator:
    """
    特征验证器

    对特征进行质量检查和验证
    """

    def __init__(self, validation_config: Optional[Dict] = None):
        """
        初始化特征验证器

        Parameters:
        -----------
        validation_config : dict, optional
            验证配置
        """
        self.config = validation_config or self._get_default_config()
        self.logger = logging.getLogger("FeatureValidator")

    def validate_features(self,
                         data: pd.DataFrame,
                         target_col: Optional[str] = None,
                         feature_cols: Optional[List[str]] = None) -> Dict:
        """
        对特征进行全面验证

        Parameters:
        -----------
        data : pd.DataFrame
            数据集
        target_col : str, optional
            目标变量列名
        feature_cols : list, optional
            特征列名列表，如果不提供则自动推断

        Returns:
        --------
        validation_result : dict
            验证结果
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]

        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(feature_cols),
            'total_samples': len(data),
            'validation_summary': {},
            'feature_reports': {},
            'data_quality_issues': [],
            'recommendations': []
        }

        # 对每个特征进行验证
        feature_issues = []
        for feature in feature_cols:
            feature_report = self.validate_single_feature(
                data[feature], feature_name=feature
            )
            validation_result['feature_reports'][feature] = feature_report

            # 收集问题
            if feature_report['issues']:
                feature_issues.extend([
                    f"{feature}: {issue}" for issue in feature_report['issues']
                ])

        validation_result['data_quality_issues'] = feature_issues

        # 验证特征与目标变量的关系（如果提供了目标变量）
        if target_col is not None and target_col in data.columns:
            target_validation = self.validate_feature_target_relationship(
                data[feature_cols], data[target_col]
            )
            validation_result['target_relationship'] = target_validation

        # 验证特征间关系
        correlation_validation = self.validate_feature_correlations(data[feature_cols])
        validation_result['correlation_analysis'] = correlation_validation

        # 生成验证总结
        validation_result['validation_summary'] = self._create_validation_summary(
            validation_result
        )

        # 生成建议
        validation_result['recommendations'] = self._generate_recommendations(
            validation_result
        )

        return validation_result

    def validate_single_feature(self,
                               feature_data: pd.Series,
                               feature_name: str = "feature") -> Dict:
        """
        验证单个特征

        Parameters:
        -----------
        feature_data : pd.Series
            特征数据
        feature_name : str, default="feature"
            特征名称

        Returns:
        --------
        feature_report : dict
            特征验证报告
        """
        report = {
            'feature_name': feature_name,
            'data_type': str(feature_data.dtype),
            'basic_stats': {},
            'quality_metrics': {},
            'issues': [],
            'warnings': [],
            'quality_score': 0.0
        }

        # 基础统计信息
        report['basic_stats'] = self._calculate_basic_stats(feature_data)

        # 质量指标
        report['quality_metrics'] = self._calculate_quality_metrics(feature_data)

        # 检查数据质量问题
        issues, warnings = self._check_feature_quality_issues(
            feature_data, feature_name
        )
        report['issues'] = issues
        report['warnings'] = warnings

        # 计算质量分数
        report['quality_score'] = self._calculate_feature_quality_score(
            report['quality_metrics'], issues
        )

        return report

    def validate_feature_target_relationship(self,
                                           features: pd.DataFrame,
                                           target: pd.Series) -> Dict:
        """
        验证特征与目标变量的关系

        Parameters:
        -----------
        features : pd.DataFrame
            特征数据
        target : pd.Series
            目标变量

        Returns:
        --------
        relationship_report : dict
            关系验证报告
        """
        relationship_report = {
            'target_stats': self._calculate_basic_stats(target),
            'feature_target_correlations': {},
            'predictive_power': {},
            'issues': [],
            'warnings': []
        }

        for feature_name in features.columns:
            feature_data = features[feature_name]

            # 计算相关性
            correlation = self._calculate_feature_target_correlation(
                feature_data, target
            )
            relationship_report['feature_target_correlations'][feature_name] = correlation

            # 计算预测能力
            predictive_power = self._calculate_predictive_power(
                feature_data, target
            )
            relationship_report['predictive_power'][feature_name] = predictive_power

            # 检查关系问题
            rel_issues, rel_warnings = self._check_feature_target_issues(
                feature_data, target, feature_name
            )
            relationship_report['issues'].extend(rel_issues)
            relationship_report['warnings'].extend(rel_warnings)

        return relationship_report

    def validate_feature_correlations(self, features: pd.DataFrame) -> Dict:
        """
        验证特征间相关性

        Parameters:
        -----------
        features : pd.DataFrame
            特征数据

        Returns:
        --------
        correlation_report : dict
            相关性验证报告
        """
        correlation_report = {
            'correlation_matrix': {},
            'high_correlation_pairs': [],
            'multicollinearity_issues': [],
            'vif_scores': {}
        }

        # 计算相关性矩阵
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr()
                correlation_report['correlation_matrix'] = corr_matrix.to_dict()

                # 找出高相关性特征对
                high_corr_pairs = self._find_high_correlation_pairs(
                    corr_matrix, threshold=self.config['high_correlation_threshold']
                )
                correlation_report['high_correlation_pairs'] = high_corr_pairs

                # 计算VIF（方差膨胀因子）
                if len(numeric_features.columns) <= 20:  # 限制计算量
                    vif_scores = self._calculate_vif_scores(numeric_features)
                    correlation_report['vif_scores'] = vif_scores

                    # 检查多重共线性问题
                    multicollinearity_issues = [
                        f"{feature}: VIF={vif:.2f}"
                        for feature, vif in vif_scores.items()
                        if vif > self.config['vif_threshold']
                    ]
                    correlation_report['multicollinearity_issues'] = multicollinearity_issues

        except Exception as e:
            self.logger.warning(f"计算特征相关性时出错: {e}")
            correlation_report['error'] = str(e)

        return correlation_report

    def validate_data_consistency(self,
                                train_data: pd.DataFrame,
                                test_data: pd.DataFrame,
                                feature_cols: Optional[List[str]] = None) -> Dict:
        """
        验证训练集和测试集数据一致性

        Parameters:
        -----------
        train_data : pd.DataFrame
            训练数据
        test_data : pd.DataFrame
            测试数据
        feature_cols : list, optional
            特征列名列表

        Returns:
        --------
        consistency_report : dict
            一致性验证报告
        """
        if feature_cols is None:
            feature_cols = list(set(train_data.columns) & set(test_data.columns))

        consistency_report = {
            'timestamp': datetime.now().isoformat(),
            'common_features': len(feature_cols),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'distribution_comparisons': {},
            'schema_consistency': {},
            'issues': [],
            'warnings': []
        }

        # 检查模式一致性
        schema_check = self._check_schema_consistency(
            train_data[feature_cols], test_data[feature_cols]
        )
        consistency_report['schema_consistency'] = schema_check

        # 比较分布
        for feature in feature_cols:
            if feature in train_data.columns and feature in test_data.columns:
                dist_comparison = self._compare_feature_distributions(
                    train_data[feature], test_data[feature], feature
                )
                consistency_report['distribution_comparisons'][feature] = dist_comparison

                # 检查分布差异问题
                if dist_comparison.get('significant_difference', False):
                    consistency_report['issues'].append(
                        f"{feature}: 训练集和测试集分布存在显著差异"
                    )

        return consistency_report

    def validate_feature_stability(self,
                                 data_periods: List[pd.DataFrame],
                                 feature_cols: Optional[List[str]] = None,
                                 period_names: Optional[List[str]] = None) -> Dict:
        """
        验证特征在不同时期的稳定性

        Parameters:
        -----------
        data_periods : list
            不同时期的数据列表
        feature_cols : list, optional
            特征列名列表
        period_names : list, optional
            时期名称列表

        Returns:
        --------
        stability_report : dict
            稳定性验证报告
        """
        if len(data_periods) < 2:
            return {'error': '需要至少两个时期的数据进行稳定性验证'}

        if period_names is None:
            period_names = [f"Period_{i+1}" for i in range(len(data_periods))]

        if feature_cols is None:
            feature_cols = list(set.intersection(*[set(df.columns) for df in data_periods]))

        stability_report = {
            'timestamp': datetime.now().isoformat(),
            'periods_count': len(data_periods),
            'period_names': period_names,
            'analyzed_features': len(feature_cols),
            'feature_stability': {},
            'overall_stability': {},
            'unstable_features': [],
            'warnings': []
        }

        # 对每个特征分析稳定性
        for feature in feature_cols:
            feature_stability = self._analyze_feature_temporal_stability(
                [df[feature] for df in data_periods], feature, period_names
            )
            stability_report['feature_stability'][feature] = feature_stability

            # 识别不稳定特征
            if feature_stability.get('stability_score', 1.0) < self.config['stability_threshold']:
                stability_report['unstable_features'].append({
                    'feature': feature,
                    'stability_score': feature_stability['stability_score'],
                    'issues': feature_stability.get('issues', [])
                })

        # 计算总体稳定性
        stability_scores = [
            info.get('stability_score', 0)
            for info in stability_report['feature_stability'].values()
        ]

        if stability_scores:
            stability_report['overall_stability'] = {
                'mean_stability': np.mean(stability_scores),
                'min_stability': np.min(stability_scores),
                'stable_features_ratio': sum(1 for score in stability_scores
                                           if score >= self.config['stability_threshold']) / len(stability_scores)
            }

        return stability_report

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'missing_threshold': 0.1,  # 缺失值比例阈值
            'single_value_threshold': 0.95,  # 单一值比例阈值
            'high_correlation_threshold': 0.8,  # 高相关性阈值
            'vif_threshold': 10.0,  # VIF阈值
            'outlier_threshold': 3.0,  # 异常值阈值（Z-score）
            'stability_threshold': 0.7,  # 稳定性阈值
            'min_predictive_power': 0.01  # 最小预测能力
        }

    def _calculate_basic_stats(self, series: pd.Series) -> Dict:
        """计算基础统计信息"""
        stats = {
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().mean(),
            'unique_count': series.nunique(),
            'value_count_num1': series.value_counts(dropna=False).iloc[0] / len(series) if len(series) > 0 else 0,
            'unique_percentage': series.nunique() / len(series) if len(series) > 0 else 0
        }

        # 数值型特征的额外统计
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna()
            if len(numeric_series) > 0:
                stats.update({
                    'mean': numeric_series.mean(),
                    'std': numeric_series.std(),
                    'min': numeric_series.min(),
                    'max': numeric_series.max(),
                    'median': numeric_series.median(),
                    'q25': numeric_series.quantile(0.25),
                    'q75': numeric_series.quantile(0.75),
                    'skewness': numeric_series.skew(), # 偏度，描述数据分布的偏斜程度
                    'kurtosis': numeric_series.kurtosis() # 峰度，描述数据分布的峰态
                })

        return stats

    def _calculate_quality_metrics(self, series: pd.Series) -> Dict:
        """计算质量指标"""
        metrics = {}

        # 完整性指标
        metrics['completeness'] = 1 - series.isnull().mean()

        # 一致性指标（单一值占比）
        if len(series) > 0:
            value_counts = series.value_counts()
            most_frequent_ratio = value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0
            metrics['consistency'] = 1 - most_frequent_ratio
        else:
            metrics['consistency'] = 0

        # 有效性指标（基于数据类型和范围）
        metrics['validity'] = self._calculate_validity_score(series)

        # 唯一性指标
        metrics['uniqueness'] = series.nunique() / len(series) if len(series) > 0 else 0

        return metrics

    def _calculate_validity_score(self, series: pd.Series) -> float:
        """计算有效性分数"""
        valid_count = 0
        total_count = len(series.dropna())

        if total_count == 0:
            return 0.0

        if pd.api.types.is_numeric_dtype(series):
            # 数值型：检查是否为有限数值
            numeric_series = series.dropna()
            valid_count = np.isfinite(numeric_series).sum()
        else:
            # 非数值型：检查是否为空字符串或无效值
            non_null_series = series.dropna()
            valid_count = len(non_null_series[non_null_series.astype(str).str.strip() != ''])

        return valid_count / total_count

    def _check_feature_quality_issues(self, series: pd.Series, feature_name: str) -> Tuple[List[str], List[str]]:
        """检查特征质量问题"""
        issues = []
        warnings = []

        # 检查缺失值比例
        missing_ratio = series.isnull().mean()
        if missing_ratio > self.config['missing_threshold']:
            issues.append(f"缺失值比例过高 ({missing_ratio:.2%})")
        elif missing_ratio > self.config['missing_threshold'] / 2:
            warnings.append(f"缺失值比例较高 ({missing_ratio:.2%})")

        # 检查单一值占比
        if len(series) > 0:
            value_counts = series.value_counts()
            if len(value_counts) > 0:
                most_frequent_ratio = value_counts.iloc[0] / len(series)
                if most_frequent_ratio > self.config['single_value_threshold']:
                    issues.append(f"单一值占比过高 ({most_frequent_ratio:.2%})")

        # 检查数值型特征的异常值
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna()
            if len(numeric_series) > 0:
                z_scores = np.abs(stats.zscore(numeric_series))
                outlier_ratio = (z_scores > self.config['outlier_threshold']).mean()
                if outlier_ratio > 0.1:  # 超过10%的异常值
                    warnings.append(f"异常值比例较高 ({outlier_ratio:.2%})")

        # 检查数据类型一致性
        if series.dtype == 'object':
            # 对于object类型，检查是否混合了不同类型的数据
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                sample_types = set(type(x).__name__ for x in non_null_series.head(100))
                if len(sample_types) > 1:
                    warnings.append(f"数据类型不一致: {sample_types}")

        return issues, warnings

    def _calculate_feature_quality_score(self, quality_metrics: Dict, issues: List[str]) -> float:
        """计算特征质量分数"""
        # 基础分数基于质量指标
        base_score = np.mean([
            quality_metrics.get('completeness', 0),
            quality_metrics.get('consistency', 0),
            quality_metrics.get('validity', 0),
            min(quality_metrics.get('uniqueness', 0) * 2, 1.0)  # 唯一性不要过高
        ])

        # 根据问题数量扣分
        issue_penalty = len(issues) * 0.2
        final_score = max(0, base_score - issue_penalty)

        return final_score

    def _calculate_feature_target_correlation(self, feature: pd.Series, target: pd.Series) -> Dict:
        """计算特征与目标变量的相关性"""
        correlation_info = {
            'correlation_type': 'unknown',
            'correlation_value': 0,
            'p_value': 1.0,
            'is_significant': False
        }

        try:
            # 移除缺失值
            valid_mask = feature.notna() & target.notna()
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]

            if len(feature_clean) < 10:  # 样本太少
                return correlation_info

            # 根据数据类型选择相关性计算方法
            if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(target):
                # 数值-数值：Pearson相关
                corr, p_val = stats.pearsonr(feature_clean, target_clean)
                correlation_info.update({
                    'correlation_type': 'pearson',
                    'correlation_value': corr,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05
                })
            else:
                # 分类-数值或分类-分类：使用点双列相关或Cramer's V
                if pd.api.types.is_numeric_dtype(target):
                    # 分类特征与连续目标变量
                    feature_encoded = pd.Categorical(feature_clean).codes
                    corr, p_val = stats.pearsonr(feature_encoded, target_clean)
                    correlation_info.update({
                        'correlation_type': 'point_biserial',
                        'correlation_value': corr,
                        'p_value': p_val,
                        'is_significant': p_val < 0.05
                    })
                else:
                    # 分类特征与分类目标变量：使用Cramer's V
                    cramers_v = self._calculate_cramers_v(feature_clean, target_clean)
                    correlation_info.update({
                        'correlation_type': 'cramers_v',
                        'correlation_value': cramers_v,
                        'p_value': 0.0,  # Cramer's V没有p值
                        'is_significant': cramers_v > 0.1
                    })

        except Exception as e:
            self.logger.warning(f"计算相关性时出错: {e}")
            correlation_info['error'] = str(e)

        return correlation_info

    def _calculate_predictive_power(self, feature: pd.Series, target: pd.Series) -> Dict:
        """计算特征的预测能力"""
        predictive_power = {
            'method': 'unknown',
            'score': 0,
            'interpretation': 'no_predictive_power'
        }

        try:
            # 移除缺失值
            valid_mask = feature.notna() & target.notna()
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]

            if len(feature_clean) < 10:
                return predictive_power

            # 对于分类目标变量，计算信息价值(IV)
            if target_clean.nunique() == 2:  # 二分类
                iv_score = self._calculate_information_value(feature_clean, target_clean)
                predictive_power.update({
                    'method': 'information_value',
                    'score': iv_score,
                    'interpretation': self._interpret_iv_score(iv_score)
                })
            else:
                # 连续目标变量，使用R²
                if pd.api.types.is_numeric_dtype(feature):
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score

                    X = feature_clean.values.reshape(-1, 1)
                    y = target_clean.values

                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)

                    predictive_power.update({
                        'method': 'r_squared',
                        'score': r2,
                        'interpretation': 'good' if r2 > 0.1 else 'weak' if r2 > 0.01 else 'no_predictive_power'
                    })

        except Exception as e:
            self.logger.warning(f"计算预测能力时出错: {e}")
            predictive_power['error'] = str(e)

        return predictive_power

    def _calculate_information_value(self, feature: pd.Series, target: pd.Series) -> float:
        """计算信息价值(IV)"""
        try:
            # 对连续变量进行分箱
            if pd.api.types.is_numeric_dtype(feature):
                feature_binned = pd.qcut(feature, q=10, duplicates='drop')
            else:
                feature_binned = feature

            # 计算WOE和IV
            cross_tab = pd.crosstab(feature_binned, target)

            if cross_tab.shape[1] != 2:
                return 0

            cross_tab.columns = ['good', 'bad']

            # 避免除零错误
            cross_tab['good'] = cross_tab['good'] + 0.5
            cross_tab['bad'] = cross_tab['bad'] + 0.5

            total_good = cross_tab['good'].sum()
            total_bad = cross_tab['bad'].sum()

            cross_tab['good_rate'] = cross_tab['good'] / total_good
            cross_tab['bad_rate'] = cross_tab['bad'] / total_bad
            cross_tab['woe'] = np.log(cross_tab['good_rate'] / cross_tab['bad_rate'])
            cross_tab['iv'] = (cross_tab['good_rate'] - cross_tab['bad_rate']) * cross_tab['woe']

            return cross_tab['iv'].sum()

        except Exception:
            return 0

    def _interpret_iv_score(self, iv_score: float) -> str:
        """解释IV分数"""
        if iv_score < 0.02:
            return 'no_predictive_power'
        elif iv_score < 0.1:
            return 'weak'
        elif iv_score < 0.3:
            return 'medium'
        elif iv_score < 0.5:
            return 'strong'
        else:
            return 'very_strong'

    def _calculate_cramers_v(self, feature: pd.Series, target: pd.Series) -> float:
        """计算Cramer's V"""
        try:
            cross_tab = pd.crosstab(feature, target)
            chi2, _, _, _ = stats.chi2_contingency(cross_tab)
            n = cross_tab.sum().sum()
            min_dim = min(cross_tab.shape) - 1

            if min_dim == 0:
                return 0

            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return cramers_v
        except Exception:
            return 0

    def _check_feature_target_issues(self, feature: pd.Series, target: pd.Series, feature_name: str) -> Tuple[List[str], List[str]]:
        """检查特征与目标变量关系的问题"""
        issues = []
        warnings = []

        # 计算相关性
        correlation_info = self._calculate_feature_target_correlation(feature, target)
        corr_value = abs(correlation_info.get('correlation_value', 0))

        # 检查预测能力
        predictive_power = self._calculate_predictive_power(feature, target)
        power_score = predictive_power.get('score', 0)

        if power_score < self.config['min_predictive_power']:
            issues.append(f"预测能力不足 (score={power_score:.4f})")
        elif corr_value < 0.05:
            warnings.append(f"与目标变量相关性较低 (r={corr_value:.3f})")

        # 检查是否存在完美分离（对于分类问题）
        if target.nunique() == 2 and pd.api.types.is_numeric_dtype(feature):
            try:
                feature_by_target = feature.groupby(target)
                ranges = [(group.min(), group.max()) for name, group in feature_by_target]
                if len(ranges) == 2 and ranges[0][1] < ranges[1][0]:  # 完全分离
                    warnings.append("特征可能存在完美分离，需要检查数据泄露")
            except Exception:
                pass

        return issues, warnings

    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float) -> List[Dict]:
        """找出高相关性特征对"""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })

        # 按相关性大小排序
        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

        return high_corr_pairs

    def _calculate_vif_scores(self, data: pd.DataFrame) -> Dict:
        """计算方差膨胀因子(VIF)"""
        vif_scores = {}

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # 移除缺失值
            data_clean = data.dropna()

            if len(data_clean) < 10 or data_clean.shape[1] < 2:
                return vif_scores

            # 计算每个特征的VIF
            for i, feature in enumerate(data_clean.columns):
                try:
                    vif = variance_inflation_factor(data_clean.values, i)
                    vif_scores[feature] = vif if np.isfinite(vif) else np.inf
                except Exception:
                    vif_scores[feature] = np.inf

        except ImportError:
            self.logger.warning("statsmodels未安装，无法计算VIF")
        except Exception as e:
            self.logger.warning(f"计算VIF时出错: {e}")

        return vif_scores

    def _check_schema_consistency(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """检查模式一致性"""
        schema_check = {
            'columns_match': True,
            'dtypes_match': True,
            'missing_columns_in_test': [],
            'extra_columns_in_test': [],
            'dtype_mismatches': []
        }

        # 检查列是否匹配
        train_cols = set(train_data.columns)
        test_cols = set(test_data.columns)

        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols

        if missing_in_test:
            schema_check['columns_match'] = False
            schema_check['missing_columns_in_test'] = list(missing_in_test)

        if extra_in_test:
            schema_check['columns_match'] = False
            schema_check['extra_columns_in_test'] = list(extra_in_test)

        # 检查数据类型是否匹配
        common_cols = train_cols & test_cols
        for col in common_cols:
            if train_data[col].dtype != test_data[col].dtype:
                schema_check['dtypes_match'] = False
                schema_check['dtype_mismatches'].append({
                    'column': col,
                    'train_dtype': str(train_data[col].dtype),
                    'test_dtype': str(test_data[col].dtype)
                })

        return schema_check

    def _compare_feature_distributions(self, train_feature: pd.Series, test_feature: pd.Series, feature_name: str) -> Dict:
        """比较特征在训练集和测试集中的分布"""
        comparison = {
            'feature_name': feature_name,
            'train_stats': self._calculate_basic_stats(train_feature),
            'test_stats': self._calculate_basic_stats(test_feature),
            'statistical_test': {},
            'significant_difference': False
        }

        try:
            # 移除缺失值
            train_clean = train_feature.dropna()
            test_clean = test_feature.dropna()

            if len(train_clean) < 10 or len(test_clean) < 10:
                return comparison

            # 根据数据类型选择统计检验
            if pd.api.types.is_numeric_dtype(train_feature):
                # 数值型：使用KS检验
                ks_stat, ks_p = stats.ks_2samp(train_clean, test_clean)
                comparison['statistical_test'] = {
                    'test_type': 'kolmogorov_smirnov',
                    'statistic': ks_stat,
                    'p_value': ks_p
                }
                comparison['significant_difference'] = ks_p < 0.05
            else:
                # 分类型：使用卡方检验
                train_counts = train_clean.value_counts()
                test_counts = test_clean.value_counts()

                # 合并计数，确保类别一致
                all_categories = set(train_counts.index) | set(test_counts.index)
                train_aligned = [train_counts.get(cat, 0) for cat in all_categories]
                test_aligned = [test_counts.get(cat, 0) for cat in all_categories]

                if len(all_categories) > 1 and sum(train_aligned) > 0 and sum(test_aligned) > 0:
                    chi2_stat, chi2_p, _, _ = stats.chi2_contingency([train_aligned, test_aligned])
                    comparison['statistical_test'] = {
                        'test_type': 'chi_square',
                        'statistic': chi2_stat,
                        'p_value': chi2_p
                    }
                    comparison['significant_difference'] = chi2_p < 0.05

        except Exception as e:
            self.logger.warning(f"比较分布时出错: {e}")
            comparison['error'] = str(e)

        return comparison

    def _analyze_feature_temporal_stability(self, feature_periods: List[pd.Series], feature_name: str, period_names: List[str]) -> Dict:
        """分析特征的时间稳定性"""
        stability_analysis = {
            'feature_name': feature_name,
            'periods_count': len(feature_periods),
            'period_stats': {},
            'stability_metrics': {},
            'stability_score': 1.0,
            'issues': []
        }

        # 计算每个时期的统计信息
        period_stats = []
        for i, period_data in enumerate(feature_periods):
            period_name = period_names[i] if i < len(period_names) else f"Period_{i+1}"
            stats = self._calculate_basic_stats(period_data)
            period_stats.append(stats)
            stability_analysis['period_stats'][period_name] = stats

        # 计算稳定性指标
        if len(period_stats) >= 2:
            # 计算均值的变异系数
            if pd.api.types.is_numeric_dtype(feature_periods[0]):
                means = [stats.get('mean', 0) for stats in period_stats if stats.get('mean') is not None]
                if means and np.mean(means) != 0:
                    mean_cv = np.std(means) / np.mean(means)
                    stability_analysis['stability_metrics']['mean_coefficient_variation'] = mean_cv

                    if mean_cv > 0.3:
                        stability_analysis['issues'].append("均值变化较大")
                        stability_analysis['stability_score'] *= 0.7

            # 计算缺失率变化
            missing_rates = [stats.get('null_percentage', 0) for stats in period_stats]
            missing_rate_change = max(missing_rates) - min(missing_rates)
            stability_analysis['stability_metrics']['missing_rate_change'] = missing_rate_change

            if missing_rate_change > 0.1:
                stability_analysis['issues'].append("缺失率变化较大")
                stability_analysis['stability_score'] *= 0.8

            # 计算唯一值比例变化
            unique_ratios = [stats.get('unique_percentage', 0) for stats in period_stats]
            unique_ratio_change = max(unique_ratios) - min(unique_ratios)
            stability_analysis['stability_metrics']['unique_ratio_change'] = unique_ratio_change

            if unique_ratio_change > 0.2:
                stability_analysis['issues'].append("唯一值比例变化较大")
                stability_analysis['stability_score'] *= 0.9

        return stability_analysis

    def _create_validation_summary(self, validation_result: Dict) -> Dict:
        """创建验证总结"""
        feature_reports = validation_result.get('feature_reports', {})

        # 统计质量分数
        quality_scores = [report.get('quality_score', 0) for report in feature_reports.values()]

        # 统计问题
        total_issues = len(validation_result.get('data_quality_issues', []))
        features_with_issues = sum(1 for report in feature_reports.values() if report.get('issues'))

        summary = {
            'total_features': validation_result.get('total_features', 0),
            'features_with_issues': features_with_issues,
            'total_issues': total_issues,
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'quality_distribution': {
                'high_quality': sum(1 for score in quality_scores if score >= 0.8),
                'medium_quality': sum(1 for score in quality_scores if 0.5 <= score < 0.8),
                'low_quality': sum(1 for score in quality_scores if score < 0.5)
            }
        }

        # 评估总体数据质量
        avg_score = summary['average_quality_score']
        if avg_score >= 0.8:
            summary['overall_quality'] = 'excellent'
        elif avg_score >= 0.6:
            summary['overall_quality'] = 'good'
        elif avg_score >= 0.4:
            summary['overall_quality'] = 'fair'
        else:
            summary['overall_quality'] = 'poor'

        return summary

    def _generate_recommendations(self, validation_result: Dict) -> List[str]:
        """生成建议"""
        recommendations = []

        summary = validation_result.get('validation_summary', {})
        overall_quality = summary.get('overall_quality', 'unknown')

        # 基于总体质量的建议
        if overall_quality == 'poor':
            recommendations.append("数据质量较差，建议进行全面的数据清洗")
        elif overall_quality == 'fair':
            recommendations.append("数据质量一般，建议重点处理质量分数较低的特征")

        # 基于具体问题的建议
        issues = validation_result.get('data_quality_issues', [])

        missing_issues = [issue for issue in issues if '缺失值' in issue]
        if len(missing_issues) > len(issues) * 0.3:
            recommendations.append("多个特征存在缺失值问题，考虑使用插值或删除策略")

        single_value_issues = [issue for issue in issues if '单一值' in issue]
        if single_value_issues:
            recommendations.append("移除单一值占比过高的特征")

        # 基于相关性分析的建议
        correlation_analysis = validation_result.get('correlation_analysis', {})
        high_corr_pairs = correlation_analysis.get('high_correlation_pairs', [])
        if len(high_corr_pairs) > 0:
            recommendations.append("存在高度相关的特征对，考虑进行特征降维或选择")

        multicollinearity = correlation_analysis.get('multicollinearity_issues', [])
        if multicollinearity:
            recommendations.append("检测到多重共线性问题，建议使用正则化方法或移除相关特征")

        return recommendations


def validate_dataset(data: pd.DataFrame,
                    target_col: Optional[str] = None,
                    feature_cols: Optional[List[str]] = None,
                    validation_config: Optional[Dict] = None) -> Dict:
    """
    快捷函数：对数据集进行全面验证

    Parameters:
    -----------
    data : pd.DataFrame
        数据集
    target_col : str, optional
        目标变量列名
    feature_cols : list, optional
        特征列名列表
    validation_config : dict, optional
        验证配置

    Returns:
    --------
    validation_result : dict
        验证结果
    """
    validator = FeatureValidator(validation_config)
    return validator.validate_features(data, target_col, feature_cols)


def compare_datasets(train_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    feature_cols: Optional[List[str]] = None) -> Dict:
    """
    快捷函数：比较训练集和测试集的一致性

    Parameters:
    -----------
    train_data : pd.DataFrame
        训练数据
    test_data : pd.DataFrame
        测试数据
    feature_cols : list, optional
        特征列名列表

    Returns:
    --------
    consistency_result : dict
        一致性验证结果
    """
    validator = FeatureValidator()
    return validator.validate_data_consistency(train_data, test_data, feature_cols)