"""
测试features模块
"""

import pytest
import pandas as pd
import numpy as np

from model_tools.features.selection import (
    filter_features_by_single_value_ratio,
    FeatureSelector
)
from model_tools.features.importance import FeatureImportanceAnalyzer
from model_tools.features.validation import FeatureValidator, validate_dataset


class TestFeatureSelection:
    """测试特征选择功能"""

    def test_filter_features_by_single_value_ratio(self, sample_data):
        """测试单一值比例筛选"""
        # 创建包含高单一值比例特征的数据
        test_data = sample_data.copy()
        test_data['high_single_value'] = 'A'  # 100%单一值
        test_data.loc[:50, 'high_single_value'] = 'B'  # 降低到95%单一值

        kept_features, filter_log = filter_features_by_single_value_ratio(
            test_data, threshold=0.95, exclude_cols=['target']
        )

        # 验证结果
        assert isinstance(kept_features, list)
        assert isinstance(filter_log, pd.DataFrame)
        assert 'high_single_value' not in kept_features  # 应该被过滤掉
        assert 'constant_feature' not in kept_features  # 应该被过滤掉

    def test_feature_selector_init(self):
        """测试FeatureSelector初始化"""
        selector = FeatureSelector(
            method='iv',
            single_value_threshold=0.9,
            iv_threshold=0.1
        )
        assert selector.method == 'iv'
        assert selector.single_value_threshold == 0.9
        assert selector.iv_threshold == 0.1

    def test_feature_selector_fit_transform(self, sample_data):
        """测试特征选择器的fit_transform"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']

        selector = FeatureSelector(
            method='iv',
            single_value_threshold=0.95,
            iv_threshold=0.01,
            k_features=10
        )

        X_selected = selector.fit_transform(X, y, target_col='target')

        # 验证结果
        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X.shape[0]  # 行数不变
        assert X_selected.shape[1] <= X.shape[1]  # 列数减少或不变
        assert X_selected.shape[1] <= 10  # 不超过k_features

    def test_feature_selector_get_selection_summary(self, sample_data):
        """测试获取选择总结"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']

        selector = FeatureSelector(method='iv')
        selector.fit_transform(X, y, target_col='target')

        summary = selector.get_selection_summary()

        # 验证总结结构
        assert isinstance(summary, dict)
        assert 'original_features' in summary
        assert 'selected_features' in summary
        assert 'filter_steps' in summary

    def test_feature_selector_get_iv_log(self, sample_data):
        """测试获取IV日志"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']

        selector = FeatureSelector(method='iv')
        selector.fit_transform(X, y, target_col='target')

        iv_log = selector.get_iv_log()

        # 验证IV日志结构
        assert isinstance(iv_log, pd.DataFrame)
        assert 'feature' in iv_log.columns
        assert 'iv_value' in iv_log.columns
        assert 'interpretation' in iv_log.columns


class TestFeatureImportance:
    """测试特征重要性分析"""

    def test_feature_importance_analyzer_init(self):
        """测试FeatureImportanceAnalyzer初始化"""
        analyzer = FeatureImportanceAnalyzer(random_state=42)
        assert analyzer.random_state == 42

    def test_calculate_random_forest_importance(self, sample_data):
        """测试随机森林重要性计算"""
        X = sample_data.drop(['target', 'constant_feature'], axis=1).dropna()
        y = sample_data.loc[X.index, 'target']

        analyzer = FeatureImportanceAnalyzer(random_state=42)
        importance = analyzer.calculate_random_forest_importance(X, y)

        # 验证结果
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == X.shape[1]

    def test_calculate_iv_importance(self, sample_data):
        """测试IV重要性计算"""
        X = sample_data.drop(['target', 'constant_feature'], axis=1).dropna()
        y = sample_data.loc[X.index, 'target']

        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.calculate_iv_importance(X, y)

        # 验证结果
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == X.shape[1]

    def test_calculate_correlation_importance(self, sample_data):
        """测试相关性重要性计算"""
        X = sample_data.drop(['target', 'constant_feature'], axis=1).dropna()
        y = sample_data.loc[X.index, 'target']

        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.calculate_correlation_importance(X, y)

        # 验证结果
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_calculate_all_importance(self, sample_data):
        """测试计算所有重要性"""
        X = sample_data.drop(['target', 'constant_feature'], axis=1).dropna()
        y = sample_data.loc[X.index, 'target']

        analyzer = FeatureImportanceAnalyzer(random_state=42)
        methods = ['random_forest', 'iv', 'correlation']

        all_importance = analyzer.calculate_all_importance(X, y, methods)

        # 验证结果
        assert isinstance(all_importance, dict)
        for method in methods:
            assert method in all_importance
            assert isinstance(all_importance[method], pd.DataFrame)

    def test_get_consensus_ranking(self, sample_data):
        """测试一致性排名"""
        X = sample_data.drop(['target', 'constant_feature'], axis=1).dropna()
        y = sample_data.loc[X.index, 'target']

        analyzer = FeatureImportanceAnalyzer(random_state=42)
        methods = ['random_forest', 'iv']

        # 先计算重要性
        analyzer.calculate_all_importance(X, y, methods)

        # 获取一致性排名
        consensus = analyzer.get_consensus_ranking(methods, top_k=5)

        # 验证结果
        assert isinstance(consensus, pd.DataFrame)
        assert 'feature' in consensus.columns
        assert 'avg_rank' in consensus.columns
        assert len(consensus) <= 5


class TestFeatureValidation:
    """测试特征验证功能"""

    def test_feature_validator_init(self):
        """测试FeatureValidator初始化"""
        validator = FeatureValidator()
        assert hasattr(validator, 'config')

    def test_validate_single_feature(self, sample_data):
        """测试单个特征验证"""
        validator = FeatureValidator()
        feature = sample_data['feature_0']

        report = validator.validate_single_feature(feature, 'feature_0')

        # 验证报告结构
        assert isinstance(report, dict)
        assert 'feature_name' in report
        assert 'basic_stats' in report
        assert 'quality_metrics' in report
        assert 'issues' in report
        assert 'quality_score' in report

    def test_validate_features(self, sample_data):
        """测试特征验证"""
        validator = FeatureValidator()

        result = validator.validate_features(
            sample_data, target_col='target'
        )

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'total_features' in result
        assert 'feature_reports' in result
        assert 'validation_summary' in result
        assert 'recommendations' in result

    def test_validate_feature_target_relationship(self, sample_data):
        """测试特征与目标变量关系验证"""
        validator = FeatureValidator()
        features = sample_data.drop('target', axis=1)
        target = sample_data['target']

        result = validator.validate_feature_target_relationship(features, target)

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'feature_target_correlations' in result
        assert 'predictive_power' in result

    def test_validate_data_consistency(self, sample_data):
        """测试数据一致性验证"""
        # 创建训练集和测试集
        train_data = sample_data[:800]
        test_data = sample_data[800:]

        validator = FeatureValidator()
        result = validator.validate_data_consistency(train_data, test_data)

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'schema_consistency' in result
        assert 'distribution_comparisons' in result

    def test_validate_dataset_function(self, sample_data):
        """测试快捷验证函数"""
        result = validate_dataset(sample_data, target_col='target')

        # 验证结果结构
        assert isinstance(result, dict)
        assert 'validation_summary' in result
        assert 'feature_reports' in result


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe(self):
        """测试空DataFrame"""
        empty_df = pd.DataFrame()

        kept_features, filter_log = filter_features_by_single_value_ratio(empty_df)

        assert kept_features == []
        assert len(filter_log) == 0

    def test_single_column_dataframe(self):
        """测试单列DataFrame"""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})

        kept_features, filter_log = filter_features_by_single_value_ratio(single_col_df)

        assert 'col1' in kept_features
        assert len(filter_log) == 1

    def test_all_nan_feature(self):
        """测试全为NaN的特征"""
        data = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'nan_feature': [np.nan] * 5,
            'target': [0, 1, 0, 1, 0]
        })

        validator = FeatureValidator()
        result = validator.validate_features(data, target_col='target')

        # NaN特征应该被识别为问题
        nan_report = result['feature_reports']['nan_feature']
        assert len(nan_report['issues']) > 0

    def test_constant_feature(self):
        """测试常数特征"""
        data = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'constant_feature': [1] * 5,
            'target': [0, 1, 0, 1, 0]
        })

        kept_features, filter_log = filter_features_by_single_value_ratio(
            data, threshold=0.95, exclude_cols=['target']
        )

        # 常数特征应该被过滤掉
        assert 'constant_feature' not in kept_features
        assert 'good_feature' in kept_features