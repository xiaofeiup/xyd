"""
测试evaluation模块
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from model_tools.evaluation.metrics import (
    calculate_auc, calculate_ks, calculate_psi,
    ModelEvaluator
)


class TestBasicMetrics:
    """测试基础指标计算"""

    def test_calculate_auc(self, sample_binary_classification_data):
        """测试AUC计算"""
        _, y_true, y_scores = sample_binary_classification_data

        auc = calculate_auc(y_true, y_scores)

        # 验证AUC值合理性
        assert 0 <= auc <= 1
        assert isinstance(auc, float)

        # 与sklearn结果对比
        sklearn_auc = roc_auc_score(y_true, y_scores)
        assert abs(auc - sklearn_auc) < 1e-6

    def test_calculate_ks(self, sample_binary_classification_data):
        """测试KS值计算"""
        _, y_true, y_scores = sample_binary_classification_data

        ks, ks_df, ks_index = calculate_ks(y_true, y_scores)

        # 验证KS值合理性
        assert 0 <= ks <= 1
        assert isinstance(ks, float)
        assert isinstance(ks_df, pd.DataFrame)
        assert isinstance(ks_index, (int, np.integer))

    def test_calculate_psi(self, sample_drift_data):
        """测试PSI计算"""
        baseline_data, current_data = sample_drift_data

        psi, psi_detail = calculate_psi(
            baseline_data['feature_1'],
            current_data['feature_1']
        )

        # 验证PSI值合理性
        assert psi >= 0
        assert isinstance(psi, float)
        if psi_detail is not None:
            assert isinstance(psi_detail, pd.DataFrame)

    def test_calculate_psi_with_bins(self, sample_drift_data):
        """测试指定分箱数的PSI计算"""
        baseline_data, current_data = sample_drift_data

        psi, psi_detail = calculate_psi(
            baseline_data['feature_1'],
            current_data['feature_1'],
            bins=5
        )

        assert psi >= 0
        assert isinstance(psi, float)


class TestModelEvaluator:
    """测试模型评估器"""

    def test_init(self):
        """测试初始化"""
        evaluator = ModelEvaluator("test_model")
        assert evaluator.model_name == "test_model"

    def test_evaluate_binary_classification(self, sample_binary_classification_data):
        """测试二分类评估"""
        _, y_true, y_scores = sample_binary_classification_data

        evaluator = ModelEvaluator("test_model")
        result = evaluator.evaluate_binary_classification(y_true, y_scores)

        # 验证返回结果结构
        assert isinstance(result, dict)
        assert 'model_name' in result
        assert 'basic_metrics' in result
        assert 'lift_metrics' in result
        assert 'sample_info' in result

        # 验证基础指标
        basic_metrics = result['basic_metrics']
        assert 'auc' in basic_metrics
        assert 'ks' in basic_metrics
        assert 'precision' in basic_metrics
        assert 'recall' in basic_metrics

        # 验证指标值合理性
        assert 0 <= basic_metrics['auc'] <= 1
        assert 0 <= basic_metrics['ks'] <= 1
        assert 0 <= basic_metrics['precision'] <= 1
        assert 0 <= basic_metrics['recall'] <= 1

    def test_evaluate_binary_classification_with_custom_threshold(self, sample_binary_classification_data):
        """测试自定义阈值的二分类评估"""
        _, y_true, y_scores = sample_binary_classification_data

        evaluator = ModelEvaluator("test_model")
        result = evaluator.evaluate_binary_classification(
            y_true, y_scores, threshold=0.6
        )

        assert 'threshold' in result
        assert result['threshold'] == 0.6

    def test_calculate_lift_analysis(self, sample_binary_classification_data):
        """测试Lift分析"""
        _, y_true, y_scores = sample_binary_classification_data

        evaluator = ModelEvaluator("test_model")
        lift_result = evaluator.calculate_lift_analysis(y_true, y_scores)

        # 验证结果结构
        assert isinstance(lift_result, dict)
        assert 'lift_table' in lift_result
        assert 'top_decile_lift' in lift_result

        # 验证lift表结构
        lift_table = lift_result['lift_table']
        assert isinstance(lift_table, pd.DataFrame)
        assert 'decile' in lift_table.columns
        assert 'lift' in lift_table.columns

    def test_calculate_ks_table(self, sample_binary_classification_data):
        """测试KS表计算"""
        _, y_true, y_scores = sample_binary_classification_data

        evaluator = ModelEvaluator("test_model")
        ks_table = evaluator.calculate_ks_table(y_true, y_scores)

        # 验证结果结构
        assert isinstance(ks_table, pd.DataFrame)
        assert 'decile' in ks_table.columns
        assert 'ks' in ks_table.columns
        assert len(ks_table) == 10  # 10个分位数

    def test_get_evaluation_summary(self, sample_binary_classification_data):
        """测试评估总结"""
        _, y_true, y_scores = sample_binary_classification_data

        evaluator = ModelEvaluator("test_model")
        result = evaluator.evaluate_binary_classification(y_true, y_scores)
        summary = evaluator.get_evaluation_summary(result)

        # 验证总结结构
        assert isinstance(summary, dict)
        assert 'model_performance' in summary
        assert 'key_metrics' in summary
        assert 'recommendations' in summary

    def test_invalid_inputs(self):
        """测试无效输入"""
        evaluator = ModelEvaluator("test_model")

        # 测试长度不匹配
        with pytest.raises(ValueError):
            evaluator.evaluate_binary_classification([0, 1], [0.1, 0.2, 0.3])

        # 测试空数组
        with pytest.raises(ValueError):
            evaluator.evaluate_binary_classification([], [])


class TestEdgeCases:
    """测试边界情况"""

    def test_perfect_classification(self):
        """测试完美分类情况"""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        auc = calculate_auc(y_true, y_scores)
        assert auc == 1.0

    def test_random_classification(self):
        """测试随机分类情况"""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000)
        y_scores = np.random.random(1000)

        auc = calculate_auc(y_true, y_scores)
        # 随机分类的AUC应该接近0.5
        assert 0.4 < auc < 0.6

    def test_single_class(self):
        """测试单一类别情况"""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        # AUC在单一类别情况下应该抛出异常或返回特殊值
        with pytest.raises(ValueError):
            calculate_auc(y_true, y_scores)

    def test_identical_scores(self):
        """测试相同预测分数情况"""
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5])

        auc = calculate_auc(y_true, y_scores)
        assert auc == 0.5  # 相同分数应该得到0.5的AUC