"""
pytest配置文件
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_data():
    """创建示例数据集"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # 添加一些问题数据
    df['constant_feature'] = 1  # 单一值特征
    df.loc[df.sample(frac=0.1).index, 'feature_0'] = np.nan  # 添加缺失值

    return df


@pytest.fixture
def sample_binary_classification_data():
    """创建二分类示例数据"""
    np.random.seed(42)
    n_samples = 500

    # 特征数据
    X = np.random.randn(n_samples, 5)

    # 目标变量（二分类）
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    # 预测概率
    y_scores = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1)))

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y, y_scores


@pytest.fixture
def sample_monitoring_data():
    """创建监控示例数据"""
    np.random.seed(42)

    # 历史性能数据
    historical_performance = []
    for i in range(10):
        historical_performance.append({
            'timestamp': f'2023-01-{i+1:02d}T00:00:00',
            'basic_metrics': {
                'auc': 0.75 + np.random.normal(0, 0.02),
                'ks': 0.45 + np.random.normal(0, 0.02),
                'precision': 0.70 + np.random.normal(0, 0.02),
                'recall': 0.65 + np.random.normal(0, 0.02)
            }
        })

    # 当前性能数据
    current_performance = {
        'timestamp': '2023-01-11T00:00:00',
        'basic_metrics': {
            'auc': 0.72,  # 略有下降
            'ks': 0.42,
            'precision': 0.68,
            'recall': 0.63
        }
    }

    return historical_performance, current_performance


@pytest.fixture
def sample_drift_data():
    """创建漂移检测示例数据"""
    np.random.seed(42)

    # 基准数据
    baseline_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.exponential(1, 1000),
        'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })

    # 当前数据（带有轻微漂移）
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.2, 1.1, 800),  # 均值和方差轻微变化
        'feature_2': np.random.exponential(1.2, 800),  # 参数轻微变化
        'feature_3': np.random.choice(['A', 'B', 'C'], 800, p=[0.5, 0.3, 0.2]),  # 分布轻微变化
        'target': np.random.choice([0, 1], 800)
    })

    return baseline_data, current_data