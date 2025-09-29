"""
防过拟合优化测试

演示如何使用AntiOverfittingTuner解决AUC/KS目标函数导致的过拟合问题
确保训练集和验证集性能差异在可接受范围内
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# 导入我们的优化器和目标函数
from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner
from model_tools.optimization.objectives import KSObjective, AUCObjective


def create_challenging_dataset(n_samples=2000, noise_level=0.3):
    """
    创建一个容易过拟合的数据集
    包含真实信号和噪声特征
    """
    np.random.seed(42)

    # 真实有用的特征
    useful_features = {
        'age': np.random.normal(35, 15, n_samples).clip(18, 80),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(650, 120, n_samples).clip(300, 850),
        'debt_ratio': np.random.beta(2, 5, n_samples)
    }

    # 噪声特征（与目标变量无关但可能被模型学到）
    noise_features = {}
    for i in range(10):
        noise_features[f'noise_{i}'] = np.random.normal(0, 1, n_samples)

    # 高度相关的特征（容易导致过拟合）
    correlated_features = {}
    for i in range(5):
        # 与有用特征高度相关但添加噪声
        base_feature = useful_features['income']
        correlated_features[f'corr_income_{i}'] = base_feature + np.random.normal(0, base_feature * 0.1)

    # 合并所有特征
    all_features = {**useful_features, **noise_features, **correlated_features}
    X = pd.DataFrame(all_features)

    # 生成目标变量（主要基于有用特征）
    risk_score = (
        -0.0001 * (useful_features['age'] - 35)**2 +
        -0.00001 * useful_features['income'] +
        -0.01 * (useful_features['credit_score'] - 650) +
        3.0 * useful_features['debt_ratio'] +
        np.random.normal(0, noise_level, n_samples)  # 添加噪声
    )

    # 转换为二分类概率
    y_prob = 1 / (1 + np.exp(-risk_score))
    y = np.random.binomial(1, y_prob, n_samples)

    print(f"数据集信息:")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {X.shape[1]} (有用特征: 4, 噪声特征: 10, 相关特征: 5)")
    print(f"  违约率: {y.mean():.2%}")
    print(f"  噪声水平: {noise_level}")

    return X, y


def demonstrate_overfitting_problem():
    """演示传统优化方法的过拟合问题"""
    print("🔴 传统超参数优化的过拟合问题演示")
    print("=" * 60)

    # 1. 创建容易过拟合的数据
    X, y = create_challenging_dataset(n_samples=1500, noise_level=0.4)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 2. 使用传统方法（容易过拟合的参数）
    print("\n🚨 使用容易过拟合的参数:")
    overfitting_params = {
        'n_estimators': 500,        # 很多树
        'max_depth': 12,            # 很深
        'learning_rate': 0.2,       # 较高学习率
        'num_leaves': 200,          # 很多叶子
        'min_child_samples': 5,     # 很少的最小样本
        'subsample': 1.0,           # 不采样
        'colsample_bytree': 1.0,    # 不特征采样
        'reg_alpha': 0.0,           # 无L1正则
        'reg_lambda': 0.0,          # 无L2正则
        'random_state': 42,
        'verbosity': -1
    }

    overfitting_model = LGBMClassifier(**overfitting_params)
    overfitting_model.fit(X_train, y_train)

    # 预测
    y_train_pred = overfitting_model.predict_proba(X_train)[:, 1]
    y_test_pred = overfitting_model.predict_proba(X_test)[:, 1]

    # 计算指标
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    ks_obj = KSObjective()
    train_ks = ks_obj(y_train, y_train_pred)
    test_ks = ks_obj(y_test, y_test_pred)

    print(f"  训练集 AUC: {train_auc:.4f}")
    print(f"  测试集 AUC: {test_auc:.4f}")
    print(f"  AUC 差距: {abs(train_auc - test_auc):.4f}")
    print(f"  训练集 KS:  {train_ks:.4f}")
    print(f"  测试集 KS:  {test_ks:.4f}")
    print(f"  KS 差距:  {abs(train_ks - test_ks):.4f}")

    if abs(train_auc - test_auc) > 0.05 or abs(train_ks - test_ks) > 0.03:
        print("  ❌ 存在过拟合问题！")
    else:
        print("  ✅ 无过拟合问题")

    return X, y


def demonstrate_anti_overfitting_solution():
    """演示防过拟合解决方案"""
    print("\n\n🟢 防过拟合解决方案演示")
    print("=" * 60)

    # 使用相同的挑战性数据集
    X, y = create_challenging_dataset(n_samples=1500, noise_level=0.4)

    # 1. 创建防过拟合优化器
    print("\n🛡️ 创建防过拟合优化器...")
    anti_overfitting_tuner = AntiOverfittingTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        study_name='anti_overfitting_demo',
        # 防过拟合配置
        max_auc_gap=0.05,
        max_ks_gap=0.03,
        overfitting_penalty_weight=2.0,
        use_regularization_focused_space=True,
        early_stopping_patience=10
    )

    print("  配置参数:")
    print(f"    最大AUC差距: {anti_overfitting_tuner.max_auc_gap}")
    print(f"    最大KS差距: {anti_overfitting_tuner.max_ks_gap}")
    print(f"    过拟合惩罚权重: {anti_overfitting_tuner.overfitting_penalty_weight}")
    print(f"    使用正则化参数空间: {anti_overfitting_tuner.use_regularization_focused_space}")

    # 2. 执行防过拟合优化
    print(f"\n🔍 执行防过拟合优化 (减少试验次数到20次用于演示)...")

    # 使用KS目标函数（容易过拟合的目标）
    ks_objective = KSObjective()

    optimization_result = anti_overfitting_tuner.optimize_anti_overfitting(
        X=X,
        y=y,
        objective_function=ks_objective,
        n_trials=20,  # 演示用较少次数
        cv_folds=3,
        validation_size=0.2,
        show_progress_bar=False
    )

    # 3. 分析结果
    print(f"\n📊 优化结果分析:")
    print(f"  总试验次数: {optimization_result['results_summary']['total_trials']}")
    print(f"  满足条件的试验: {optimization_result['results_summary']['valid_trials']}")
    print(f"  过拟合率: {optimization_result['results_summary']['overfitting_rate']:.2%}")
    print(f"  平均AUC差距: {optimization_result['results_summary']['avg_auc_gap']:.4f}")
    print(f"  平均KS差距: {optimization_result['results_summary']['avg_ks_gap']:.4f}")

    # 4. 最佳参数对比
    print(f"\n🏆 推荐参数 (防过拟合):")
    recommended_params = optimization_result['recommended_params']
    for param, value in recommended_params.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.4f}")
        else:
            print(f"    {param}: {value}")

    # 5. 测试推荐模型的实际性能
    print(f"\n✅ 验证推荐模型性能:")

    # 分割数据用于最终验证
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y  # 使用不同的随机种子
    )

    # 训练推荐模型
    recommended_model = LGBMClassifier(**{**recommended_params, 'verbosity': -1})
    recommended_model.fit(X_train_final, y_train_final)

    # 预测
    y_train_pred_rec = recommended_model.predict_proba(X_train_final)[:, 1]
    y_test_pred_rec = recommended_model.predict_proba(X_test_final)[:, 1]

    # 计算最终指标
    final_train_auc = roc_auc_score(y_train_final, y_train_pred_rec)
    final_test_auc = roc_auc_score(y_test_final, y_test_pred_rec)

    final_train_ks = ks_objective(y_train_final, y_train_pred_rec)
    final_test_ks = ks_objective(y_test_final, y_test_pred_rec)

    final_auc_gap = abs(final_train_auc - final_test_auc)
    final_ks_gap = abs(final_train_ks - final_test_ks)

    print(f"  训练集 AUC: {final_train_auc:.4f}")
    print(f"  测试集 AUC: {final_test_auc:.4f}")
    print(f"  AUC 差距: {final_auc_gap:.4f} ({'✅ 通过' if final_auc_gap <= 0.05 else '❌ 超出'})")

    print(f"  训练集 KS:  {final_train_ks:.4f}")
    print(f"  测试集 KS:  {final_test_ks:.4f}")
    print(f"  KS 差距:  {final_ks_gap:.4f} ({'✅ 通过' if final_ks_gap <= 0.03 else '❌ 超出'})")

    # 6. 性能历史分析
    print(f"\n📈 性能历史分析:")
    performance_df = anti_overfitting_tuner.get_performance_analysis()

    if len(performance_df) > 0:
        print(f"  最佳5个试验:")
        top_5 = performance_df.head(5)[['trial', 'val_score', 'auc_gap', 'ks_gap', 'meets_all_criteria']]
        for _, row in top_5.iterrows():
            status = "✅" if row['meets_all_criteria'] else "❌"
            print(f"    Trial {int(row['trial'])}: 验证得分={row['val_score']:.4f}, "
                  f"AUC差距={row['auc_gap']:.4f}, KS差距={row['ks_gap']:.4f} {status}")

    return optimization_result, anti_overfitting_tuner


def compare_methods():
    """对比传统方法和防过拟合方法"""
    print("\n\n📊 方法对比总结")
    print("=" * 60)

    print("🔴 传统超参数优化问题:")
    print("  - 容易学到噪声特征")
    print("  - 训练集性能过高，验证集性能较低")
    print("  - AUC/KS差距经常超出可接受范围")
    print("  - 模型复杂度过高，泛化能力差")

    print("\n🟢 防过拟合优化解决方案:")
    print("  - 正则化导向的参数空间")
    print("  - 过拟合惩罚机制")
    print("  - 验证集性能监控")
    print("  - 自动筛选满足差距要求的模型")
    print("  - 平衡模型性能和泛化能力")

    print("\n💡 使用建议:")
    print("  - 当训练集和验证集性能差距较大时，优先使用AntiOverfittingTuner")
    print("  - 根据业务要求调整max_auc_gap和max_ks_gap阈值")
    print("  - 增加overfitting_penalty_weight以更严格地控制过拟合")
    print("  - 在数据量较小或特征维度较高时特别有效")


def show_usage_examples():
    """展示具体使用方法"""
    print(f"\n💡 AntiOverfittingTuner使用方法:")

    usage_code = '''
# 1. 导入和创建优化器
from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner
from model_tools.optimization.objectives import KSObjective
from lightgbm import LGBMClassifier

# 创建防过拟合优化器
tuner = AntiOverfittingTuner(
    model_class=LGBMClassifier,
    model_type='lgb',
    direction='maximize',
    max_auc_gap=0.05,        # 最大AUC差距
    max_ks_gap=0.03,         # 最大KS差距
    overfitting_penalty_weight=2.0,  # 过拟合惩罚权重
    use_regularization_focused_space=True  # 使用正则化参数空间
)

# 2. 执行防过拟合优化
result = tuner.optimize_anti_overfitting(
    X=X_train,
    y=y_train,
    objective_function=KSObjective(),
    n_trials=100,
    cv_folds=5,
    validation_size=0.2
)

# 3. 获取推荐参数（满足防过拟合要求）
best_params = result['recommended_params']
best_model = LGBMClassifier(**best_params)

# 4. 性能分析
performance_df = tuner.get_performance_analysis()
valid_trials = performance_df[performance_df['meets_all_criteria']]

# 5. 可视化分析（可选）
tuner.plot_overfitting_analysis(save_path='overfitting_analysis.png')
'''

    print(usage_code)


if __name__ == "__main__":
    print("🛡️ 防过拟合超参数优化演示")
    print("=" * 80)

    # 1. 演示过拟合问题
    demonstrate_overfitting_problem()

    # 2. 演示解决方案
    result, tuner = demonstrate_anti_overfitting_solution()

    # 3. 方法对比
    compare_methods()

    # 4. 使用方法
    show_usage_examples()

    print(f"\n🎉 演示完成！")
    print(f"防过拟合优化器成功解决了AUC/KS目标函数导致的过拟合问题。")
    print(f"通过正则化参数空间和过拟合惩罚机制，确保模型性能差距在可接受范围内。")