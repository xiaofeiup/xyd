"""
防过拟合超参数优化解决方案

解决AUC/KS目标函数导致的过拟合问题，确保训练集和验证集性能差异在可接受范围内：
- AUC差异 ≤ 0.05
- KS差异 ≤ 0.03

使用方法：
1. 使用AntiOverfittingTuner替代HyperparameterTuner
2. 设置max_auc_gap和max_ks_gap阈值
3. 调整overfitting_penalty_weight控制惩罚强度
4. 使用正则化导向的参数空间
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner
from model_tools.optimization.objectives import KSObjective


def solve_overfitting_example():
    """防过拟合解决方案示例"""

    print("🛡️ 防过拟合超参数优化解决方案")
    print("=" * 60)

    # 1. 准备数据（模拟容易过拟合的场景）
    print("📊 准备数据...")
    np.random.seed(42)
    n_samples = 1000

    # 包含噪声和相关特征的数据集
    X = pd.DataFrame({
        # 有用特征
        'useful1': np.random.normal(0, 1, n_samples),
        'useful2': np.random.normal(0, 1, n_samples),
        # 噪声特征
        'noise1': np.random.normal(0, 1, n_samples),
        'noise2': np.random.normal(0, 1, n_samples),
        'noise3': np.random.normal(0, 1, n_samples),
    })

    # 目标变量（主要基于有用特征）
    y_prob = 1 / (1 + np.exp(-(X['useful1'] * 0.8 + X['useful2'] * 0.6)))
    y = np.random.binomial(1, y_prob, n_samples)

    print(f"   数据规模: {n_samples} 样本, {X.shape[1]} 特征")
    print(f"   违约率: {y.mean():.2%}")

    # 2. 创建防过拟合优化器
    print("\n🛡️ 创建防过拟合优化器...")

    tuner = AntiOverfittingTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        study_name='anti_overfitting_example',
        # 防过拟合设置
        max_auc_gap=0.05,                           # 最大AUC差距
        max_ks_gap=0.03,                            # 最大KS差距
        overfitting_penalty_weight=2.0,             # 过拟合惩罚权重
        use_regularization_focused_space=True,      # 使用正则化参数空间
        early_stopping_patience=10                  # 早停耐心
    )

    print("   配置完成 ✅")
    print(f"     AUC差距阈值: ≤ {tuner.max_auc_gap}")
    print(f"     KS差距阈值: ≤ {tuner.max_ks_gap}")
    print(f"     过拟合惩罚权重: {tuner.overfitting_penalty_weight}")

    # 3. 执行防过拟合优化
    print(f"\n🔍 执行防过拟合优化...")

    optimization_result = tuner.optimize_anti_overfitting(
        X=X,
        y=y,
        objective_function=KSObjective(),           # 使用KS目标（容易过拟合）
        n_trials=50,                               # 试验次数
        cv_folds=5,                                # 交叉验证折数
        validation_size=0.2,                       # 验证集比例
        show_progress_bar=True                     # 显示进度
    )

    # 4. 结果分析
    print(f"\n📊 优化结果分析:")
    summary = optimization_result['results_summary']
    print(f"   总试验次数: {summary['total_trials']}")
    print(f"   满足条件的试验: {summary['valid_trials']}")
    print(f"   过拟合率: {summary['overfitting_rate']:.1%}")
    print(f"   平均AUC差距: {summary['avg_auc_gap']:.4f}")
    print(f"   平均KS差距: {summary['avg_ks_gap']:.4f}")

    if summary['best_valid_auc_gap'] is not None:
        print(f"   最佳试验AUC差距: {summary['best_valid_auc_gap']:.4f}")
        print(f"   最佳试验KS差距: {summary['best_valid_ks_gap']:.4f}")

    # 5. 获取推荐参数
    print(f"\n🏆 推荐参数（防过拟合优化）:")
    recommended_params = optimization_result['recommended_params']

    # 显示关键参数
    key_params = ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                  'reg_alpha', 'reg_lambda', 'min_child_samples']
    for param in key_params:
        if param in recommended_params:
            value = recommended_params[param]
            if isinstance(value, float):
                print(f"     {param}: {value:.4f}")
            else:
                print(f"     {param}: {value}")

    # 6. 验证推荐模型
    print(f"\n✅ 验证推荐模型性能:")

    # 分割数据用于验证
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    # 训练推荐模型
    final_params = {**recommended_params, 'verbosity': -1}
    model = LGBMClassifier(**final_params)
    model.fit(X_train, y_train)

    # 预测和评估
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    ks_obj = KSObjective()
    train_ks = ks_obj(y_train, y_train_pred)
    test_ks = ks_obj(y_test, y_test_pred)

    auc_gap = abs(train_auc - test_auc)
    ks_gap = abs(train_ks - test_ks)

    print(f"   训练集 AUC: {train_auc:.4f}")
    print(f"   测试集 AUC: {test_auc:.4f}")
    print(f"   AUC差距: {auc_gap:.4f} ({'✅ 通过' if auc_gap <= 0.05 else '❌ 超出'})")

    print(f"   训练集 KS: {train_ks:.4f}")
    print(f"   测试集 KS: {test_ks:.4f}")
    print(f"   KS差距: {ks_gap:.4f} ({'✅ 通过' if ks_gap <= 0.03 else '❌ 超出'})")

    # 7. 性能历史分析（可选）
    performance_df = tuner.get_performance_analysis()
    if len(performance_df) > 0:
        print(f"\n📈 最佳5个满足条件的试验:")
        valid_trials = performance_df[performance_df['meets_all_criteria']]
        if len(valid_trials) > 0:
            top_valid = valid_trials.head(5)
            for idx, row in top_valid.iterrows():
                print(f"     Trial {int(row['trial'])}: 验证得分={row['val_score']:.4f}, "
                      f"AUC差距={row['auc_gap']:.4f}, KS差距={row['ks_gap']:.4f}")
        else:
            print("     没有完全满足条件的试验")

    print(f"\n🎉 防过拟合优化完成!")

    return tuner, optimization_result


if __name__ == "__main__":
    # 运行示例
    tuner, result = solve_overfitting_example()

    print(f"\n💡 核心优势:")
    print(f"   ✅ 自动检测和惩罚过拟合")
    print(f"   ✅ 正则化导向的参数空间")
    print(f"   ✅ 验证集性能监控")
    print(f"   ✅ 满足业务指标要求的参数推荐")
    print(f"   ✅ 平衡模型性能和泛化能力")

    print(f"\n🛠️ 适用场景:")
    print(f"   • 训练集和验证集性能差距较大")
    print(f"   • 数据量较小或特征维度较高")
    print(f"   • 对模型泛化能力要求较高")
    print(f"   • 需要满足严格的性能差距要求")