"""
简化的防过拟合测试，用于调试
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# 导入我们的模块
from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner
from model_tools.optimization.objectives import KSObjective


def test_simple():
    """简单测试"""
    print("🧪 简化防过拟合测试")
    print("=" * 50)

    # 创建简单数据
    np.random.seed(42)
    n_samples = 500

    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.uniform(0, 1, n_samples)
    })

    # 简单的逻辑关系
    y_prob = 1 / (1 + np.exp(-(X['feature1'] * 0.5 + X['feature2'] * 0.3)))
    y = np.random.binomial(1, y_prob, n_samples)

    print(f"数据: {n_samples} 样本, {X.shape[1]} 特征, 违约率 {y.mean():.2%}")

    # 创建优化器
    print("\n创建优化器...")
    tuner = AntiOverfittingTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        max_auc_gap=0.05,
        max_ks_gap=0.03,
        overfitting_penalty_weight=1.0
    )

    print("优化器创建成功")

    # 测试单个试验
    print("\n测试单个参数组合...")
    try:
        # 手动分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 简单参数
        test_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'min_child_samples': 20,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1
        }

        # 训练模型
        model = LGBMClassifier(**test_params)
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        # 计算指标
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)

        ks_obj = KSObjective()
        train_ks = ks_obj(y_train, y_train_pred)
        val_ks = ks_obj(y_val, y_val_pred)

        print(f"  训练集 AUC: {train_auc:.4f}")
        print(f"  验证集 AUC: {val_auc:.4f}")
        print(f"  AUC差距: {abs(train_auc - val_auc):.4f}")
        print(f"  训练集 KS: {train_ks:.4f}")
        print(f"  验证集 KS: {val_ks:.4f}")
        print(f"  KS差距: {abs(train_ks - val_ks):.4f}")

        # 测试过拟合惩罚计算
        penalty_info = tuner._calculate_overfitting_penalty(train_auc, val_auc, train_ks, val_ks)
        print(f"  过拟合惩罚: {penalty_info['total_penalty']:.4f}")
        print(f"  是否过拟合: {penalty_info['is_overfitting']}")

        print("✅ 单个试验测试成功")

    except Exception as e:
        print(f"❌ 单个试验测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试优化过程（少量试验）
    print("\n测试优化过程...")
    try:
        # 先检查study状态
        print(f"  Study方向: {tuner.direction}")
        print(f"  Study状态: {len(tuner.study.trials)} trials")

        result = tuner.optimize_anti_overfitting(
            X=X,
            y=y,
            objective_function=KSObjective(),
            n_trials=3,  # 只测试3次
            cv_folds=3,
            validation_size=0.3,
            show_progress_bar=False
        )

        print(f"  Study完成后试验数: {len(tuner.study.trials)}")
        print(f"  性能历史长度: {len(tuner.performance_history_)}")
        print(f"  优化完成，试验次数: {result['results_summary']['total_trials']}")
        print(f"  满足条件的试验: {result['results_summary']['valid_trials']}")

        # 显示performance_history的内容
        if len(tuner.performance_history_) > 0:
            print("  性能历史记录:")
            for record in tuner.performance_history_:
                print(f"    Trial {record.get('trial', 'N/A')}: val_score={record.get('val_score', 'N/A'):.4f}")

        if result['results_summary']['total_trials'] > 0:
            print("✅ 优化测试成功")
            return True
        else:
            print("❌ 没有执行任何试验")
            return False

    except Exception as e:
        print(f"❌ 优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败")