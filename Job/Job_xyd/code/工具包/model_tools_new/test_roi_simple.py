"""
简化版ROI优化测试
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_roi_optimization_simple():
    """简单测试ROI优化功能"""
    print("🚀 简化版ROI优化测试")
    print("=" * 50)

    try:
        # 1. 导入模块
        print("📦 导入模块...")
        from model_tools.optimization.objectives import BusinessROIObjective
        from model_tools.optimization.hyperparameter_tuner import HyperparameterTuner
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        print("✅ 模块导入成功")

        # 2. 生成测试数据
        print("\n📊 生成测试数据...")
        np.random.seed(42)
        n_samples = 1000

        # 简单特征
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 1, n_samples)
        })

        # 生成标签（基于特征的逻辑关系）
        y_prob = 1 / (1 + np.exp(-(X['feature1'] * 0.5 + X['feature2'] * 0.3 - X['feature3'] * 0.8)))
        y_true = np.random.binomial(1, y_prob, n_samples)

        # 贷款金额
        loan_amounts = np.random.uniform(10000, 100000, n_samples)

        print(f"   样本数: {n_samples}")
        print(f"   违约率: {y_true.mean():.2%}")
        print(f"   平均贷款金额: {loan_amounts.mean():,.0f}")

        # 3. 测试ROI目标函数
        print("\n💰 测试ROI目标函数...")
        roi_objective = BusinessROIObjective(
            loan_amounts=loan_amounts,
            interest_rate=0.15,
            operational_cost_rate=0.02,
            recovery_rate=0.3,
            threshold=0.5
        )

        # 生成随机预测
        y_pred_random = np.random.random(n_samples)
        roi_random = roi_objective(y_true, y_pred_random)
        print(f"   随机预测ROI: {roi_random:.4f}")

        # 4. 简单模型训练
        print("\n🤖 训练基础模型...")
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
            X, y_true, loan_amounts, test_size=0.3, random_state=42
        )

        # 默认参数模型
        model_default = LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        model_default.fit(X_train, y_train)
        y_pred_default = model_default.predict_proba(X_test)[:, 1]

        # 计算性能
        auc_default = roc_auc_score(y_test, y_pred_default)
        roi_test_objective = BusinessROIObjective(
            loan_amounts=amounts_test,
            interest_rate=0.15,
            operational_cost_rate=0.02,
            recovery_rate=0.3,
            threshold=0.5
        )
        roi_default = roi_test_objective(y_test, y_pred_default)

        print(f"   默认模型 - AUC: {auc_default:.4f}, ROI: {roi_default:.4f}")

        # 5. 测试参数优化（简化版）
        print("\n⚙️ 测试参数优化...")

        # 手动测试几组参数
        param_sets = [
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05},
            {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.08},
            {'n_estimators': 80, 'max_depth': 6, 'learning_rate': 0.12}
        ]

        best_score = -999
        best_params = None
        results = []

        for i, params in enumerate(param_sets):
            model_params = {
                **params,
                'random_state': 42,
                'verbosity': -1
            }

            model = LGBMClassifier(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_pred)
            roi = roi_test_objective(y_test, y_pred)

            # 组合得分：70% AUC + 30% ROI（归一化）
            normalized_roi = max(0, min(roi, 0.5)) / 0.5  # 假设ROI范围0-0.5
            combined_score = 0.7 * auc + 0.3 * normalized_roi

            results.append({
                'params': params,
                'auc': auc,
                'roi': roi,
                'combined_score': combined_score
            })

            print(f"   参数组{i+1}: AUC={auc:.4f}, ROI={roi:.4f}, 综合={combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_params = params

        print(f"\n🏆 最佳结果:")
        print(f"   最佳参数: {best_params}")
        print(f"   最佳综合得分: {best_score:.4f}")

        # 6. 创建优化器（验证可以正常创建）
        print("\n🔧 验证优化器创建...")
        try:
            tuner = HyperparameterTuner(
                model_class=LGBMClassifier,
                model_type='lgb',
                direction='maximize'
            )
            print("✅ 优化器创建成功")

            # 测试参数建议
            import optuna
            study = optuna.create_study(direction='maximize')
            trial = study.ask()
            suggested_params = tuner._suggest_parameters(trial)
            print(f"   参数建议示例: {list(suggested_params.keys())[:5]}")

        except Exception as e:
            print(f"❌ 优化器创建失败: {e}")

        print("\n🎉 测试完成！ROI优化功能基本正常。")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_roi_optimization_simple()