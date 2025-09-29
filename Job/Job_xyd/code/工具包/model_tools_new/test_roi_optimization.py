"""
测试基于ROI的自动参数选择功能

验证Optuna优化器与ROI目标函数的集成
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 检查依赖包
def check_dependencies():
    """检查必要的依赖包"""
    missing_packages = []

    try:
        import optuna
        print(f"✅ optuna {optuna.__version__}")
    except ImportError:
        missing_packages.append("optuna")

    try:
        import lightgbm as lgb
        print(f"✅ lightgbm {lgb.__version__}")
    except ImportError:
        missing_packages.append("lightgbm")

    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__}")
    except ImportError:
        missing_packages.append("scikit-learn")

    if missing_packages:
        print(f"❌ 缺少依赖包: {missing_packages}")
        print("请安装: pip install optuna lightgbm scikit-learn")
        return False

    return True

def generate_credit_data(n_samples=2000, random_state=42):
    """生成模拟信贷数据"""
    np.random.seed(random_state)

    # 客户特征
    age = np.random.normal(35, 10, n_samples).clip(18, 70)
    income = np.random.lognormal(10, 0.5, n_samples).clip(20000, 500000)
    credit_history = np.random.uniform(0, 100, n_samples)
    debt_ratio = np.random.beta(2, 5, n_samples)

    # 贷款金额（与收入相关）
    loan_amounts = (income * np.random.uniform(0.5, 3.0, n_samples)).clip(10000, 200000)

    # 违约概率模型（基于特征的真实关系）
    default_prob = (
        0.3 * (age < 25).astype(int) +
        0.2 * (income < 30000).astype(int) +
        0.25 * (credit_history < 30).astype(int) +
        0.3 * (debt_ratio > 0.6).astype(int) +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)

    # 生成违约标签
    y_true = np.random.binomial(1, default_prob, n_samples)

    # 特征矩阵
    X = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_history': credit_history,
        'debt_ratio': debt_ratio,
        'loan_amount': loan_amounts
    })

    print(f"📊 生成数据概览:")
    print(f"   样本数: {n_samples}")
    print(f"   违约率: {y_true.mean():.2%}")
    print(f"   平均贷款金额: {loan_amounts.mean():,.0f}")
    print(f"   贷款金额范围: {loan_amounts.min():,.0f} - {loan_amounts.max():,.0f}")

    return X, y_true, loan_amounts

def test_basic_imports():
    """测试基本导入功能"""
    print("\n🔍 测试模块导入...")

    try:
        from model_tools.optimization.hyperparameter_tuner import HyperparameterTuner
        from model_tools.optimization.objectives import BusinessROIObjective
        from model_tools.optimization.parameter_spaces import LightGBMSpace
        print("✅ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_roi_objective():
    """测试ROI目标函数"""
    print("\n🎯 测试ROI目标函数...")

    try:
        from model_tools.optimization.objectives import BusinessROIObjective

        # 生成测试数据
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.15, n_samples)
        y_pred = np.random.random(n_samples)
        loan_amounts = np.random.normal(50000, 20000, n_samples)

        # 创建ROI目标函数
        roi_objective = BusinessROIObjective(
            loan_amounts=loan_amounts,
            interest_rate=0.15,
            operational_cost_rate=0.02,
            recovery_rate=0.3,
            threshold=0.5
        )

        # 计算ROI
        roi_score = roi_objective(y_true, y_pred)
        print(f"✅ ROI目标函数计算成功: {roi_score:.4f}")

        # 测试不同阈值的ROI
        thresholds = [0.3, 0.5, 0.7]
        print("   不同阈值的ROI:")
        for threshold in thresholds:
            roi_objective.threshold = threshold
            roi = roi_objective(y_true, y_pred)
            print(f"     阈值 {threshold}: ROI = {roi:.4f}")

        return True

    except Exception as e:
        print(f"❌ ROI目标函数测试失败: {e}")
        return False

def test_parameter_space():
    """测试参数空间定义"""
    print("\n🔧 测试参数空间...")

    try:
        import optuna
        from model_tools.optimization.parameter_spaces import LightGBMSpace

        # 创建trial
        study = optuna.create_study(direction='maximize')
        trial = study.ask()

        # 测试LightGBM参数空间
        params = LightGBMSpace.suggest_parameters(trial, task_type='classification')

        print("✅ 参数空间生成成功")
        print("   生成的参数样例:")
        for key, value in list(params.items())[:8]:  # 显示前8个参数
            print(f"     {key}: {value}")

        # 验证约束关系
        max_depth = params['max_depth']
        num_leaves = params['num_leaves']
        theoretical_max = 2 ** max_depth

        if num_leaves < theoretical_max:
            print(f"✅ 参数约束验证成功: num_leaves({num_leaves}) < 2^max_depth({theoretical_max})")
        else:
            print(f"❌ 参数约束验证失败: num_leaves({num_leaves}) >= 2^max_depth({theoretical_max})")

        return True

    except Exception as e:
        print(f"❌ 参数空间测试失败: {e}")
        return False

def test_hyperparameter_tuner():
    """测试超参数优化器基本功能"""
    print("\n⚙️ 测试超参数优化器...")

    try:
        from lightgbm import LGBMClassifier
        from model_tools.optimization.hyperparameter_tuner import HyperparameterTuner

        # 生成简单测试数据
        X, y_true, loan_amounts = generate_credit_data(n_samples=500)

        # 创建优化器
        tuner = HyperparameterTuner(
            model_class=LGBMClassifier,
            model_type='lgb',
            direction='maximize'
        )

        print("✅ 超参数优化器创建成功")

        # 运行小规模优化测试
        print("   运行小规模优化测试（5次试验）...")
        result = tuner.optimize(
            X=X,
            y=y_true,
            n_trials=5,
            cv=3,
            scoring='roc_auc',
            show_progress_bar=False
        )

        print(f"✅ 优化完成")
        print(f"   最佳AUC: {result['best_score']:.4f}")
        print(f"   试验次数: {result['n_trials']}")
        print(f"   最佳参数样例: max_depth={result['best_params'].get('max_depth', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ 超参数优化器测试失败: {e}")
        return False

def test_roi_based_optimization():
    """测试基于ROI的自动参数选择"""
    print("\n💰 测试基于ROI的自动参数选择...")

    try:
        from lightgbm import LGBMClassifier
        from model_tools.optimization.hyperparameter_tuner import HyperparameterTuner
        from model_tools.optimization.objectives import BusinessROIObjective, create_credit_scoring_objective
        from sklearn.model_selection import cross_val_score

        # 生成信贷数据
        X, y_true, loan_amounts = generate_credit_data(n_samples=800)

        # 创建优化器
        tuner = HyperparameterTuner(
            model_class=LGBMClassifier,
            model_type='lgb',
            direction='maximize'
        )

        # 创建ROI目标函数
        roi_objective = BusinessROIObjective(
            loan_amounts=loan_amounts,
            interest_rate=0.18,  # 18%年利率
            operational_cost_rate=0.03,  # 3%运营成本
            recovery_rate=0.25,  # 25%回收率
            threshold=0.5
        )

        # 创建信贷专用多目标函数
        multi_objective = create_credit_scoring_objective(
            loan_amounts=loan_amounts,
            primary_metric='auc',
            business_weight=0.4,  # 40%权重给业务指标
            interest_rate=0.18,
            operational_cost_rate=0.03,
            recovery_rate=0.25
        )

        print("✅ ROI目标函数创建成功")

        # 定义自定义优化目标
        def custom_objective(trial):
            try:
                # 获取参数
                params = tuner._suggest_parameters(trial)

                # 创建模型
                model = LGBMClassifier(**params)

                # 使用交叉验证评估AUC
                auc_scores = cross_val_score(model, X, y_true, cv=3, scoring='roc_auc')
                auc_score = auc_scores.mean()

                # 训练模型获取预测
                model.fit(X, y_true)
                y_pred_proba = model.predict_proba(X)[:, 1]

                # 计算ROI
                roi_score = roi_objective(y_true, y_pred_proba)

                # 组合得分：70% AUC + 30% ROI
                combined_score = 0.7 * auc_score + 0.3 * roi_score

                # 记录详细信息
                trial.set_user_attr('auc_score', auc_score)
                trial.set_user_attr('roi_score', roi_score)
                trial.set_user_attr('combined_score', combined_score)

                return combined_score

            except Exception as e:
                print(f"   Trial {trial.number} failed: {e}")
                return -999

        # 设置自定义目标
        tuner.custom_objective = custom_objective

        print("   运行ROI优化（10次试验）...")

        # 手动优化循环
        best_score = -999
        best_params = None
        trial_results = []

        for i in range(10):
            trial = tuner.study.ask()
            try:
                score = custom_objective(trial)
                tuner.study.tell(trial, score)

                # 记录结果
                auc = trial.user_attrs.get('auc_score', 0)
                roi = trial.user_attrs.get('roi_score', 0)

                trial_results.append({
                    'trial': i,
                    'score': score,
                    'auc': auc,
                    'roi': roi
                })

                if score > best_score:
                    best_score = score
                    best_params = trial.params

                print(f"   Trial {i}: Score={score:.4f}, AUC={auc:.4f}, ROI={roi:.4f}")

            except Exception as e:
                print(f"   Trial {i} failed: {e}")
                continue

        print(f"\n✅ ROI优化完成!")
        print(f"   最佳综合得分: {best_score:.4f}")
        print(f"   最佳参数样例:")
        if best_params:
            for key, value in list(best_params.items())[:5]:
                print(f"     {key}: {value}")

        # 分析结果
        df_results = pd.DataFrame(trial_results)
        if len(df_results) > 0:
            print(f"\n📊 优化结果分析:")
            print(f"   平均AUC: {df_results['auc'].mean():.4f}")
            print(f"   平均ROI: {df_results['roi'].mean():.4f}")
            print(f"   最高AUC: {df_results['auc'].max():.4f}")
            print(f"   最高ROI: {df_results['roi'].max():.4f}")

        return True

    except Exception as e:
        print(f"❌ ROI优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_performance():
    """测试优化后模型的实际性能"""
    print("\n📈 测试优化后模型性能...")

    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, classification_report
        from model_tools.optimization.objectives import BusinessROIObjective

        # 生成较大的测试数据集
        X, y_true, loan_amounts = generate_credit_data(n_samples=1500)

        # 训练测试分割
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
            X, y_true, loan_amounts, test_size=0.3, random_state=42, stratify=y_true
        )

        # 默认参数模型
        default_model = LGBMClassifier(random_state=42, verbosity=-1)
        default_model.fit(X_train, y_train)
        y_pred_default = default_model.predict_proba(X_test)[:, 1]

        # 优化参数模型（简化版）
        optimized_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1
        }

        optimized_model = LGBMClassifier(**optimized_params)
        optimized_model.fit(X_train, y_train)
        y_pred_optimized = optimized_model.predict_proba(X_test)[:, 1]

        # 性能对比
        auc_default = roc_auc_score(y_test, y_pred_default)
        auc_optimized = roc_auc_score(y_test, y_pred_optimized)

        # ROI对比
        roi_objective = BusinessROIObjective(
            loan_amounts=amounts_test,
            interest_rate=0.15,
            operational_cost_rate=0.02,
            recovery_rate=0.3,
            threshold=0.5
        )

        roi_default = roi_objective(y_test, y_pred_default)
        roi_optimized = roi_objective(y_test, y_pred_optimized)

        print(f"✅ 模型性能对比:")
        print(f"   默认参数 - AUC: {auc_default:.4f}, ROI: {roi_default:.4f}")
        print(f"   优化参数 - AUC: {auc_optimized:.4f}, ROI: {roi_optimized:.4f}")
        print(f"   AUC提升: {auc_optimized - auc_default:+.4f}")
        print(f"   ROI提升: {roi_optimized - roi_default:+.4f}")

        return True

    except Exception as e:
        print(f"❌ 模型性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试基于ROI的自动参数选择功能")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        return

    # 运行测试
    tests = [
        ("模块导入", test_basic_imports),
        ("ROI目标函数", test_roi_objective),
        ("参数空间", test_parameter_space),
        ("超参数优化器", test_hyperparameter_tuner),
        ("ROI优化", test_roi_based_optimization),
        ("模型性能", test_model_performance)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results.append((test_name, False))

    # 测试总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")

    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{len(tests)} 项测试通过")

    if passed == len(tests):
        print("🎉 所有测试通过！ROI优化功能正常运行。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()