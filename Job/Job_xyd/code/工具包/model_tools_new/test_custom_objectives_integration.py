"""
演示如何将HyperparameterTuner与自定义目标函数（如KSObjective）集成使用
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# 导入相关模块
from model_tools.optimization.hyperparameter_tuner import HyperparameterTuner
from model_tools.optimization.objectives import KSObjective, AUCObjective, BusinessROIObjective
from lightgbm import LGBMClassifier

class CustomObjectiveHyperparameterTuner(HyperparameterTuner):
    """
    增强版超参数调优器，支持自定义目标函数
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_objective_func = None
        self.objective_params = {}

    def set_custom_objective(self, objective_func, **objective_params):
        """
        设置自定义目标函数

        Args:
            objective_func: 目标函数实例 (如KSObjective())
            **objective_params: 目标函数的额外参数
        """
        self.custom_objective_func = objective_func
        self.objective_params = objective_params

    def optimize_with_custom_objective(
        self,
        X,
        y,
        n_trials=50,
        cv=5,
        n_jobs=1,
        show_progress_bar=True
    ):
        """
        使用自定义目标函数进行优化

        Args:
            X: 特征数据
            y: 目标变量
            n_trials: 试验次数
            cv: 交叉验证折数
            n_jobs: 并行数
            show_progress_bar: 是否显示进度条

        Returns:
            优化结果
        """
        if self.custom_objective_func is None:
            raise ValueError("请先使用set_custom_objective()设置目标函数")

        # 设置交叉验证
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        def objective(trial):
            try:
                # 获取参数
                params = self._suggest_parameters(trial)

                # 创建模型
                model = self.model_class(**params)

                # 使用交叉验证获取预测结果
                y_pred_proba = cross_val_predict(
                    model, X, y,
                    cv=cv_splitter,
                    method='predict_proba',
                    n_jobs=1
                )[:, 1]  # 获取正类概率

                # 使用自定义目标函数计算得分
                score = self.custom_objective_func(y, y_pred_proba, **self.objective_params)

                # 记录历史
                self.optimization_history_.append({
                    'trial': trial.number,
                    'score': score,
                    'params': params.copy(),
                    'objective_type': type(self.custom_objective_func).__name__
                })

                return score

            except Exception as e:
                warnings.warn(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf') if self.direction == 'maximize' else float('inf')

        # 执行优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )

        # 保存最佳结果
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history_,
            'objective_type': type(self.custom_objective_func).__name__
        }


def demonstrate_custom_objectives():
    """演示不同自定义目标函数的使用"""
    print("🎯 自定义目标函数集成演示")
    print("=" * 60)

    # 生成模拟信贷数据
    np.random.seed(42)
    n_samples = 1000

    # 特征数据
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.uniform(0, 1, n_samples),
        'feature4': np.random.exponential(1, n_samples)
    })

    # 生成标签（基于特征的逻辑关系）
    y_prob = 1 / (1 + np.exp(-(X['feature1'] * 0.5 + X['feature2'] * 0.3 - X['feature3'] * 0.8)))
    y = np.random.binomial(1, y_prob, n_samples)

    # 贷款金额（用于ROI计算）
    loan_amounts = np.random.uniform(10000, 100000, n_samples)

    print(f"📊 数据概览:")
    print(f"   样本数: {n_samples}")
    print(f"   违约率: {y.mean():.2%}")
    print(f"   特征数: {X.shape[1]}")

    # 1. 使用KS目标函数优化
    print(f"\n🔍 1. 使用KS目标函数优化:")

    ks_tuner = CustomObjectiveHyperparameterTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        study_name='ks_optimization'
    )

    # 设置KS目标函数
    ks_objective = KSObjective()
    ks_tuner.set_custom_objective(ks_objective)

    # 执行优化
    ks_result = ks_tuner.optimize_with_custom_objective(X, y, n_trials=10, cv=3, show_progress_bar=False)

    print(f"   最佳KS值: {ks_result['best_score']:.4f}")
    print(f"   最佳参数: max_depth={ks_result['best_params'].get('max_depth', 'N/A')}")

    # 2. 使用AUC目标函数优化
    print(f"\n📈 2. 使用AUC目标函数优化:")

    auc_tuner = CustomObjectiveHyperparameterTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        study_name='auc_optimization'
    )

    # 设置AUC目标函数
    auc_objective = AUCObjective()
    auc_tuner.set_custom_objective(auc_objective)

    # 执行优化
    auc_result = auc_tuner.optimize_with_custom_objective(X, y, n_trials=10, cv=3, show_progress_bar=False)

    print(f"   最佳AUC值: {auc_result['best_score']:.4f}")
    print(f"   最佳参数: learning_rate={auc_result['best_params'].get('learning_rate', 'N/A'):.4f}")

    # 3. 使用ROI目标函数优化
    print(f"\n💰 3. 使用ROI目标函数优化:")

    roi_tuner = CustomObjectiveHyperparameterTuner(
        model_class=LGBMClassifier,
        model_type='lgb',
        direction='maximize',
        study_name='roi_optimization'
    )

    # 设置ROI目标函数
    roi_objective = BusinessROIObjective(
        loan_amounts=loan_amounts,
        interest_rate=0.18,
        operational_cost_rate=0.03,
        recovery_rate=0.25,
        threshold=0.5
    )
    roi_tuner.set_custom_objective(roi_objective)

    # 执行优化
    roi_result = roi_tuner.optimize_with_custom_objective(X, y, n_trials=10, cv=3, show_progress_bar=False)

    print(f"   最佳ROI值: {roi_result['best_score']:.4f}")
    print(f"   最佳参数: n_estimators={roi_result['best_params'].get('n_estimators', 'N/A')}")

    # 4. 结果对比
    print(f"\n📊 不同目标函数优化结果对比:")

    results_comparison = pd.DataFrame([
        {
            'objective': 'KS',
            'best_score': ks_result['best_score'],
            'max_depth': ks_result['best_params'].get('max_depth', 0),
            'learning_rate': ks_result['best_params'].get('learning_rate', 0),
            'n_estimators': ks_result['best_params'].get('n_estimators', 0)
        },
        {
            'objective': 'AUC',
            'best_score': auc_result['best_score'],
            'max_depth': auc_result['best_params'].get('max_depth', 0),
            'learning_rate': auc_result['best_params'].get('learning_rate', 0),
            'n_estimators': auc_result['best_params'].get('n_estimators', 0)
        },
        {
            'objective': 'ROI',
            'best_score': roi_result['best_score'],
            'max_depth': roi_result['best_params'].get('max_depth', 0),
            'learning_rate': roi_result['best_params'].get('learning_rate', 0),
            'n_estimators': roi_result['best_params'].get('n_estimators', 0)
        }
    ])

    print(results_comparison.round(4))

    # 5. 实际模型性能验证
    print(f"\n✅ 实际模型性能验证:")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, loan_amounts, test_size=0.3, random_state=42, stratify=y
    )

    # 训练最佳KS模型
    ks_model = LGBMClassifier(**ks_result['best_params'], random_state=42, verbosity=-1)
    ks_model.fit(X_train, y_train)
    ks_pred = ks_model.predict_proba(X_test)[:, 1]

    # 验证指标
    test_auc = roc_auc_score(y_test, ks_pred)
    test_ks = ks_objective(y_test, ks_pred)
    test_roi = roi_objective(y_test, ks_pred)

    print(f"   KS优化模型在测试集上:")
    print(f"     AUC: {test_auc:.4f}")
    print(f"     KS:  {test_ks:.4f}")
    print(f"     ROI: {test_roi:.4f}")

    return {
        'ks_result': ks_result,
        'auc_result': auc_result,
        'roi_result': roi_result,
        'comparison': results_comparison
    }


def show_usage_examples():
    """展示具体的使用方法"""
    print(f"\n💡 具体使用方法:")

    usage_code = '''
# 1. 创建增强版调优器
from model_tools.optimization.objectives import KSObjective
from lightgbm import LGBMClassifier

tuner = CustomObjectiveHyperparameterTuner(
    model_class=LGBMClassifier,
    model_type='lgb',
    direction='maximize'
)

# 2. 设置KS目标函数
ks_objective = KSObjective()
tuner.set_custom_objective(ks_objective)

# 3. 执行优化
result = tuner.optimize_with_custom_objective(X, y, n_trials=50)

# 4. 获取最佳模型
best_model = tuner.get_best_model(X, y)
'''

    print(usage_code)


if __name__ == "__main__":
    # 运行演示
    results = demonstrate_custom_objectives()

    # 显示使用方法
    show_usage_examples()

    print(f"\n🎉 演示完成！成功集成了KSObjective等自定义目标函数。")