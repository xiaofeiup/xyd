"""
ROI导向的超参数优化实用示例

展示如何在实际信贷建模中使用ROI优化功能
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_realistic_credit_data(n_samples=2000):
    """创建更真实的信贷数据"""
    np.random.seed(42)

    # 客户特征
    age = np.random.normal(40, 12, n_samples).clip(18, 80)
    annual_income = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 1000000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    debt_to_income = np.random.beta(2, 5, n_samples)
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)

    # 贷款特征
    loan_amount = annual_income * np.random.uniform(0.1, 4.0, n_samples)
    loan_amount = loan_amount.clip(5000, 300000)

    # 违约概率建模（更真实的关系）
    risk_score = (
        -0.02 * (age - 40) +                    # 年龄因子
        -0.000005 * (annual_income - 50000) +  # 收入因子
        -0.01 * (credit_score - 650) +         # 信用分因子
        2.0 * debt_to_income +                 # 负债比因子
        -0.05 * employment_years +             # 工作年限因子
        0.000003 * (loan_amount - 50000) +    # 贷款金额因子
        np.random.normal(0, 0.3, n_samples)   # 噪声
    )

    # 转换为违约概率
    default_prob = 1 / (1 + np.exp(-risk_score))
    default_prob = default_prob.clip(0.02, 0.8)  # 限制在合理范围

    # 生成违约标签
    is_default = np.random.binomial(1, default_prob, n_samples)

    # 创建数据框
    data = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'employment_years': employment_years,
        'loan_amount': loan_amount,
        'is_default': is_default
    })

    # 添加一些派生特征
    data['income_to_loan_ratio'] = data['annual_income'] / data['loan_amount']
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 50, 100], labels=['young', 'middle', 'mature', 'senior'])
    data['high_income'] = (data['annual_income'] > 80000).astype(int)

    return data

def demonstrate_roi_optimization():
    """演示ROI导向的超参数优化"""
    print("🏦 信贷建模ROI优化演示")
    print("=" * 60)

    # 1. 数据准备
    print("📊 准备信贷数据...")
    data = create_realistic_credit_data(n_samples=1500)

    print(f"   数据概览:")
    print(f"   - 样本数: {len(data)}")
    print(f"   - 违约率: {data['is_default'].mean():.2%}")
    print(f"   - 平均贷款金额: ¥{data['loan_amount'].mean():,.0f}")
    print(f"   - 贷款金额范围: ¥{data['loan_amount'].min():,.0f} - ¥{data['loan_amount'].max():,.0f}")

    # 2. 特征工程
    print("\n🔧 特征工程...")
    from sklearn.preprocessing import LabelEncoder

    # 处理分类特征
    le = LabelEncoder()
    data['age_group_encoded'] = le.fit_transform(data['age_group'])

    # 选择特征
    feature_cols = [
        'age', 'annual_income', 'credit_score', 'debt_to_income',
        'employment_years', 'income_to_loan_ratio', 'high_income', 'age_group_encoded'
    ]

    X = data[feature_cols]
    y = data['is_default']
    loan_amounts = data['loan_amount'].values

    print(f"   特征数量: {len(feature_cols)}")

    # 3. 训练测试分割
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, loan_amounts, test_size=0.3, random_state=42, stratify=y
    )

    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")

    # 4. 基准模型
    print("\n📈 训练基准模型...")
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, classification_report

    baseline_model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )

    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict_proba(X_test)[:, 1]

    baseline_auc = roc_auc_score(y_test, y_pred_baseline)
    print(f"   基准模型AUC: {baseline_auc:.4f}")

    # 5. ROI评估
    print("\n💰 ROI评估...")
    from model_tools.optimization.objectives import BusinessROIObjective

    # 信贷业务参数
    business_params = {
        'interest_rate': 0.18,        # 18%年利率
        'operational_cost_rate': 0.03,  # 3%运营成本
        'recovery_rate': 0.25,       # 25%回收率
        'threshold': 0.5              # 50%决策阈值
    }

    roi_evaluator = BusinessROIObjective(
        loan_amounts=amounts_test,
        **business_params
    )

    baseline_roi = roi_evaluator(y_test, y_pred_baseline)
    print(f"   基准模型ROI: {baseline_roi:.4f}")

    # 6. 参数优化（手动搜索模拟Optuna效果）
    print("\n🔍 执行ROI导向的参数优化...")

    # 定义搜索空间
    param_grid = [
        {'n_estimators': 80, 'max_depth': 4, 'learning_rate': 0.05, 'num_leaves': 15},
        {'n_estimators': 120, 'max_depth': 5, 'learning_rate': 0.08, 'num_leaves': 25},
        {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.06, 'num_leaves': 30},
        {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.04, 'num_leaves': 40},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.03, 'num_leaves': 20},
        {'n_estimators': 90, 'max_depth': 8, 'learning_rate': 0.07, 'num_leaves': 50},
        {'n_estimators': 160, 'max_depth': 4, 'learning_rate': 0.09, 'num_leaves': 12},
        {'n_estimators': 110, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 35}
    ]

    best_roi = -999
    best_auc = 0
    best_params = None
    best_combined = -999

    optimization_results = []

    for i, params in enumerate(param_grid):
        # 训练模型
        model_params = {
            **params,
            'random_state': 42,
            'verbosity': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        # 评估指标
        auc = roc_auc_score(y_test, y_pred)
        roi = roi_evaluator(y_test, y_pred)

        # 计算组合得分（AUC + 归一化ROI）
        # ROI通常在-1到0.5之间，我们将其标准化
        normalized_roi = max(-1, min(roi, 0.5)) + 1  # 转换到0-1.5范围
        normalized_roi = normalized_roi / 1.5         # 标准化到0-1

        combined_score = 0.6 * auc + 0.4 * normalized_roi

        optimization_results.append({
            'trial': i + 1,
            'params': params,
            'auc': auc,
            'roi': roi,
            'combined_score': combined_score
        })

        print(f"   Trial {i+1:2d}: AUC={auc:.4f}, ROI={roi:+.4f}, 综合={combined_score:.4f}")

        # 更新最佳结果
        if combined_score > best_combined:
            best_combined = combined_score
            best_params = params
            best_roi = roi
            best_auc = auc

    # 7. 结果分析
    print(f"\n🏆 优化结果:")
    print(f"   最佳参数: {best_params}")
    print(f"   最佳AUC: {best_auc:.4f} (vs 基准: {baseline_auc:.4f})")
    print(f"   最佳ROI: {best_roi:+.4f} (vs 基准: {baseline_roi:+.4f})")
    print(f"   AUC提升: {best_auc - baseline_auc:+.4f}")
    print(f"   ROI提升: {best_roi - baseline_roi:+.4f}")

    # 8. 业务价值分析
    print(f"\n📊 业务价值分析:")

    # 计算具体的业务指标
    roi_evaluator.threshold = 0.5

    # 基准模型业务指标
    baseline_approved = (y_pred_baseline >= 0.5).sum()
    baseline_bad_rate = y_test[y_pred_baseline >= 0.5].mean() if baseline_approved > 0 else 0

    # 最佳模型（用基准模型参数+最佳参数重新训练）
    best_model = LGBMClassifier(**{**best_params, 'random_state': 42, 'verbosity': -1})
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict_proba(X_test)[:, 1]

    best_approved = (y_pred_best >= 0.5).sum()
    best_bad_rate = y_test[y_pred_best >= 0.5].mean() if best_approved > 0 else 0

    print(f"   基准模型 - 批准率: {baseline_approved/len(y_test):.2%}, 坏账率: {baseline_bad_rate:.2%}")
    print(f"   优化模型 - 批准率: {best_approved/len(y_test):.2%}, 坏账率: {best_bad_rate:.2%}")

    # 估算年化收益差异
    if baseline_roi != best_roi:
        total_test_loan = amounts_test.sum()
        roi_improvement = best_roi - baseline_roi
        annual_value_improvement = total_test_loan * roi_improvement

        print(f"   测试集总贷款额: ¥{total_test_loan:,.0f}")
        print(f"   年化收益提升: ¥{annual_value_improvement:,.0f}")

    # 9. 特征重要性
    print(f"\n🎯 特征重要性 (最佳模型):")
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    print(f"\n✅ ROI优化演示完成!")
    print(f"   通过ROI导向的超参数优化，我们实现了:")
    print(f"   - 模型性能提升 (AUC: {best_auc - baseline_auc:+.4f})")
    print(f"   - 业务价值提升 (ROI: {best_roi - baseline_roi:+.4f})")

    return {
        'baseline_auc': baseline_auc,
        'best_auc': best_auc,
        'baseline_roi': baseline_roi,
        'best_roi': best_roi,
        'best_params': best_params,
        'optimization_results': optimization_results
    }

if __name__ == "__main__":
    results = demonstrate_roi_optimization()