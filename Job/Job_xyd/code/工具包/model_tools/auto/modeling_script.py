"""
使用 AntiOverfittingTuner 进行 LGB 建模
- 特征: ym.csv
- 标签: df.csv (flag, mid_sample_type)
- 调参: n_trials=30, 5折交叉验证
- 选参策略: train_auc / val_auc 差距最小且 val_auc 最高
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 添加工具包路径
sys.path.insert(0, os.path.expanduser('~/Job/Job_xyd/code/工具包/model_tools_new'))

import lightgbm as lgb
from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner

warnings.filterwarnings('ignore')


class MemoryEfficientTuner(AntiOverfittingTuner):
    """限制 LightGBM 并行度以降低内存峰值"""
    def _suggest_parameters(self, trial):
        params = super()._suggest_parameters(trial)
        params['n_jobs'] = 2  # 限制并行度
        return params

# ============ 1. 数据加载与合并 ============
DATA_DIR = '/Users/mayongzhi/Job/Job_xyd/数据/WorkBuddy_data/data_upload'

print("=" * 60)
print("步骤 1: 数据加载与合并")
print("=" * 60)

ym = pd.read_csv(os.path.join(DATA_DIR, 'ym.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, 'df.csv'))

# 按 test_id 合并
df_merged = ym.merge(df[['test_id', 'flag', 'mid_sample_type']], on='test_id', how='inner')
print(f"合并后形状: {df_merged.shape}")
print(f"mid_sample_type 分布:\n{df_merged['mid_sample_type'].value_counts()}")
print(f"flag 分布:\n{df_merged['flag'].value_counts()}")
print(f"坏样本率: {df_merged['flag'].mean():.4f}")

# ============ 2. 特征选择 ============
print("\n" + "=" * 60)
print("步骤 2: 特征选择")
print("=" * 60)

# 排除非特征列
exclude_cols = ['Unnamed: 0', 'mobile', 'apply_date', 'test_id', 'id_number',
                'flag', 'mid_sample_type','apply_date_label']
feas = [c for c in df_merged.columns if c not in exclude_cols]
print(f"特征数量: {len(feas)}")

# 检查特征中的非数值列
non_numeric = df_merged[feas].select_dtypes(exclude=['number']).columns.tolist()
if non_numeric:
    print(f"非数值特征 (将剔除): {non_numeric}")
    feas = [c for c in feas if c not in non_numeric]
    print(f"剔除后特征数量: {len(feas)}")
print(feas)
# 检查 NaN，剔除缺失率 > 50% 的特征
nan_ratio = df_merged[feas].isna().mean()
high_nan = nan_ratio[nan_ratio > 0.5].sort_values(ascending=False)
if len(high_nan) > 0:
    print(f"缺失率 > 50% 的特征 ({len(high_nan)} 个)，将剔除:")
    print(high_nan.head(10))
    feas = [c for c in feas if c not in high_nan.index]
    print(f"剔除后特征数量: {len(feas)}")
else:
    print("无缺失率 > 50% 的特征")

# 剔除方差为 0 的特征
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)
vt.fit(df_merged.loc[df_merged['mid_sample_type'] == '1-train', feas].fillna(-999))
zero_var_cols = [feas[i] for i in range(len(feas)) if vt.get_support()[i] == False]
if zero_var_cols:
    print(f"方差为 0 的特征 ({len(zero_var_cols)} 个): {zero_var_cols[:10]}")
    feas = [c for c in feas if c not in zero_var_cols]
    print(f"剔除后特征数量: {len(feas)}")

# ============ 3. 划分训练集和 OOS ============
print("\n" + "=" * 60)
print("步骤 3: 划分训练集和 OOS")
print("=" * 60)

train_mask = df_merged['mid_sample_type'] == '1-train'
oos_mask = df_merged['mid_sample_type'] == '3-oos'

X_train = df_merged.loc[train_mask, feas].copy()
y_train = df_merged.loc[train_mask, 'flag'].copy()
X_oos = df_merged.loc[oos_mask, feas].copy()
y_oos = df_merged.loc[oos_mask, 'flag'].copy()

print(f"训练集: {X_train.shape}, 坏样本率: {y_train.mean():.4f}")
print(f"OOS集:  {X_oos.shape}, 坏样本率: {y_oos.mean():.4f}")

# ============ 4. AntiOverfittingTuner 调参 ============
print("\n" + "=" * 60)
print("步骤 4: AntiOverfittingTuner 调参 (n_trials=5)")
print("=" * 60)

tuner = MemoryEfficientTuner(
    model_class=lgb.LGBMClassifier,
    model_type='lgb',
    direction='maximize'
)

result = tuner.optimize_anti_overfitting(
    X_train,
    y_train,
    n_trials=5,
    cv_folds=5,
    n_jobs=1,
    show_progress_bar=True
)

print(f"\n调参完成!")
print(f"  总 trial 数: {result['results_summary']['total_trials']}")
print(f"  有效 trial 数 (满足防过拟合条件): {result['results_summary']['valid_trials']}")
print(f"  过拟合 trial 比例: {result['results_summary']['overfitting_rate']:.1%}")
print(f"  平均 AUC gap: {result['results_summary']['avg_auc_gap']:.4f}")
print(f"  平均 KS gap: {result['results_summary']['avg_ks_gap']:.4f}")

# ============ 5. 从 performance_history 中选最优参数 ============
print("\n" + "=" * 60)
print("步骤 5: 参数选择 (train_auc/val_auc 差距最小且 val_auc 最高)")
print("=" * 60)

history = pd.DataFrame(result['performance_history'])
print(f"performance_history 记录数: {len(history)}")

# 计算选参指标
history['auc_gap_abs'] = (history['train_auc'] - history['val_auc']).abs()
history['ks_gap_abs'] = (history['train_ks'] - history['val_ks']).abs()

# 归一化到 [0,1]
val_auc_norm = (history['val_auc'] - history['val_auc'].min()) / (history['val_auc'].max() - history['val_auc'].min() + 1e-10)
gap_norm = (history['auc_gap_abs'] - history['auc_gap_abs'].min()) / (history['auc_gap_abs'].max() - history['auc_gap_abs'].min() + 1e-10)

# 综合评分: val_auc 越高越好, gap 越小越好 (权重各 50%)
history['composite_score'] = val_auc_norm * 0.5 + (1 - gap_norm) * 0.5

# 排序展示 Top 10
top_candidates = history.nlargest(10, 'composite_score')[
    ['trial', 'val_auc', 'train_auc', 'auc_gap_abs', 'val_ks', 'train_ks', 'ks_gap_abs', 'composite_score', 'is_overfitting']
].reset_index(drop=True)

print("\nTop 10 候选参数:")
print(top_candidates.to_string(index=False))

# 选出最优
best_row = history.loc[history['composite_score'].idxmax()]
best_params = best_row['params']

print(f"\n>>> 选中的 Trial #{best_row['trial']}:")
print(f"    val_auc = {best_row['val_auc']:.4f}, train_auc = {best_row['train_auc']:.4f}")
print(f"    auc_gap = {best_row['auc_gap_abs']:.4f}")
print(f"    val_ks  = {best_row['val_ks']:.4f}, train_ks  = {best_row['train_ks']:.4f}")
print(f"    ks_gap  = {best_row['ks_gap_abs']:.4f}")
print(f"    composite_score = {best_row['composite_score']:.4f}")
print(f"    is_overfitting  = {best_row['is_overfitting']}")
print(f"\n    参数:")
for k, v in sorted(best_params.items()):
    print(f"      {k}: {v}")

# ============ 6. 使用选中参数训练最终模型 ============
print("\n" + "=" * 60)
print("步骤 6: 训练最终模型并在 OOS 上评估")
print("=" * 60)

final_model = lgb.LGBMClassifier(**best_params)
final_model.set_params(n_jobs=2)
final_model.fit(X_train, y_train)

# 训练集评估
y_pred_train = final_model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_pred_train)

# OOS 评估
y_pred_oos = final_model.predict_proba(X_oos)[:, 1]
oos_auc = roc_auc_score(y_oos, y_pred_oos)

# KS 计算
def calc_ks(y_true, y_pred):
    df_tmp = pd.DataFrame({'y': y_true, 'p': y_pred}).sort_values('p', ascending=False)
    total_bad = df_tmp['y'].sum()
    total_good = (1 - df_tmp['y']).sum()
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = df_tmp['y'].cumsum() / total_bad
    cum_good = (1 - df_tmp['y']).cumsum() / total_good
    return (cum_bad - cum_good).abs().max()

train_ks = calc_ks(y_train.values, y_pred_train)
oos_ks = calc_ks(y_oos.values, y_pred_oos)

print(f"\n最终模型评估结果:")
print(f"  {'指标':<15} {'训练集':>10} {'OOS':>10} {'差距':>10}")
print(f"  {'AUC':<15} {train_auc:>10.4f} {oos_auc:>10.4f} {abs(train_auc - oos_auc):>10.4f}")
print(f"  {'KS':<15} {train_ks:>10.4f} {oos_ks:>10.4f} {abs(train_ks - oos_ks):>10.4f}")
print(f"\n  Train-OOS AUC gap: {abs(train_auc - oos_auc):.4f}")
print(f"  过拟合判定 (gap <= 0.05): {'通过' if abs(train_auc - oos_auc) <= 0.05 else '未通过'}")

# ============ 7. 保存结果 ============
print("\n" + "=" * 60)
print("步骤 7: 保存结果")
print("=" * 60)

# 保存 performance_history
history_export = history.drop(columns=['params']).copy()
history_export.to_csv(os.path.join(DATA_DIR, 'tuning_history.csv'), index=False)
print(f"调参历史已保存: tuning_history.csv")

# 保存最优参数
import json
params_path = os.path.join(DATA_DIR, 'best_params.json')
# 转换 numpy 类型为 python 原生类型
best_params_serializable = {}
for k, v in best_params.items():
    if isinstance(v, (np.integer,)):
        best_params_serializable[k] = int(v)
    elif isinstance(v, (np.floating,)):
        best_params_serializable[k] = float(v)
    elif isinstance(v, (np.bool_,)):
        best_params_serializable[k] = bool(v)
    else:
        best_params_serializable[k] = v

with open(params_path, 'w') as f:
    json.dump(best_params_serializable, f, indent=2, ensure_ascii=False)
print(f"最优参数已保存: best_params.json")

# 保存 OOS 预测结果
oos_pred_df = pd.DataFrame({
    'test_id': df_merged.loc[oos_mask, 'test_id'].values,
    'flag': y_oos.values,
    'pred_proba': y_pred_oos
})
oos_pred_df.to_csv(os.path.join(DATA_DIR, 'oos_predictions.csv'), index=False)
print(f"OOS 预测结果已保存: oos_predictions.csv")

print("\n" + "=" * 60)
print("建模流程完成!")
print("=" * 60)
