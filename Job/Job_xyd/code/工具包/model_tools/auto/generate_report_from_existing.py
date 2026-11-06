"""
基于已有调参结果生成完整 HTML 报告 v2
- 调用 ModelDeliveryReport.generate_full_report() 获取全部报表
- 生成可排序/可搜索/全量展示的 HTML 报告
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

TOOLKIT_PATH = os.path.expanduser('~/Job/Job_xyd/code/工具包/model_tools_new')
sys.path.insert(0, TOOLKIT_PATH)

import lightgbm as lgb
from model_tools.features.data_analysis_optimized import DataAnalyzer
from model_tools.evaluation.delivery_report import ModelDeliveryReport
from html_report_generator import generate_data_analysis_html, generate_model_report_html

DATA_DIR = '/Users/mayongzhi/Job/Job_xyd/数据/WorkBuddy_data/data_upload'
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载数据
print("加载数据...")
ym = pd.read_csv(os.path.join(DATA_DIR, 'ym.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, 'df.csv'))
data = ym.merge(df[['test_id', 'flag', 'mid_sample_type', 'apply_date']], on='test_id', how='inner', suffixes=('', '_label'))

# 特征处理
exclude_cols = ['Unnamed: 0', 'mobile', 'apply_date', 'test_id', 'id_number',
                'flag', 'mid_sample_type', 'partner_code',
                '蜜蜂分_中利率版7_2_proba', '蜜蜂分_中利率版6_2_proba', '蜜蜂分_中利率版2_2_proba','apply_date_label']
feas = [c for c in data.columns if c not in exclude_cols]
feas = data[feas].select_dtypes(include=[np.number]).columns.tolist()
nan_ratio = data[feas].isna().mean()
feas = [f for f in feas if nan_ratio[f] <= 0.5]
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)
vt.fit(data[feas].fillna(-999))
feas = [feas[i] for i in range(len(feas)) if vt.get_support()[i]]
print(f"特征数: {len(feas)}")

# 2. 划分 + 训练
train_mask = data['mid_sample_type'] == '1-train'
oos_mask = data['mid_sample_type'] == '3-oos'
X_train = data.loc[train_mask, feas].copy()
y_train = data.loc[train_mask, 'flag'].copy()
X_oos = data.loc[oos_mask, feas].copy()
y_oos = data.loc[oos_mask, 'flag'].copy()

with open(os.path.join(DATA_DIR, 'best_params.json')) as f:
    best_params = json.load(f)
best_params['n_jobs'] = 1

print("训练模型...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_oos = model.predict_proba(X_oos)[:, 1]

train_auc = roc_auc_score(y_train, y_pred_train)
oos_auc = roc_auc_score(y_oos, y_pred_oos)
print(f"训练集 AUC={train_auc:.4f}, OOS AUC={oos_auc:.4f}")

# 3. 特征重要性
feature_importance = pd.DataFrame({
    'feature': feas,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 4. 调参历史
tuning_history = pd.read_csv(os.path.join(DATA_DIR, 'tuning_history.csv'))

# 5. ===== 数据分析 HTML (可排序/搜索/全量) =====
print("生成数据分析 HTML...")
analyzer = DataAnalyzer(data, n_jobs=1)
generate_data_analysis_html(
    analyzer,
    features=feas,
    save_path=os.path.join(OUTPUT_DIR, 'step1_data_analysis.html'),
    title="数据分析报告 - LGB 建模"
)

# 6. ===== 调用 generate_full_report() =====
print("生成模型交付报告 (generate_full_report)...")
report_data_all = data.copy()
report_data_all['pred_score'] = 0.0
report_data_all.loc[train_mask, 'pred_score'] = y_pred_train
report_data_all.loc[oos_mask, 'pred_score'] = y_pred_oos

type_map = {'1-train': 'train', '3-oos': 'test'}
report_data_all['sample_type_report'] = report_data_all['mid_sample_type'].map(type_map).fillna('all')
report_data_all['apply_date'] = pd.to_datetime(report_data_all['apply_date'], errors='coerce')

try:
    reporter = ModelDeliveryReport(
        data=report_data_all,
        target_col='flag',
        score_col='pred_score',
        date_col='apply_date',
        sample_type_col='sample_type_report',
        score_direction='higher_is_bad'
    )
    delivery_report = reporter.generate_full_report(
        features=feas[:20],  # TOP20 特征
        model=model,
        importance_type='gain',
        n_bins=10
    )
    print(f"generate_full_report 返回 {len([k for k in delivery_report if not k.startswith('_')])} 个报表")
    for k, v in delivery_report.items():
        if not k.startswith('_'):
            shape = v.shape if hasattr(v, 'shape') else 'N/A'
            print(f"  - {k}: {shape}")
except Exception as e:
    print(f"generate_full_report 失败: {e}")
    import traceback; traceback.print_exc()
    delivery_report = {}

# 7. ===== 生成模型 HTML 报告 =====
print("生成模型 HTML 报告...")
generate_model_report_html(
    report_data=delivery_report,
    y_true_train=y_train.values,
    y_score_train=y_pred_train,
    y_true_oos=y_oos.values,
    y_score_oos=y_pred_oos,
    best_params=best_params,
    tuning_history=tuning_history,
    feature_importance=feature_importance,
    save_path=os.path.join(OUTPUT_DIR, 'model_report.html'),
    title="模型交付报告 - LGB 防过拟合调参"
)

print(f"\n完成! 报告位置: {OUTPUT_DIR}")
