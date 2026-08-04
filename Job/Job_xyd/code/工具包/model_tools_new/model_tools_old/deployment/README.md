# Deployment 模块

模型部署工具，覆盖从 `joblib` 训练产物到线上 PMML 的完整链路：格式转换、双端预测封装、以及线下/线上一致性校验。

## 主要能力

### 1. pkl → pmml 转换
- `ModelDeployer.convert_pkl_to_pmml(pkl_path, pmml_path, feature_names, target_name)`
- 自动识别 `PMMLPipeline` / `Pipeline` / 单一估计器，按需包成 `PMMLPipeline`
- 注入 `active_fields` / `target_fields`，避免线上字段名错位
- 依赖 `sklearn2pmml` + 本机 Java（需要时安装：`pip install sklearn2pmml`）

### 2. 模型加载与预测
- `load_pkl` / `load_pmml`（基于 `pypmml`）
- `predict_with_pkl(model, X, mode='auto')`：分类自动走 `predict_proba`，回归走 `predict`
- `predict_with_pmml(...)`：自动定位 `probability(<positive_label>)` / `predicted_*`，找不到时按规则兜底并 `warn`

### 3. pkl vs PMML 一致性验证（核心）
- `ModelDeployer.verify_consistency(pkl_path, pmml_path, X, atol, rtol, ...)`
- 输出 `ConsistencyResult`：
  - `is_consistent`、`max_abs_diff`、`mean/median/p99_abs_diff`
  - `n_mismatch` / `mismatch_ratio`（基于 `np.isclose`）
  - 自动剔除 NaN，长度对齐校验
  - Top-K 差异明细表（`diff_detail`）
  - 支持 `save_report=path.json` 落盘 JSON 报告

## 便捷函数

```python
from model_tools.Deployment import (
    ModelDeployer, ConsistencyResult,
    convert_pkl_to_pmml, verify_pkl_pmml_consistency,
)

# pkl -> pmml
convert_pkl_to_pmml(
    pkl_path='m.pkl',
    pmml_path='m.pmml',
    feature_names=feats,
    target_name='label',
)

# 一致性校验
result = verify_pkl_pmml_consistency(
    pkl_path='m.pkl',
    pmml_path='m.pmml',
    X=X_test,
    feature_names=feats,
    atol=1e-6,
    save_report='consistency_report.json',
)
print(result.summary())
```

## 依赖说明

| 用途 | 包 | 是否必选 |
| --- | --- | --- |
| 加载 pkl | `joblib` | 必选 |
| 数值/数据 | `numpy`、`pandas`、`scikit-learn` | 必选 |
| pkl → pmml | `sklearn2pmml`（需 JDK 8+） | 转换时必选 |
| 加载 PMML 评分 | `pypmml` | 一致性校验时必选 |
| 树模型导出 | `lightgbm`、`xgboost` | 可选 |

## 上线前确认

- 真实落 PMML 时本机需安装 JDK 8+（`java -version` 可见）
- 调 PMML 时建议传 `feature_names` 与训练顺序严格一致
- 分类模型类别非 0/1 时，通过 `positive_label` 指定正类
- 验证样本建议 1k~10w，使用真实分布的代表性数据
