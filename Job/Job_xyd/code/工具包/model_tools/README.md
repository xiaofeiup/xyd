# Model Tools

机器学习模型工具包，提供数据分析、特征选择、模型评估和可视化等功能。

## 安装

### 开发模式安装 (推荐)
```bash
pip install -e .
```

### 普通安装
```bash
pip install .
```

## 功能模块

### 1. 指标报告 (metric_report)
- `generate_metric_table`: 生成模型评估指标表
- `numerical_univerate`: 数值变量单变量分析
- `analyze_lift_performance`: 提升度性能分析

### 2. 特征选择 (feature_select)
- `by_missing_nunique_iv`: 基于缺失值、唯一值和IV值的特征筛选

### 3. 数据可视化 (plot)
- `plot_ks_curve`: 绘制KS曲线
- `plot_detailed_ks_analysis`: 详细KS分析图
- `plot_lift_gain`: 绘制提升度和增益图
- `plot_lift_curve`: 绘制提升度曲线
- `plot_combined_ks_lift_analysis`: 组合KS和提升度分析
- `metric_report_plot`: 指标报告可视化

### 4. 通用工具 (gentools)
- `calc_auc`: 计算AUC值
- `calculate_ks`: 计算KS值
- `calculate_lift`: 计算提升度
- `calc_gain`: 计算增益
- `calculate_psi`:计算psi，返回psi和分箱stat_df，一般<0.2算稳定

## 使用示例

```python
import model_tools as mt

# 计算AUC
auc_score = mt.calc_auc(y_true, y_pred)

# 绘制KS曲线
mt.plot_ks_curve(y_true, y_pred)

# 特征选择
selected_features = mt.by_missing_nunique_iv(df, target_col='target')

# 生成指标报告
report = mt.generate_metric_table(y_true, y_pred)
```

## 子模块使用

```python
# 也可以直接导入子模块
from model_tools.plot import plot_ks_curve
from model_tools.gentools import calc_auc
from model_tools.feature_select import by_missing_nunique_iv
```

## 依赖要求

- Python >= 3.7
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## 许可证

MIT License 


## 版本记录

- 0.1.0: 初始版本
- 0.1.1: 添加特征iv并行计算