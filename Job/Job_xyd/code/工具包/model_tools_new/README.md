# Model Tools 2.0

专业的机器学习模型工具包，提供特征工程、模型评估、监控和报告功能。

## 🚀 主要特性

- **特征工程**: 特征选择、重要性分析、质量验证
- **模型评估**: AUC、KS、Lift、PSI等全面评估指标
- **模型监控**: 实时性能监控、特征漂移检测、自动报警
- **可视化**: 丰富的模型评估和监控图表
- **报告生成**: 自动化的评估和监控报告
- **工具装饰器**: 提升开发效率的实用装饰器

## 📦 安装

### 基础安装

```bash
pip install model-tools
```

### 开发环境安装

```bash
git clone https://github.com/your-org/model-tools.git
cd model-tools
pip install -e .
```

### 可选依赖

```bash
# 机器学习增强包
pip install model-tools[ml]

# 高级可视化
pip install model-tools[viz]

# 监控集成
pip install model-tools[monitoring]

# 全部功能
pip install model-tools[all]
```

## 🏃‍♂️ 快速开始

### 基础使用

```python
import model_tools as mt
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'target': np.random.choice([0, 1], 1000)
})

# 数据质量检查
quality_report = mt.check_data_quality(data, target_col='target')
print(f"数据质量评估: {quality_report['basic_info']}")

# 特征选择
X, y = data.drop('target', axis=1), data['target']
selector = mt.FeatureSelector(method='iv', iv_threshold=0.1)
X_selected = selector.fit_transform(X, y, target_col='target')
print(f"特征选择: {X.shape[1]} -> {X_selected.shape[1]}")

# 模型评估
y_pred = np.random.random(len(y))  # 模拟预测概率
auc = mt.calculate_auc(y, y_pred)
ks, _, _ = mt.calculate_ks(y, y_pred)
print(f"模型性能: AUC={auc:.3f}, KS={ks:.3f}")
```

### 模型监控

```python
# 创建监控器
stability_monitor = mt.ModelStabilityMonitor(
    model_name="my_model",
    thresholds={'auc_drop_threshold': 0.05}
)

# 设置基准
baseline_metrics = stability_monitor.set_baseline(y_train, y_pred_train)

# 监控新数据
monitor_result = stability_monitor.monitor_performance(y_test, y_pred_test)
print(f"监控状态: {monitor_result['status']}")

# 特征漂移检测
drift_detector = mt.FeatureDriftDetector(
    feature_names=X.columns.tolist(),
    psi_threshold=0.25
)
drift_detector.set_baseline(X_train)
drift_result = drift_detector.detect_drift(X_test)
print(f"特征漂移率: {drift_result['drift_rate']:.2%}")
```

### 报警系统

```python
# 配置报警
alert_config = {
    'enable_email': False,
    'enable_log': True,
    'enable_webhook': True,
    'webhook_url': 'https://your-webhook-url.com'
}

alert_manager = mt.AlertManager(alert_config)

# 发送测试报警
alert_manager.send_alert(
    alert_type="MODEL_DEGRADATION",
    message="模型AUC下降超过阈值",
    severity="HIGH"
)
```

## 📚 详细文档

### 核心模块

#### 1. 评估模块 (evaluation)

提供全面的模型评估功能：

```python
from model_tools.evaluation import ModelEvaluator, calculate_auc, calculate_ks

# 创建评估器
evaluator = ModelEvaluator("my_model")

# 二分类评估
result = evaluator.evaluate_binary_classification(y_true, y_pred)
print(result['basic_metrics'])  # AUC, KS, Precision, Recall等
print(result['lift_metrics'])   # Lift分析结果
```

#### 2. 特征模块 (features)

特征工程和验证工具：

```python
from model_tools.features import FeatureSelector, FeatureImportanceAnalyzer

# 特征选择
selector = FeatureSelector(
    method='iv',                    # 选择方法
    single_value_threshold=0.95,    # 单一值阈值
    iv_threshold=0.1,              # IV阈值
    k_features=20                  # 最终特征数
)

# 特征重要性分析
analyzer = FeatureImportanceAnalyzer()
methods = ['random_forest', 'iv', 'correlation']
importance = analyzer.calculate_all_importance(X, y, methods)
consensus = analyzer.get_consensus_ranking(methods, top_k=10)
```

#### 3. 监控模块 (monitoring)

实时模型监控：

```python
from model_tools.monitoring import (
    ModelStabilityMonitor,
    FeatureDriftDetector,
    MonitoringWorkflow
)

# 自动化监控工作流
workflow = MonitoringWorkflow(
    model_name="production_model",
    stability_monitor=stability_monitor,
    drift_detector=drift_detector,
    alert_manager=alert_manager
)

# 设置基准数据
workflow.setup_baseline(X_baseline, y_baseline, y_pred_baseline)

# 运行监控周期
def get_current_data():
    # 获取当前数据的函数
    return X_current, y_current, y_pred_current

# 启动自动监控
workflow.start_automated_monitoring(
    data_source=get_current_data,
    interval_minutes=60
)
```

#### 4. 工具模块 (utils)

实用工具和装饰器：

```python
from model_tools.utils import timer, retry, validate_inputs

# 性能计时
@timer
def expensive_function():
    # 耗时操作
    pass

# 自动重试
@retry(max_attempts=3, delay=1.0)
def unstable_api_call():
    # 可能失败的API调用
    pass

# 输入验证
@validate_inputs(
    data=lambda x: isinstance(x, pd.DataFrame),
    threshold=lambda x: 0 <= x <= 1
)
def process_data(data, threshold):
    return data
```

### 高级功能

#### 可视化

```python
from model_tools.evaluation import ModelVisualizationTool

# 创建可视化工具
viz = ModelVisualizationTool()

# 生成各种图表
viz.plot_roc_curve(y_true, y_scores, save_path="roc.png")
viz.plot_ks_curve(y_true, y_scores, save_path="ks.png")
viz.plot_lift_curve(y_true, y_scores, save_path="lift.png")

# 创建综合仪表板
viz.create_dashboard(
    y_true, y_scores,
    feature_importance=feature_importance_dict,
    save_path="dashboard.png"
)
```

#### 报告生成

```python
from model_tools.evaluation import ModelReportGenerator

# 创建报告生成器
generator = ModelReportGenerator("my_model", "./reports")

# 生成综合报告
report = generator.generate_comprehensive_report(
    evaluation_results=evaluation_results,
    monitoring_results=monitoring_results,
    drift_results=drift_results
)

# 导出报告
report_path = generator.export_report(report, format='excel')
print(f"报告已保存到: {report_path}")
```

#### 深度分析

```python
from model_tools.evaluation import ModelAnalyzer

# 创建分析器
analyzer = ModelAnalyzer("my_model")

# 性能分析
performance_analysis = analyzer.analyze_model_performance(y_true, y_scores)

# 特征影响分析
feature_analysis = analyzer.analyze_feature_impact(X, y, y_scores)

# 模型退化分析
degradation_analysis = analyzer.analyze_model_degradation(
    historical_performance, current_performance
)

# 生成健康报告
health_report = analyzer.generate_model_health_report(
    performance_analysis, feature_analysis, degradation_analysis
)
```

## 🔧 配置

### 全局配置

```python
import model_tools as mt

# 设置全局配置
mt.set_config('feature_selection.iv_threshold', 0.1)
mt.set_config('monitoring.psi_threshold', 0.25)
mt.set_config('visualization.figsize', (12, 8))

# 获取配置
threshold = mt.get_config('feature_selection.iv_threshold', 0.05)
```

### 环境变量

支持通过环境变量配置：

```bash
export MODEL_TOOLS_LOG_LEVEL=INFO
export MODEL_TOOLS_CACHE_DIR=/tmp/model_tools_cache
export MODEL_TOOLS_ALERT_WEBHOOK_URL=https://your-webhook.com
```

## 📊 示例项目

完整的使用示例请参考 `examples/` 目录：

- `basic_usage.py` - 基础功能演示
- `monitoring_example.py` - 模型监控示例
- `evaluation_example.py` - 模型评估示例
- `feature_engineering_example.py` - 特征工程示例

## 🧪 测试

运行测试：

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_evaluation.py

# 运行测试并生成覆盖率报告
pytest --cov=model_tools
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 这个项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-org/model-tools.git
cd model-tools

# 安装开发依赖
pip install -e .[dev]

# 安装pre-commit hooks
pre-commit install

# 运行代码格式化
black model_tools tests
flake8 model_tools tests
```

## 📝 更新日志

### 版本 2.0.0 (2024-01-XX)

- 🎉 完全重构的模块架构
- ✨ 新增模型监控和报警系统
- ✨ 新增特征漂移检测
- ✨ 新增可视化和报告生成
- ✨ 新增实用装饰器工具
- 🐛 修复已知问题
- 📚 完善文档和示例

### 版本 1.x.x

- 基础的模型评估功能
- 特征选择工具

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙋‍♂️ 支持

如果您遇到问题或有建议，请：

1. 查看 [文档](https://model-tools.readthedocs.io/)
2. 搜索 [现有问题](https://github.com/your-org/model-tools/issues)
3. 创建 [新问题](https://github.com/your-org/model-tools/issues/new)

## 🌟 致谢

感谢所有贡献者的努力！

- 核心开发团队
- 社区贡献者
- 依赖的开源项目

---

**Model Tools 2.0** - 让机器学习模型开发更简单、更可靠！
# 自动化建模流水线

现在可以通过配置驱动的公共 API 完成数据校验、数值特征分析、特征筛选、（可选）Optuna 调参、Train/OOS 评估，以及 HTML + Excel 报告生成。

```bash
PYTHONPATH=. python -m model_tools.cli \
  --auto-modeling-config model_tools/auto/pipeline_config.example.yaml
```

Python 调用：

```python
from model_tools.auto import AutoModelingConfig, AutoModelingPipeline

config = AutoModelingConfig.from_dict({...})
result = AutoModelingPipeline(config).run()
print(result.html_report_path)
```

每次运行会生成独立目录，包含 `model_report.html`、`model_delivery_report.xlsx`、`model.joblib`、`preprocessor.joblib`、`predictions.csv`、`metrics.json`、特征筛选日志和配置快照。输入支持单表 CSV/Parquet，也支持特征表 + 标签表按主键合并；关键字段和 Train/OOS 分区采用严格校验。
