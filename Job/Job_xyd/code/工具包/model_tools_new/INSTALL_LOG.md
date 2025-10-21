# Model Tools 安装记录

## 安装时间
2025-10-21

## 环境信息
- **环境名称**: mifeng_py38
- **Python版本**: Python 3.8.18
- **环境路径**: `/opt/anaconda3/envs/mifeng_py38`

## 安装详情

### 安装前操作
1. 卸载了旧版本的 model_tools
   - 旧版本位置: `/Users/mayongzhi/Job/Job_xyd/code/量化派联合建模/建模/model_tools_new`
   - 版本号: 2.0.0

### 安装命令
```bash
cd /Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new
/opt/anaconda3/envs/mifeng_py38/bin/pip install -e .
```

### 安装模式
- **可编辑模式 (Editable Install)**: 以开发模式安装,代码修改即时生效

### 包信息
- **包名**: model-tools
- **版本**: 2.0.0
- **描述**: 专业的机器学习模型工具包，提供特征工程、模型评估、监控和报告功能
- **安装路径**: `/Users/mayongzhi/Job/Job_xyd/code/工具包/model_tools_new`

### 依赖包
所有依赖包已满足,无需额外安装:
- pandas >= 1.3.0 (已安装: 2.0.3)
- numpy >= 1.20.0 (已安装: 1.24.4)
- scipy >= 1.7.0 (已安装: 1.10.1)
- scikit-learn >= 1.0.0 (已安装: 1.3.2)
- statsmodels >= 0.13.0 (已安装: 0.14.1)
- matplotlib >= 3.4.0 (已安装: 3.7.5)
- seaborn >= 0.11.0 (已安装: 0.13.2)
- openpyxl >= 3.0.0 (已安装: 3.1.5)
- psutil >= 5.8.0 (已安装: 5.8.0)
- typing-extensions >= 4.0.0 (已安装: 4.12.2)
- python-dateutil >= 2.8.0 (已安装: 2.9.0)
- PyYAML >= 5.4.0 (已安装: 6.0.2)
- requests >= 2.25.0 (已安装: 2.32.3)

## 最近更新

### 2025-10-21 更新内容
修改了特征验证模块 (`model_tools/features/validation.py`):

1. **新增配置参数**
   - 添加 `max_features_for_correlation: 500` 配置项
   - 用于控制执行特征间相关性计算的最大特征数量

2. **优化相关性计算**
   - 在 `validate_feature_correlations()` 方法中添加特征数量检查
   - 当数值型特征数量超过 500 时,自动跳过相关性计算
   - 避免在高维特征场景下因计算相关性矩阵导致长时间等待

3. **改进原因**
   - 相关性矩阵计算复杂度为 O(n²),特征数量过多时计算时间过长
   - 500个特征的相关性矩阵需要计算 124,750 个相关系数
   - 跳过后会在日志和返回结果中明确说明原因

## 验证安装

可以通过以下方式验证安装:

```python
# 导入测试
import model_tools
from model_tools.features.validation import FeatureValidator

# 查看版本
print(model_tools.__version__)

# 测试特征验证器
validator = FeatureValidator()
print(validator.config)
```

## 注意事项

1. 使用可编辑模式安装后,直接修改源代码即可生效,无需重新安装
2. 本次安装使用清华大学 PyPI 镜像源
3. pip 提示将来版本会强制使用 PEP517 标准,建议后续添加 `pyproject.toml` 文件

## 安装状态
✅ 安装成功
