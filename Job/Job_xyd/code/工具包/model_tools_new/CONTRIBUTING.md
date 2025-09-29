# 贡献指南

欢迎为 Model Tools 2.0 项目做出贡献！本文档将帮助您了解如何参与项目开发。

## 🚀 快速开始

### 环境准备

1. **Fork 项目**
   ```bash
   # 在 GitHub 上 fork 项目到您的账户
   # 然后 clone 到本地
   git clone https://github.com/your-username/model-tools.git
   cd model-tools
   ```

2. **设置开发环境**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows

   # 安装开发依赖
   pip install -e .[dev]

   # 安装 pre-commit hooks
   pre-commit install
   ```

3. **运行测试**
   ```bash
   # 运行所有测试
   pytest

   # 运行特定模块测试
   pytest tests/test_evaluation.py

   # 生成覆盖率报告
   pytest --cov=model_tools --cov-report=html
   ```

## 📋 贡献类型

我们欢迎以下类型的贡献：

### 🐛 Bug 修复
- 修复现有功能的错误
- 提升代码稳定性
- 处理边界情况

### ✨ 新功能
- 添加新的评估指标
- 实现新的特征选择算法
- 扩展监控能力
- 新增可视化图表

### 📚 文档改进
- 修正文档错误
- 添加使用示例
- 完善API文档
- 翻译文档

### 🧪 测试增强
- 增加测试用例
- 提升测试覆盖率
- 性能测试
- 集成测试

### 🔧 性能优化
- 算法优化
- 内存使用优化
- 计算效率提升

## 🔄 开发流程

### 1. 创建分支
```bash
# 从 main 分支创建新分支
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 分支命名规范：
# feature/功能名称 - 新功能
# bugfix/问题描述 - bug修复
# docs/文档主题 - 文档更新
# refactor/重构内容 - 代码重构
```

### 2. 开发代码
- 遵循代码规范（见下文）
- 添加必要的测试
- 更新相关文档
- 确保测试通过

### 3. 提交更改
```bash
# 提交规范：使用约定式提交
git add .
git commit -m "type(scope): description"

# 提交类型：
# feat: 新功能
# fix: bug修复
# docs: 文档更新
# style: 代码格式
# refactor: 重构
# test: 测试相关
# chore: 构建过程或辅助工具的变动
```

### 4. 推送和创建 PR
```bash
git push origin feature/your-feature-name
# 然后在 GitHub 上创建 Pull Request
```

## 📝 代码规范

### Python 代码风格
我们使用以下工具保证代码质量：

```bash
# 代码格式化
black model_tools tests examples

# 代码检查
flake8 model_tools tests

# 类型检查
mypy model_tools

# 导入排序
isort model_tools tests examples
```

### 代码规范要求

1. **PEP 8 兼容**：遵循 Python 官方代码规范
2. **类型提示**：为公共 API 添加类型注解
3. **文档字符串**：使用 Google 风格的 docstring
4. **命名规范**：
   - 类名：PascalCase
   - 函数名：snake_case
   - 常量：UPPER_CASE
   - 私有方法：_method_name

### 文档字符串示例
```python
def calculate_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    计算AUC (Area Under Curve) 指标

    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_scores: 预测概率，形状为 (n_samples,)

    Returns:
        AUC值，范围 [0, 1]

    Raises:
        ValueError: 当输入数组长度不匹配时

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> auc = calculate_auc(y_true, y_scores)
        >>> print(f"AUC: {auc:.3f}")
        AUC: 0.750
    """
```

## 🧪 测试指南

### 测试结构
```
tests/
├── __init__.py
├── conftest.py           # 测试配置和fixtures
├── test_evaluation.py    # 评估模块测试
├── test_features.py      # 特征模块测试
├── test_monitoring.py    # 监控模块测试
├── test_utils.py         # 工具模块测试
└── data/                 # 测试数据
```

### 编写测试
```python
import pytest
import numpy as np
from model_tools.evaluation import calculate_auc

class TestCalculateAUC:
    """AUC计算功能测试"""

    def test_perfect_classifier(self):
        """测试完美分类器"""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0, 0, 1, 1])
        assert calculate_auc(y_true, y_scores) == 1.0

    def test_random_classifier(self):
        """测试随机分类器"""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000)
        y_scores = np.random.random(1000)
        auc = calculate_auc(y_true, y_scores)
        assert 0.4 < auc < 0.6  # 随机分类器AUC约为0.5

    def test_input_validation(self):
        """测试输入验证"""
        with pytest.raises(ValueError):
            calculate_auc([1, 2], [1])  # 长度不匹配
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_evaluation.py

# 运行特定测试类
pytest tests/test_evaluation.py::TestCalculateAUC

# 运行特定测试方法
pytest tests/test_evaluation.py::TestCalculateAUC::test_perfect_classifier

# 显示测试覆盖率
pytest --cov=model_tools

# 生成HTML覆盖率报告
pytest --cov=model_tools --cov-report=html
```

## 📖 文档贡献

### 文档类型
1. **API 文档**：自动从代码生成
2. **用户指南**：README.md 和示例
3. **开发文档**：贡献指南、架构说明
4. **变更日志**：CHANGELOG.md

### 文档写作规范
- 使用简洁清晰的语言
- 提供完整的示例代码
- 包含预期输出
- 及时更新相关文档

## 🔍 代码审查

### Pull Request 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了相应的测试
- [ ] 测试全部通过
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] 没有引入破坏性变更（或有详细说明）

### 审查重点
- 代码质量和可读性
- 测试覆盖率和有效性
- 性能影响
- 向后兼容性
- 安全性考虑

## 🎯 最佳实践

### 1. 功能开发
- 先写测试，再写实现（TDD）
- 保持函数简洁，单一职责
- 考虑边界情况和异常处理
- 优化性能，但保持代码可读性

### 2. 模块设计
- 保持模块间低耦合
- 提供清晰的API接口
- 考虑扩展性和可维护性
- 遵循现有的架构模式

### 3. 性能考虑
- 避免不必要的计算
- 合理使用缓存
- 考虑内存使用
- 支持批量处理

## 🐛 问题报告

### 报告 Bug
创建 issue 时请包含：
- 问题的详细描述
- 重现步骤
- 预期行为 vs 实际行为
- 环境信息（Python版本、依赖版本等）
- 错误日志或截图

### 功能请求
- 详细描述需要的功能
- 说明使用场景和价值
- 提供可能的实现思路
- 考虑对现有功能的影响

## 📞 联系方式

- **GitHub Issues**: 报告问题和功能请求
- **Pull Requests**: 代码贡献
- **Discussions**: 技术讨论和问答
- **Email**: model-tools@example.com

## 🙏 致谢

感谢所有为项目做出贡献的开发者！每一个贡献都让 Model Tools 变得更好。

---

**Happy Coding!** 🚀