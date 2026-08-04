"""
Model Tools 2.0 安装配置
"""

from setuptools import setup, find_packages
import os


def read_file(filename):
    """读取文件内容"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_requirements():
    """获取依赖包列表"""
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return requirements
    except FileNotFoundError:
        # 如果requirements.txt不存在，返回基础依赖
        return [
            'pandas>=1.3.0',
            'numpy>=1.20.0',
            'scikit-learn>=1.0.0',
            'scipy>=1.7.0',
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
            'openpyxl>=3.0.0',
            'statsmodels>=0.13.0',
            'psutil>=5.8.0'
        ]


# 读取长描述
long_description = read_file('README.md')

# 版本信息
__version__ = '2.0.0'

setup(
    name='model-tools',
    version=__version__,
    author='Model Tools Team',
    author_email='model-tools@example.com',
    description='专业的机器学习模型工具包，提供特征工程、模型评估、监控和报告功能',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/model-tools',

    packages=find_packages(),
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Data Scientists',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    python_requires='>=3.7',
    install_requires=get_requirements(),

    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'jupyter>=1.0',
            'nbconvert>=6.0',
        ],
        'viz': [
            'plotly>=5.0',
            'bokeh>=2.0',
        ],
        'ml': [
            'lightgbm>=3.0',
            'xgboost>=1.4',
            'catboost>=1.0',
            'optuna>=2.0',
        ],
        'monitoring': [
            'prometheus-client>=0.12',
            'requests>=2.25',
        ],
        'all': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'jupyter>=1.0',
            'nbconvert>=6.0',
            'plotly>=5.0',
            'bokeh>=2.0',
            'lightgbm>=3.0',
            'xgboost>=1.4',
            'catboost>=1.0',
            'optuna>=2.0',
            'prometheus-client>=0.12',
            'requests>=2.25',
        ]
    },

    entry_points={
        'console_scripts': [
            'model-tools=model_tools.cli:main',
        ],
    },

    project_urls={
        'Bug Reports': 'https://github.com/your-org/model-tools/issues',
        'Source': 'https://github.com/your-org/model-tools',
        'Documentation': 'https://model-tools.readthedocs.io/',
    },

    keywords='machine-learning model-evaluation feature-engineering monitoring data-science',

    zip_safe=False,
)
