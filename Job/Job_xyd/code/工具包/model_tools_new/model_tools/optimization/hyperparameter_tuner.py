"""
超参数优化器

基于Optuna的智能超参数搜索，支持：
- 参数间约束关系处理
- 多种优化目标（准确率、损失、自定义指标）
- 常见ML模型的预定义搜索空间
- 早停机制和性能监控
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Union, Any, Tuple
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer
import joblib
import logging
from datetime import datetime
import os

# 设置optuna日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    智能超参数优化器

    支持多种模型和优化策略，考虑参数间的约束关系
    """

    def __init__(
        self,
        model_class: Any,
        model_type: str = 'auto',
        study_name: Optional[str] = None,
        direction: str = 'maximize',
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: Optional[str] = None
    ):
        """
        初始化超参数优化器

        Args:
            model_class: 模型类（如LGBMClassifier）
            model_type: 模型类型 ('lgb', 'xgb', 'rf', 'lr', 'auto')
            study_name: 研究名称
            direction: 优化方向 ('maximize', 'minimize')
            sampler: Optuna采样器
            pruner: Optuna剪枝器
            storage: 存储后端
        """
        self.model_class = model_class
        self.model_type = self._detect_model_type(model_class) if model_type == 'auto' else model_type
        self.direction = direction

        # 创建study
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner()

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )

        # 存储配置
        self.custom_param_space = None
        self.custom_objective = None
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []

    def _detect_model_type(self, model_class: Any) -> str:
        """自动检测模型类型"""
        class_name = model_class.__name__.lower()

        if 'lightgbm' in class_name or 'lgbm' in class_name:
            return 'lgb'
        elif 'xgboost' in class_name or 'xgb' in class_name:
            return 'xgb'
        elif 'randomforest' in class_name or 'rf' in class_name:
            return 'rf'
        elif 'logistic' in class_name or 'lr' in class_name:
            return 'lr'
        elif 'svm' in class_name:
            return 'svm'
        else:
            return 'custom'

    def set_parameter_space(self, param_space: Dict[str, Any]):
        """
        设置自定义参数搜索空间

        Args:
            param_space: 参数空间定义字典
                格式: {
                    'param_name': {
                        'type': 'int'/'float'/'categorical',
                        'low': min_value,
                        'high': max_value,
                        'choices': [choice1, choice2, ...],  # for categorical
                        'log': True/False,  # for int/float
                        'step': step_size   # for int
                    }
                }
        """
        self.custom_param_space = param_space

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        根据模型类型建议参数

        Args:
            trial: Optuna trial对象

        Returns:
            参数字典
        """
        if self.custom_param_space:
            return self._suggest_custom_parameters(trial)

        if self.model_type == 'lgb':
            return self._suggest_lgb_parameters(trial)
        elif self.model_type == 'xgb':
            return self._suggest_xgb_parameters(trial)
        elif self.model_type == 'rf':
            return self._suggest_rf_parameters(trial)
        elif self.model_type == 'lr':
            return self._suggest_lr_parameters(trial)
        elif self.model_type == 'svm':
            return self._suggest_svm_parameters(trial)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _suggest_custom_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议自定义参数"""
        params = {}

        for param_name, param_config in self.custom_param_space.items():
            param_type = param_config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1),
                    log=param_config.get('log', False)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        return params

    def _suggest_lgb_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        LightGBM参数建议（考虑参数间约束）
        """
        # 基础参数
        params = {
            'objective': trial.suggest_categorical('objective', ['binary', 'regression']),
            'metric': trial.suggest_categorical('metric', ['binary_logloss', 'auc', 'rmse', 'mae']),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 200000),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }

        # 处理num_leaves和max_depth的约束关系
        # num_leaves应该 < 2^max_depth
        max_leaves_for_depth = 2 ** params['max_depth']
        if params['num_leaves'] >= max_leaves_for_depth:
            params['num_leaves'] = max_leaves_for_depth - 1

        # 如果选择了DART，添加特定参数
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.5)
            params['max_drop'] = trial.suggest_int('max_drop', 1, 50)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)

        return params

    def _suggest_xgb_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """XGBoost参数建议"""
        params = {
            'objective': trial.suggest_categorical('objective', ['binary:logistic', 'reg:squarederror']),
            'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'auc', 'rmse', 'mae']),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }

        # DART特定参数
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 0.01, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)

        return params

    def _suggest_rf_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """RandomForest参数建议"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }

    def _suggest_lr_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """LogisticRegression参数建议"""
        return {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': 42,
            'n_jobs': -1
        }

    def _suggest_svm_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """SVM参数建议"""
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'kernel': kernel,
            'random_state': 42
        }

        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)

        if kernel in ['poly', 'rbf', 'sigmoid']:
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])

        return params

    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_trials: int = 100,
        cv: int = 5,
        scoring: Union[str, Callable] = 'auto',
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        执行超参数优化

        Args:
            X: 特征数据
            y: 目标变量
            n_trials: 优化试验次数
            cv: 交叉验证折数
            scoring: 评分函数
            timeout: 超时时间（秒）
            n_jobs: 并行任务数
            show_progress_bar: 是否显示进度条
            callbacks: 回调函数列表

        Returns:
            优化结果字典
        """
        # 设置默认评分函数
        if scoring == 'auto':
            scoring = self._get_default_scoring()

        # 设置交叉验证
        if self._is_classification_task(y):
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

        def objective(trial):
            try:
                # 获取参数
                params = self._suggest_parameters(trial)

                # 创建模型
                model = self.model_class(**params)

                # 交叉验证评估
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=1)
                score = scores.mean()

                # 记录历史
                self.optimization_history_.append({
                    'trial': trial.number,
                    'score': score,
                    'params': params.copy(),
                    'std': scores.std()
                })

                return score

            except Exception as e:
                warnings.warn(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf') if self.direction == 'maximize' else float('inf')

        # 执行优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks
        )

        # 保存最佳结果
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history_
        }

    def _get_default_scoring(self) -> str:
        """获取默认评分函数"""
        model_name = self.model_class.__name__.lower()

        if 'classifier' in model_name:
            return 'roc_auc'
        elif 'regressor' in model_name:
            return 'neg_mean_squared_error'
        else:
            return 'accuracy'  # 默认

    def _is_classification_task(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """判断是否为分类任务"""
        unique_values = np.unique(y)
        return len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values)

    def get_best_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """使用最佳参数训练模型"""
        if self.best_params_ is None:
            raise ValueError("请先运行optimize()方法")

        model = self.model_class(**self.best_params_)
        model.fit(X, y)
        return model

    def plot_optimization_history(self):
        """绘制优化历史"""
        try:
            optuna.visualization.plot_optimization_history(self.study)
        except ImportError:
            print("需要安装plotly来显示图表: pip install plotly")

    def plot_param_importances(self):
        """绘制参数重要性"""
        try:
            optuna.visualization.plot_param_importances(self.study)
        except ImportError:
            print("需要安装plotly来显示图表: pip install plotly")

    def plot_parallel_coordinate(self):
        """绘制平行坐标图"""
        try:
            optuna.visualization.plot_parallel_coordinate(self.study)
        except ImportError:
            print("需要安装plotly来显示图表: pip install plotly")

    def save_study(self, filepath: str):
        """保存study对象"""
        with open(filepath, 'wb') as f:
            joblib.dump(self.study, f)

    def load_study(self, filepath: str):
        """加载study对象"""
        with open(filepath, 'rb') as f:
            self.study = joblib.load(f)

        if self.study.best_trial:
            self.best_params_ = self.study.best_params
            self.best_score_ = self.study.best_value

    def get_trials_dataframe(self) -> pd.DataFrame:
        """获取试验结果DataFrame"""
        return self.study.trials_dataframe()

    def continue_optimization(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        additional_trials: int = 50,
        **kwargs
    ):
        """继续优化（在已有基础上）"""
        return self.optimize(X, y, n_trials=additional_trials, **kwargs)


# 辅助函数
def create_lgb_tuner(
    model_class,
    study_name: Optional[str] = None,
    direction: str = 'maximize'
) -> HyperparameterTuner:
    """创建LightGBM调优器"""
    return HyperparameterTuner(
        model_class=model_class,
        model_type='lgb',
        study_name=study_name,
        direction=direction
    )


def create_xgb_tuner(
    model_class,
    study_name: Optional[str] = None,
    direction: str = 'maximize'
) -> HyperparameterTuner:
    """创建XGBoost调优器"""
    return HyperparameterTuner(
        model_class=model_class,
        model_type='xgb',
        study_name=study_name,
        direction=direction
    )


# 使用示例
if __name__ == "__main__":
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification

    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # 创建调优器
    tuner = create_lgb_tuner(LGBMClassifier, study_name="lgb_optimization")

    # 执行优化
    result = tuner.optimize(X, y, n_trials=50, cv=3)

    print(f"最佳参数: {result['best_params']}")
    print(f"最佳分数: {result['best_score']:.4f}")

    # 获取最佳模型
    best_model = tuner.get_best_model(X, y)
    print(f"最佳模型: {best_model}")