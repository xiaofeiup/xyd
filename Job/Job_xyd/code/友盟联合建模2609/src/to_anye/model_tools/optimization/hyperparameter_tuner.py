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
from typing import Dict, List, Optional, Callable, Union, Any
import warnings
import threading
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target
import joblib
from .parameter_spaces import get_parameter_space

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
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
        self._history_lock = threading.Lock()
        self.parameter_mode = 'default'
        self.task_type = 'classification'
        self.n_classes_ = None

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
        if self.custom_param_space is not None:
            return self._suggest_custom_parameters(trial)
        space_cls = get_parameter_space(self.model_type)
        return space_cls.suggest_parameters(
            trial=trial,
            task_type=self.task_type,
            mode=self.parameter_mode
        )

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
        # 统一判定任务类型（分类/回归）以及类别数，供 splitter 与 scoring 共用
        is_classification = self._is_classification_task(y)
        self.task_type = 'classification' if is_classification else 'regression'
        self.n_classes_ = int(len(np.unique(y))) if is_classification else None

        # 设置默认评分函数（与上面的任务判定保持一致来源）
        if scoring == 'auto':
            scoring = self._get_default_scoring()

        # 设置交叉验证
        if is_classification:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

        # 每次调用都重置历史，避免多次 optimize 的记录互相污染
        with self._history_lock:
            self.optimization_history_ = []

        if n_jobs != 1:
            warnings.warn(
                "n_jobs != 1 时，TPESampler 的采样顺序不确定，固定的 seed 无法保证结果可复现。"
                "如需严格复现，请使用 n_jobs=1。"
            )

        def objective(trial):
            try:
                # 获取参数
                params = self._suggest_parameters(trial)

                # 逐折交叉验证，并向 Optuna 上报中间结果以启用剪枝器
                fold_scores = []
                scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
                X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
                y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

                for step, (train_idx, valid_idx) in enumerate(cv_splitter.split(X_arr, y_arr)):
                    model = self.model_class(**params)
                    model.fit(X_arr[train_idx], y_arr[train_idx])
                    fold_score = scorer(model, X_arr[valid_idx], y_arr[valid_idx])
                    fold_scores.append(fold_score)

                    # 上报当前累计均值，并询问是否应当剪枝
                    trial.report(float(np.mean(fold_scores)), step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                scores = np.array(fold_scores)
                score = scores.mean()

                # 记录历史（加锁以支持 n_jobs > 1 的并行场景）
                with self._history_lock:
                    self.optimization_history_.append({
                        'trial': trial.number,
                        'score': score,
                        'params': params.copy(),
                        'std': scores.std()
                    })

                return score

            except optuna.TrialPruned:
                # 剪枝异常必须向上抛出，交由 Optuna 处理
                raise
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

        # 校验是否存在有效（已完成且分数有限）的 trial，避免“全部失败却静默成功”
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(t.value)
        ]
        if not completed:
            raise RuntimeError(
                "所有 trial 均失败或未产生有效分数，无法得到最佳参数。"
                "请检查模型参数空间、评分函数与数据是否匹配（可查看 warnings 输出）。"
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
        """
        获取默认评分函数

        基于已判定的任务类型与类别数选择，确保与交叉验证 splitter 来源一致。
        """
        if self.task_type == 'classification':
            # 二分类用 roc_auc，多分类用支持多类的 roc_auc_ovr
            if self.n_classes_ is not None and self.n_classes_ > 2:
                return 'roc_auc_ovr'
            return 'roc_auc'
        else:
            return 'neg_mean_squared_error'

    def _is_classification_task(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """
        判断是否为分类任务

        使用 sklearn 的 type_of_target，能正确处理浮点编码标签（如 0.0/1.0）、
        布尔标签、字符串标签等，避免基于 dtype 的误判。
        """
        target_type = type_of_target(y)
        return target_type in ('binary', 'multiclass')

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

        # study.best_trial 在没有已完成 trial 时会抛 ValueError，需用已完成 trial 判断
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(t.value)
        ]
        if completed:
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
