"""
预定义参数搜索空间

定义常用机器学习模型的参数搜索空间和约束关系
包括LightGBM、XGBoost、RandomForest等的优化配置
"""

from typing import Dict, Any, Callable, Optional
import optuna
import numpy as np


class ParameterSpace:
    """参数空间定义基类"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """建议参数方法，子类需要实现"""
        raise NotImplementedError


class LightGBMSpace(ParameterSpace):
    """LightGBM参数空间定义"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """
        LightGBM参数建议，考虑参数间约束关系

        Args:
            trial: Optuna trial对象
            task_type: 任务类型 ('classification', 'regression')

        Returns:
            参数字典
        """
        # 基础boosting参数
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

        # 树结构参数（关键约束：num_leaves < 2^max_depth）
        max_depth = trial.suggest_int('max_depth', 3, 15)
        # 计算该深度下的理论最大叶子数，但实际设置要更保守
        theoretical_max_leaves = 2 ** max_depth
        # 实际最大叶子数设为理论值的50%-90%，避免过拟合
        max_leaves_upper = max(10, int(theoretical_max_leaves * 0.8))
        num_leaves = trial.suggest_int('num_leaves', 10, min(300, max_leaves_upper))

        # 学习率和迭代次数（负相关关系）
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        # 学习率越小，需要更多迭代次数
        if learning_rate < 0.05:
            n_estimators_upper = 1500
        elif learning_rate < 0.1:
            n_estimators_upper = 1000
        else:
            n_estimators_upper = 500
        n_estimators = trial.suggest_int('n_estimators', 50, n_estimators_upper)

        # 基础参数
        params = {
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50000, 200000),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }

        # 任务特定参数
        if task_type == 'classification':
            params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
            })
        elif task_type == 'regression':
            params.update({
                'objective': 'regression',
                'metric': 'rmse'
            })

        # subsample相关约束
        if params['subsample'] < 1.0:
            params['subsample_freq'] = trial.suggest_int('subsample_freq', 1, 7)
        else:
            params['subsample_freq'] = 0

        # DART特定参数
        if boosting_type == 'dart':
            params.update({
                'drop_rate': trial.suggest_float('drop_rate', 0.01, 0.5),
                'max_drop': trial.suggest_int('max_drop', 1, min(50, num_leaves // 2)),
                'skip_drop': trial.suggest_float('skip_drop', 0.0, 1.0),
                'xgboost_dart_mode': trial.suggest_categorical('xgboost_dart_mode', [True, False])
            })

        # GOSS特定参数
        elif boosting_type == 'goss':
            params.update({
                'top_rate': trial.suggest_float('top_rate', 0.1, 0.5),
                'other_rate': trial.suggest_float('other_rate', 0.05, 0.2)
            })

        return params

    @staticmethod
    def get_constraints() -> Dict[str, Callable]:
        """获取参数约束函数"""
        def num_leaves_constraint(params: Dict[str, Any]) -> Dict[str, Any]:
            """确保num_leaves < 2^max_depth"""
            max_leaves = 2 ** params['max_depth']
            if params['num_leaves'] >= max_leaves:
                params['num_leaves'] = max_leaves - 1
            return params

        def learning_rate_estimators_constraint(params: Dict[str, Any]) -> Dict[str, Any]:
            """学习率和估计器数量的平衡"""
            if params['learning_rate'] < 0.05 and params['n_estimators'] < 500:
                params['n_estimators'] = 500
            return params

        return {
            'num_leaves_constraint': num_leaves_constraint,
            'learning_rate_constraint': learning_rate_estimators_constraint
        }


class XGBoostSpace(ParameterSpace):
    """XGBoost参数空间定义"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """XGBoost参数建议"""
        booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])

        # 树深度和学习率关系
        max_depth = trial.suggest_int('max_depth', 3, 12)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

        # 根据学习率调整迭代次数
        if learning_rate < 0.05:
            n_estimators_upper = 1500
        elif learning_rate < 0.1:
            n_estimators_upper = 1000
        else:
            n_estimators_upper = 500

        params = {
            'booster': booster,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': trial.suggest_int('n_estimators', 50, n_estimators_upper),
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

        # 任务特定参数
        if task_type == 'classification':
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        elif task_type == 'regression':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'

        # DART特定参数
        if booster == 'dart':
            params.update({
                'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
                'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
                'rate_drop': trial.suggest_float('rate_drop', 0.01, 0.5),
                'skip_drop': trial.suggest_float('skip_drop', 0.0, 1.0)
            })

        return params


class RandomForestSpace(ParameterSpace):
    """RandomForest参数空间定义"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """RandomForest参数建议"""
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 20)

        # min_samples_split和min_samples_leaf的约束关系
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        # min_samples_leaf应该小于min_samples_split
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, min(10, min_samples_split - 1))

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }

        # 如果不使用bootstrap，需要调整其他参数
        if not params['bootstrap']:
            params['oob_score'] = False
            # 不使用bootstrap时，可以考虑增加样本要求
            params['min_samples_split'] = max(params['min_samples_split'], 5)

        return params


class SVMSpace(ParameterSpace):
    """SVM参数空间定义"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """SVM参数建议"""
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'kernel': kernel,
            'random_state': 42
        }

        # 核函数特定参数
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        elif kernel in ['rbf', 'sigmoid']:
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])

        if kernel == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        return params


class LogisticRegressionSpace(ParameterSpace):
    """LogisticRegression参数空间定义"""

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        """LogisticRegression参数建议"""
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none'])

        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': penalty,
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': 42,
            'n_jobs': -1
        }

        # 求解器和惩罚项的兼容性约束
        if penalty == 'l1':
            params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        elif penalty == 'l2':
            params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])
        elif penalty == 'elasticnet':
            params['solver'] = 'saga'  # 只有saga支持elasticnet
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        else:  # none
            params['solver'] = trial.suggest_categorical('solver', ['saga', 'lbfgs'])

        return params


# 预定义参数空间映射
PARAMETER_SPACES = {
    'lightgbm': LightGBMSpace,
    'lgb': LightGBMSpace,
    'xgboost': XGBoostSpace,
    'xgb': XGBoostSpace,
    'randomforest': RandomForestSpace,
    'rf': RandomForestSpace,
    'svm': SVMSpace,
    'logistic': LogisticRegressionSpace,
    'lr': LogisticRegressionSpace
}


def get_parameter_space(model_type: str) -> ParameterSpace:
    """
    获取指定模型的参数空间

    Args:
        model_type: 模型类型

    Returns:
        参数空间对象
    """
    model_type = model_type.lower()
    if model_type not in PARAMETER_SPACES:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return PARAMETER_SPACES[model_type]


def create_custom_space(param_definitions: Dict[str, Dict[str, Any]]) -> Callable:
    """
    创建自定义参数空间

    Args:
        param_definitions: 参数定义字典
            格式: {
                'param_name': {
                    'type': 'int'/'float'/'categorical',
                    'low': min_value,  # for int/float
                    'high': max_value,  # for int/float
                    'choices': [choice1, choice2, ...],  # for categorical
                    'log': True/False,  # for int/float
                    'step': step_size   # for int
                }
            }

    Returns:
        参数建议函数
    """
    def suggest_parameters(trial: optuna.Trial, task_type: str = 'classification') -> Dict[str, Any]:
        params = {}

        for param_name, config in param_definitions.items():
            param_type = config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    config['low'],
                    config['high'],
                    step=config.get('step', 1),
                    log=config.get('log', False)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    config['low'],
                    config['high'],
                    log=config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config['choices']
                )

        return params

    return suggest_parameters


# 信贷领域特定的参数空间
class CreditScoringSpace:
    """信贷评分模型专用参数空间"""

    @staticmethod
    def suggest_lgb_credit_params(trial: optuna.Trial) -> Dict[str, Any]:
        """
        信贷评分LightGBM专用参数
        更保守的参数设置，注重模型稳定性和可解释性
        """
        # 更保守的树结构
        max_depth = trial.suggest_int('max_depth', 3, 7)  # 较浅的树
        num_leaves = trial.suggest_int('num_leaves', 3, min(63, 2**max_depth - 1))

        # 较低的学习率，更多的迭代
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),  # 更大的最小样本数
            'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 5.0),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            'is_unbalance': True,  # 信贷数据通常不平衡
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }

        return params


# 使用示例
if __name__ == "__main__":
    import optuna

    # 创建study
    study = optuna.create_study(direction='maximize')

    # 使用LightGBM参数空间
    lgb_space = get_parameter_space('lightgbm')

    def objective(trial):
        params = lgb_space.suggest_parameters(trial, task_type='classification')
        print(f"建议参数: {params}")
        return np.random.random()  # 模拟评分

    study.optimize(objective, n_trials=5)
    print(f"最佳参数: {study.best_params}")