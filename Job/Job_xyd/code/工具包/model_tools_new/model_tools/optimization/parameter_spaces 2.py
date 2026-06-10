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
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
        """建议参数方法，子类需要实现"""
        raise NotImplementedError


class LightGBMSpace(ParameterSpace):
    """LightGBM参数空间定义"""

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
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

        # 树结构参数：使用固定分布范围，避免依赖其它已采样参数而产生
        # “动态分布”（会触发 Optuna 警告并削弱 TPE 建模）。
        # num_leaves < 2^max_depth 的硬约束在采样后由 get_constraints() 统一裁剪。
        max_depth = trial.suggest_int('max_depth', 3, 8)
        num_leaves = trial.suggest_int('num_leaves', 4, 255)

        # 学习率与迭代次数均使用固定范围；二者的软平衡关系交由 TPE 学习，
        # 硬性下限由 get_constraints() 中的约束函数处理。
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 1500)

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
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 120) if mode == 'anti_overfitting' else trial.suggest_int('min_child_samples', 5, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95) if mode == 'anti_overfitting' else trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),  # L1正则
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),  # L2正则
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

        # 行采样（bagging）约束：GOSS 不支持 bagging，设置 subsample 会被
        # LightGBM 告警并忽略，因此仅在非 GOSS 时建议 subsample 相关参数。
        if boosting_type != 'goss':
            subsample = trial.suggest_float('subsample', 0.6, 0.95) if mode == 'anti_overfitting' else trial.suggest_float('subsample', 0.4, 1.0)
            params['subsample'] = subsample
            if subsample < 1.0:
                params['subsample_freq'] = trial.suggest_int('subsample_freq', 1, 7)
            else:
                params['subsample_freq'] = 0

        # DART特定参数
        if boosting_type == 'dart':
            params.update({
                'drop_rate': trial.suggest_float('drop_rate', 0.01, 0.5),
                # 固定范围采样，相对 num_leaves 的上限在 get_constraints() 中裁剪
                'max_drop': trial.suggest_int('max_drop', 1, 50),
                'skip_drop': trial.suggest_float('skip_drop', 0.0, 1.0),
                'xgboost_dart_mode': trial.suggest_categorical('xgboost_dart_mode', [True, False])
            })

        # GOSS特定参数
        # 注：LightGBM>=4 中 boosting_type='goss' 已弃用（推荐 data_sample_strategy='goss'），
        # 但为兼容旧版本此处仍使用 boosting_type，旧版本可正常工作。
        elif boosting_type == 'goss':
            params.update({
                'top_rate': trial.suggest_float('top_rate', 0.1, 0.5),
                'other_rate': trial.suggest_float('other_rate', 0.05, 0.2)
            })

        # 统一应用参数约束（裁剪到合法范围），同时确保 get_constraints() 被实际使用
        for constraint_fn in LightGBMSpace.get_constraints().values():
            params = constraint_fn(params)

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
            """学习率和估计器数量的平衡：学习率很小时保证足够的迭代次数"""
            if params['learning_rate'] < 0.05 and params['n_estimators'] < 500:
                params['n_estimators'] = 500
            return params

        def max_drop_constraint(params: Dict[str, Any]) -> Dict[str, Any]:
            """DART 的 max_drop 不应超过 num_leaves 的一半"""
            if 'max_drop' in params:
                upper = max(1, params['num_leaves'] // 2)
                if params['max_drop'] > upper:
                    params['max_drop'] = upper
            return params

        return {
            'num_leaves_constraint': num_leaves_constraint,
            'learning_rate_constraint': learning_rate_estimators_constraint,
            'max_drop_constraint': max_drop_constraint
        }


class XGBoostSpace(ParameterSpace):
    """XGBoost参数空间定义"""

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
        """XGBoost参数建议"""
        booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])

        # 树深度和学习率（均使用固定分布范围，避免动态分布）
        max_depth = trial.suggest_int('max_depth', 3, 12)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)

        # n_estimators 使用固定范围；anti_overfitting 模式下收紧上限
        n_estimators_upper = 600 if mode == 'anti_overfitting' else 1500

        params = {
            'booster': booster,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': trial.suggest_int('n_estimators', 50, n_estimators_upper),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 12.0, log=True) if mode == 'anti_overfitting' else trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.1, 5.0) if mode == 'anti_overfitting' else trial.suggest_float('gamma', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95) if mode == 'anti_overfitting' else trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95) if mode == 'anti_overfitting' else trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
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
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
        """RandomForest参数建议"""
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 20)

        # 使用固定分布范围采样，避免 min_samples_leaf 的上界依赖 min_samples_split
        # 而产生动态分布；二者 leaf < split 的关系在采样后裁剪。
        if mode == 'anti_overfitting':
            min_samples_split = trial.suggest_int('min_samples_split', 10, 60)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
        else:
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        # 约束：min_samples_leaf 应小于 min_samples_split
        if min_samples_leaf >= min_samples_split:
            min_samples_leaf = max(1, min_samples_split - 1)

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

        # 如果不使用bootstrap，oob_score 不可用
        if not params['bootstrap']:
            params['oob_score'] = False

        return params


class SVMSpace(ParameterSpace):
    """SVM参数空间定义"""

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
        """SVM参数建议"""
        kernel = (
            trial.suggest_categorical('kernel', ['linear', 'rbf'])
            if mode == 'anti_overfitting'
            else trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        )

        params = {
            'C': trial.suggest_float('C', 1e-4, 10.0, log=True) if mode == 'anti_overfitting' else trial.suggest_float('C', 1e-4, 1e2, log=True),
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
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
        """LogisticRegression参数建议"""
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])
        # 关键：penalty 的可选值依赖 solver，但 Optuna 要求“同名 categorical 参数”的
        # 候选集在所有 trial 间保持一致，否则报 “does not support dynamic value space”。
        # 因此为每个 solver 使用独立的参数名，各自拥有固定候选集。
        # 同时直接用 None（sklearn>=1.2 表示无正则），避免 'none' 字符串引发的报错。
        if solver == 'lbfgs':
            penalty = trial.suggest_categorical('penalty_lbfgs', ['l2', None])
        elif solver == 'liblinear':
            # liblinear 不支持无正则（penalty=None）
            penalty = trial.suggest_categorical('penalty_liblinear', ['l1', 'l2'])
        else:  # saga
            penalty = trial.suggest_categorical('penalty_saga', ['l1', 'l2', 'elasticnet', None])

        params = {
            'penalty': penalty,
            'solver': solver,
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'random_state': 42,
        }

        # penalty=None 时 C 会被忽略，不传入以避免 sklearn 警告
        if penalty is not None:
            params['C'] = trial.suggest_float('C', 1e-4, 1e2, log=True)

        # elasticnet约束：仅 elasticnet 需要 l1_ratio
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        # liblinear不支持n_jobs
        if solver != 'liblinear':
            params['n_jobs'] = -1

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
    def suggest_parameters(
        trial: optuna.Trial,
        task_type: str = 'classification',
        mode: str = 'default'
    ) -> Dict[str, Any]:
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
    def suggest_lgb_credit_params(trial: optuna.Trial, mode: str = 'default') -> Dict[str, Any]:
        """
        信贷评分LightGBM专用参数
        更保守的参数设置，注重模型稳定性和可解释性
        """
        # 更保守的树结构（固定分布范围，num_leaves<2^max_depth 约束在采样后裁剪）
        max_depth = trial.suggest_int('max_depth', 3, 7)  # 较浅的树
        num_leaves = trial.suggest_int('num_leaves', 3, 63)

        # 约束：确保 num_leaves < 2^max_depth
        leaves_upper = 2 ** max_depth - 1
        if num_leaves > leaves_upper:
            num_leaves = leaves_upper

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