"""
防过拟合超参数调优器

专门设计用于解决AUC/KS目标函数导致的过拟合问题
确保训练集和验证集性能差异控制在合理范围内:
- AUC差异 ≤ 0.05
- KS差异 ≤ 0.03
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
import warnings

from .hyperparameter_tuner import HyperparameterTuner
from .objectives import KSObjective, AUCObjective, ObjectiveFunction


class AntiOverfittingTuner(HyperparameterTuner):
    """
    防过拟合超参数调优器

    核心功能:
    1. 验证集性能监控
    2. 过拟合惩罚机制
    3. 正则化导向的参数空间
    4. 早停机制
    """

    def __init__(
        self,
        model_class,
        model_type: str,
        direction: str = 'maximize',
        study_name: Optional[str] = None,
        # 防过拟合参数
        max_auc_gap: float = 0.05,
        max_ks_gap: float = 0.03,
        overfitting_penalty_weight: float = 2.0,
        use_regularization_focused_space: bool = True,
        early_stopping_patience: int = 10
    ):
        super().__init__(
            model_class=model_class,
            model_type=model_type,
            study_name=study_name,
            direction=direction
        )

        # 防过拟合参数
        self.max_auc_gap = max_auc_gap
        self.max_ks_gap = max_ks_gap
        self.overfitting_penalty_weight = overfitting_penalty_weight
        self.early_stopping_patience = early_stopping_patience
        self.parameter_mode = 'anti_overfitting' if use_regularization_focused_space else 'default'

        # 性能追踪
        self.performance_history_ = []
        self.validation_split_seed = 42

        # 目标函数缓存
        self.ks_objective = KSObjective()
        self.auc_objective = AUCObjective()

    def _suggest_parameters(self, trial):
        """统一走 parameter_spaces.py 的单一参数空间入口"""
        return super()._suggest_parameters(trial)

    def _calculate_performance_gap(self, train_score: float, val_score: float, metric_type: str) -> float:
        """计算性能差距"""
        gap = abs(train_score - val_score)

        if metric_type == 'auc':
            max_allowed_gap = self.max_auc_gap
        elif metric_type == 'ks':
            max_allowed_gap = self.max_ks_gap
        else:
            max_allowed_gap = 0.05  # 默认值

        return gap, max_allowed_gap

    def _calculate_overfitting_penalty(self, train_auc: float, val_auc: float,
                                     train_ks: float, val_ks: float) -> float:
        """计算过拟合惩罚"""
        auc_gap, max_auc_gap = self._calculate_performance_gap(train_auc, val_auc, 'auc')
        ks_gap, max_ks_gap = self._calculate_performance_gap(train_ks, val_ks, 'ks')

        # 计算惩罚
        auc_penalty = max(0, (auc_gap - max_auc_gap)) * self.overfitting_penalty_weight
        ks_penalty = max(0, (ks_gap - max_ks_gap)) * self.overfitting_penalty_weight

        total_penalty = auc_penalty + ks_penalty

        return {
            'total_penalty': total_penalty,
            'auc_gap': auc_gap,
            'ks_gap': ks_gap,
            'auc_penalty': auc_penalty,
            'ks_penalty': ks_penalty,
            'is_overfitting': auc_gap > max_auc_gap or ks_gap > max_ks_gap
        }

    def optimize_anti_overfitting(
        self,
        X,
        y,
        objective_function: Optional[ObjectiveFunction] = None,
        n_trials: int = 100,
        cv_folds: int = 5,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        weight_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行防过拟合优化（使用纯交叉验证）

        Args:
            X: 特征数据
            y: 目标变量
            objective_function: 目标函数，如None则使用AUC
            n_trials: 试验次数
            cv_folds: 交叉验证折数
            n_jobs: 并行数
            show_progress_bar: 是否显示进度条
            weight_col: 权重列名，如果传入则使用该列作为样本权重

        Returns:
            优化结果字典
        """

        if objective_function is None:
            objective_function = self.auc_objective

        # 提取权重列
        if weight_col is not None:
            if isinstance(X, pd.DataFrame) and weight_col in X.columns:
                sample_weight = X[weight_col].values
                X = X.drop(columns=[weight_col])
            else:
                raise ValueError(f"weight_col '{weight_col}' not found in X")
        else:
            sample_weight = None

        # 设置交叉验证（不固定random_state以获得真实的交叉验证效果）
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True)

        def objective(trial):
            try:
                # 获取参数
                params = self._suggest_parameters(trial)

                # 存储每折的性能指标
                train_scores = []
                val_scores = []
                train_aucs = []
                val_aucs = []
                train_kss = []
                val_kss = []

                # 交叉验证
                for train_idx, val_idx in cv_splitter.split(X, y):
                    # 获取训练和验证数据
                    if isinstance(X, pd.DataFrame):
                        X_train_fold = X.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                    else:
                        X_train_fold = X[train_idx]
                        X_val_fold = X[val_idx]

                    if isinstance(y, pd.Series):
                        y_train_fold = y.iloc[train_idx]
                        y_val_fold = y.iloc[val_idx]
                    else:
                        y_train_fold = y[train_idx]
                        y_val_fold = y[val_idx]

                    # 获取权重
                    if sample_weight is not None:
                        weight_train_fold = sample_weight[train_idx]
                    else:
                        weight_train_fold = None

                    # 训练模型
                    model = self.model_class(**params)
                    model.fit(X_train_fold, y_train_fold, sample_weight=weight_train_fold)

                    # 预测
                    y_pred_train_fold = model.predict_proba(X_train_fold)[:, 1]
                    y_pred_val_fold = model.predict_proba(X_val_fold)[:, 1]

                    # 计算性能指标
                    train_score = objective_function(y_train_fold, y_pred_train_fold)
                    val_score = objective_function(y_val_fold, y_pred_val_fold)

                    train_auc = roc_auc_score(y_train_fold, y_pred_train_fold)
                    val_auc = roc_auc_score(y_val_fold, y_pred_val_fold)

                    train_ks = self.ks_objective(y_train_fold, y_pred_train_fold)
                    val_ks = self.ks_objective(y_val_fold, y_pred_val_fold)

                    # 收集结果
                    train_scores.append(train_score)
                    val_scores.append(val_score)
                    train_aucs.append(train_auc)
                    val_aucs.append(val_auc)
                    train_kss.append(train_ks)
                    val_kss.append(val_ks)

                # 计算平均性能
                avg_train_score = np.mean(train_scores)
                avg_val_score = np.mean(val_scores)
                avg_train_auc = np.mean(train_aucs)
                avg_val_auc = np.mean(val_aucs)
                avg_train_ks = np.mean(train_kss)
                avg_val_ks = np.mean(val_kss)

                # 计算过拟合惩罚
                penalty_info = self._calculate_overfitting_penalty(
                    avg_train_auc, avg_val_auc, avg_train_ks, avg_val_ks
                )

                # 最终得分（验证集得分 - 过拟合惩罚）
                final_score = avg_val_score - penalty_info['total_penalty']

                # 记录性能历史
                performance_record = {
                    'trial': trial.number,
                    'params': params.copy(),
                    'train_score': avg_train_score,
                    'val_score': avg_val_score,
                    'final_score': final_score,
                    'train_auc': avg_train_auc,
                    'val_auc': avg_val_auc,
                    'train_ks': avg_train_ks,
                    'val_ks': avg_val_ks,
                    'auc_gap': penalty_info['auc_gap'],
                    'ks_gap': penalty_info['ks_gap'],
                    'is_overfitting': penalty_info['is_overfitting'],
                    'total_penalty': penalty_info['total_penalty'],
                    # 添加交叉验证的方差信息
                    'train_score_std': np.std(train_scores),
                    'val_score_std': np.std(val_scores),
                    'train_auc_std': np.std(train_aucs),
                    'val_auc_std': np.std(val_aucs)
                }

                self.performance_history_.append(performance_record)

                return final_score

            except Exception as e:
                warnings.warn(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf') if self.direction == 'maximize' else float('inf')

        # 执行优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )

        # 保存最佳结果
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        # 分析结果
        results_df = pd.DataFrame(self.performance_history_)

        # 找到满足防过拟合条件的最佳模型
        if len(results_df) > 0 and 'auc_gap' in results_df.columns:
            valid_trials = results_df[
                (results_df['auc_gap'] <= self.max_auc_gap) &
                (results_df['ks_gap'] <= self.max_ks_gap)
            ]
        else:
            # 如果没有数据或列名问题，返回空DataFrame
            valid_trials = pd.DataFrame()

        if len(valid_trials) > 0:
            best_valid_trial = valid_trials.loc[valid_trials['val_score'].idxmax()]
            recommended_params = best_valid_trial['params']
            recommended_score = best_valid_trial['val_score']
            best_valid_auc_gap = best_valid_trial['auc_gap']
            best_valid_ks_gap = best_valid_trial['ks_gap']
        else:
            # 如果没有完全满足条件的，选择惩罚最小的
            if len(results_df) > 0 and 'total_penalty' in results_df.columns:
                min_penalty_trial = results_df.loc[results_df['total_penalty'].idxmin()]
                recommended_params = min_penalty_trial['params']
                recommended_score = min_penalty_trial['val_score']
                best_valid_auc_gap = min_penalty_trial.get('auc_gap', 0)
                best_valid_ks_gap = min_penalty_trial.get('ks_gap', 0)
            else:
                # 备用方案：使用研究的最佳参数
                recommended_params = self.best_params_
                recommended_score = self.best_score_
                best_valid_auc_gap = None
                best_valid_ks_gap = None

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'recommended_params': recommended_params,
            'recommended_score': recommended_score,
            'n_trials': len(self.study.trials),
            'performance_history': self.performance_history_,
            'results_summary': {
                'total_trials': len(results_df) if len(results_df) > 0 else 0,
                'valid_trials': len(valid_trials) if len(valid_trials) > 0 else 0,
                'avg_auc_gap': results_df['auc_gap'].mean() if len(results_df) > 0 and 'auc_gap' in results_df.columns else 0,
                'avg_ks_gap': results_df['ks_gap'].mean() if len(results_df) > 0 and 'ks_gap' in results_df.columns else 0,
                'overfitting_rate': (results_df['is_overfitting'].sum() / len(results_df)) if len(results_df) > 0 and 'is_overfitting' in results_df.columns else 0,
                'best_valid_auc_gap': best_valid_auc_gap,
                'best_valid_ks_gap': best_valid_ks_gap
            }
        }

    def get_performance_analysis(self) -> pd.DataFrame:
        """获取性能分析报告"""
        if not hasattr(self, 'performance_history_') or len(self.performance_history_) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(self.performance_history_)

        # 添加分析列
        df['score_gap'] = df['train_score'] - df['val_score']
        df['meets_auc_criteria'] = df['auc_gap'] <= self.max_auc_gap
        df['meets_ks_criteria'] = df['ks_gap'] <= self.max_ks_gap
        df['meets_all_criteria'] = df['meets_auc_criteria'] & df['meets_ks_criteria']

        return df.sort_values('final_score', ascending=False)

    def plot_overfitting_analysis(self, save_path: Optional[str] = None):

        """绘制过拟合分析图"""
        if not hasattr(self, 'performance_history_') or len(self.performance_history_) == 0:
            print("没有性能历史数据可绘制")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            df = pd.DataFrame(self.performance_history_)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('过拟合分析报告', fontsize=16)

            # 1. AUC差距分布
            axes[0, 0].hist(df['auc_gap'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].axvline(self.max_auc_gap, color='red', linestyle='--', label=f'阈值: {self.max_auc_gap}')
            axes[0, 0].set_xlabel('AUC差距')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].set_title('AUC差距分布')
            axes[0, 0].legend()

            # 2. KS差距分布
            axes[0, 1].hist(df['ks_gap'], bins=20, alpha=0.7, color='green')
            axes[0, 1].axvline(self.max_ks_gap, color='red', linestyle='--', label=f'阈值: {self.max_ks_gap}')
            axes[0, 1].set_xlabel('KS差距')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].set_title('KS差距分布')
            axes[0, 1].legend()

            # 3. 训练vs验证性能散点图
            valid_mask = df['meets_all_criteria']
            axes[1, 0].scatter(df[valid_mask]['train_score'], df[valid_mask]['val_score'],
                             alpha=0.6, color='green', label='满足条件')
            axes[1, 0].scatter(df[~valid_mask]['train_score'], df[~valid_mask]['val_score'],
                             alpha=0.6, color='red', label='过拟合')
            axes[1, 0].plot([df['train_score'].min(), df['train_score'].max()],
                           [df['train_score'].min(), df['train_score'].max()], 'k--', alpha=0.5)
            axes[1, 0].set_xlabel('训练集性能')
            axes[1, 0].set_ylabel('验证集性能')
            axes[1, 0].set_title('训练vs验证性能')
            axes[1, 0].legend()

            # 4. 试验进程中的性能变化
            axes[1, 1].plot(df['trial'], df['train_score'], label='训练集', alpha=0.7)
            axes[1, 1].plot(df['trial'], df['val_score'], label='验证集', alpha=0.7)
            axes[1, 1].fill_between(df['trial'],
                                  df['val_score'] - self.max_auc_gap,
                                  df['val_score'] + self.max_auc_gap,
                                  alpha=0.2, color='gray', label='可接受范围')
            axes[1, 1].set_xlabel('试验编号')
            axes[1, 1].set_ylabel('性能得分')
            axes[1, 1].set_title('优化进程')
            axes[1, 1].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表保存至: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("需要安装matplotlib和seaborn来绘制图表")



if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    tuner = AntiOverfittingTuner(
        model_class=LogisticRegression,
        model_type='lr',
        direction='maximize',
        study_name='anti_overfitting_tuner'
    )
    # 推荐写法：用Optuna的Trial对象来生成参数，这里用Mock对象模拟
    import optuna

    # 创建一个临时的trial对象用于参数建议
    study = optuna.create_study(direction='maximize')
    trial = study.ask()

    params = tuner._suggest_parameters(trial)
    model = LogisticRegression(**params)
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    print(y_pred)
    tuner.optimize_anti_overfitting(X, y)