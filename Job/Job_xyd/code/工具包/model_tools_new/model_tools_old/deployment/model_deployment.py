"""
模型部署工具模块

提供模型部署相关的全链路工具，包括：
1. joblib(pkl) 模型 -> PMML 格式转换（支持 sklearn / LightGBM / XGBoost）
2. PMML / pkl 模型加载与预测封装
3. pkl 与 PMML 预测结果一致性验证（线下线上一致性校验）

依赖说明
--------
- 必选: ``joblib``、``numpy``、``pandas``、``scikit-learn``
- 转换 PMML: ``sklearn2pmml`` (需要本机安装 Java >=8 / >=11)
- 加载 PMML 进行预测: ``pypmml``
- 可选: ``lightgbm``、``xgboost``

典型用法
--------
>>> from model_tools.Deployment.model_deployment import ModelDeployer
>>> deployer = ModelDeployer()
>>> # 1. pkl -> pmml
>>> deployer.convert_pkl_to_pmml(
...     pkl_path='model.pkl',
...     pmml_path='model.pmml',
...     feature_names=feature_names,
...     target_name='label'
... )
>>> # 2. 一致性验证
>>> result = deployer.verify_consistency(
...     pkl_path='model.pkl',
...     pmml_path='model.pmml',
...     X=X_test,
...     atol=1e-6
... )
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:  # pragma: no cover - 必选依赖
    raise ImportError("model_deployment 模块需要 joblib，请先安装: pip install joblib") from exc


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConsistencyResult:
    """pkl / PMML 一致性比对结果"""

    is_consistent: bool
    n_samples: int
    max_abs_diff: float
    mean_abs_diff: float
    median_abs_diff: float
    p99_abs_diff: float
    n_mismatch: int
    mismatch_ratio: float
    atol: float
    rtol: float
    pkl_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    pmml_predictions: Optional[np.ndarray] = field(default=None, repr=False)
    diff_detail: Optional[pd.DataFrame] = field(default=None, repr=False)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """转为可序列化的 dict（默认不包含原始预测数组）"""
        d = asdict(self)
        if not include_arrays:
            d.pop('pkl_predictions', None)
            d.pop('pmml_predictions', None)
            d.pop('diff_detail', None)
            return d

        if self.pkl_predictions is not None:
            d['pkl_predictions'] = np.asarray(self.pkl_predictions).tolist()
        if self.pmml_predictions is not None:
            d['pmml_predictions'] = np.asarray(self.pmml_predictions).tolist()
        if self.diff_detail is not None:
            d['diff_detail'] = pd.DataFrame(self.diff_detail).to_dict(orient='records')
        return d

    def summary(self) -> str:
        """生成可读的摘要文本"""
        status = '✅ 一致' if self.is_consistent else '❌ 不一致'
        lines = [
            '=' * 60,
            f'PKL vs PMML 一致性校验报告  [{status}]',
            '=' * 60,
            f'样本数            : {self.n_samples}',
            f'最大绝对误差       : {self.max_abs_diff:.3e}',
            f'平均绝对误差       : {self.mean_abs_diff:.3e}',
            f'中位绝对误差       : {self.median_abs_diff:.3e}',
            f'99 分位绝对误差    : {self.p99_abs_diff:.3e}',
            f'不一致样本数       : {self.n_mismatch} / {self.n_samples} '
            f'({self.mismatch_ratio:.4%})',
            f'容忍度 (atol/rtol) : {self.atol:.1e} / {self.rtol:.1e}',
        ]
        if self.extra:
            lines.append('-' * 60)
            for k, v in self.extra.items():
                lines.append(f'{k:<18}: {v}')
        lines.append('=' * 60)
        return '\n'.join(lines)


# ============================================================
# 核心部署类
# ============================================================

class ModelDeployer:
    """
    模型部署工具

    职责：
        - pkl(joblib) -> pmml 转换
        - 模型加载与预测
        - 线下线上(pkl vs pmml) 一致性验证
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # -------------------- 工具函数 --------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f'[ModelDeployer] {msg}')

    @staticmethod
    def _ensure_dir(path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _is_classifier(model: Any) -> bool:
        """粗略判断是否为分类模型"""
        if hasattr(model, 'predict_proba'):
            return True
        # sklearn / lightgbm / xgboost 都暴露 _estimator_type
        return getattr(model, '_estimator_type', None) == 'classifier'

    @staticmethod
    def _to_dataframe(
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """统一输入为 DataFrame，便于 PMML 评分"""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X), columns=list(feature_names))

    # -------------------- 加载 --------------------

    def load_pkl(self, pkl_path: str) -> Any:
        """加载 joblib/pickle 保存的模型"""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f'pkl 文件不存在: {pkl_path}')
        self._log(f'加载 pkl 模型: {pkl_path}')
        return joblib.load(pkl_path)

    def load_pmml(self, pmml_path: str):
        """使用 pypmml 加载 PMML 模型"""
        try:
            from pypmml import Model as PmmlModel
        except ImportError as exc:
            raise ImportError(
                '加载 PMML 需要安装 pypmml: pip install pypmml'
            ) from exc

        if not os.path.exists(pmml_path):
            raise FileNotFoundError(f'PMML 文件不存在: {pmml_path}')
        self._log(f'加载 PMML 模型: {pmml_path}')
        return PmmlModel.fromFile(pmml_path)

    # -------------------- 转换: pkl -> pmml --------------------

    def convert_pkl_to_pmml(
        self,
        pkl_path: str,
        pmml_path: str,
        feature_names: Optional[Sequence[str]] = None,
        target_name: str = 'target',
        with_repr: bool = True,
        debug: bool = False,
    ) -> str:
        """
        将 joblib 保存的 sklearn / lightgbm / xgboost 模型转换为 PMML

        Parameters
        ----------
        pkl_path : str
            原始 pkl 模型路径
        pmml_path : str
            输出的 PMML 文件路径
        feature_names : list of str, optional
            特征名（PMML 必须显式指定列名）。
            若 pkl 中是 sklearn Pipeline 且包含列变换，可省略。
        target_name : str, default='target'
            目标列名
        with_repr : bool, default=True
            是否在 PMML 中保留 Python 模型 repr，便于线上排查
        debug : bool, default=False
            是否输出 sklearn2pmml 的调试信息

        Returns
        -------
        pmml_path : str
            生成的 PMML 文件绝对路径

        Notes
        -----
        - 需要本机安装 Java（建议 JDK 11+），否则 sklearn2pmml 无法运行
        - 推荐在训练时直接使用 PMMLPipeline，可避免类型/编码不一致
        """
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn.pipeline import Pipeline
        except ImportError as exc:
            raise ImportError(
                'PMML 转换需要安装 sklearn2pmml: pip install sklearn2pmml\n'
                '同时需要本机已安装 Java (JDK 8+)'
            ) from exc

        model = self.load_pkl(pkl_path)
        self._ensure_dir(pmml_path)

        # 已经是 PMMLPipeline，直接导出
        if isinstance(model, PMMLPipeline):
            pmml_pipeline = model
        elif isinstance(model, Pipeline):
            # 普通 Pipeline -> PMMLPipeline
            pmml_pipeline = PMMLPipeline(steps=model.steps)
        else:
            # 单一模型 -> 包成 PMMLPipeline
            pmml_pipeline = PMMLPipeline([('estimator', model)])

        # 注入特征 / 目标元数据，PMML 要求
        if feature_names is not None:
            pmml_pipeline.active_fields = np.asarray(list(feature_names))
        if target_name is not None:
            pmml_pipeline.target_fields = np.asarray([target_name])

        self._log(f'开始转换 PMML -> {pmml_path}')
        sklearn2pmml(pmml_pipeline, pmml_path, with_repr=with_repr, debug=debug)
        self._log('PMML 转换完成')
        return os.path.abspath(pmml_path)

    # -------------------- 预测 --------------------

    def predict_with_pkl(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        mode: str = 'auto',
        positive_class_index: int = 1,
    ) -> np.ndarray:
        """
        使用 pkl 模型预测

        Parameters
        ----------
        model : 已加载的模型对象
        X : 特征数据
        mode : {'auto', 'proba', 'predict'}
            - 'auto': 分类器走 predict_proba，回归走 predict
            - 'proba': 强制 predict_proba (取 positive_class_index 列)
            - 'predict': 强制 predict
        positive_class_index : int
            分类器取概率时使用的列索引

        Returns
        -------
        np.ndarray, shape=(n_samples,)
        """
        if mode == 'auto':
            mode = 'proba' if self._is_classifier(model) else 'predict'

        if mode == 'proba':
            if not hasattr(model, 'predict_proba'):
                raise AttributeError('模型不支持 predict_proba')
            proba = model.predict_proba(X)
            return np.asarray(proba)[:, positive_class_index]
        return np.asarray(model.predict(X)).ravel()

    def predict_with_pmml(
        self,
        pmml_model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[Sequence[str]] = None,
        score_field: Optional[str] = None,
        positive_label: Any = 1,
    ) -> np.ndarray:
        """
        使用 PMML 模型预测

        Parameters
        ----------
        pmml_model : pypmml.Model
        X : 特征数据
        feature_names : list of str, optional
            当 X 为 ndarray 时必须提供
        score_field : str, optional
            指定 PMML 输出字段；分类任务默认取
            ``probability(<positive_label>)``，回归任务默认取 ``predicted_<target>``
        positive_label : Any, default=1
            分类任务中正样本的 label，用于拼接 ``probability(<label>)`` 字段名

        Returns
        -------
        np.ndarray
        """
        df = self._to_dataframe(X, feature_names)

        # pypmml 要求列名与 inputFields 一致
        df_input = df.copy()

        # 批量评分
        result = pmml_model.predict(df_input)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)

        # 选择输出字段
        if score_field is not None:
            if score_field not in result.columns:
                raise KeyError(
                    f'PMML 输出中不存在字段: {score_field}, '
                    f'可用字段: {list(result.columns)}'
                )
            return result[score_field].to_numpy()

        # 自动选择
        candidates_proba = [
            f'probability({positive_label})',
            f'probability_{positive_label}',
            'probability(1)',
            'probability_1',
        ]
        for col in candidates_proba:
            if col in result.columns:
                return result[col].to_numpy()

        # 回归 / 默认输出
        for col in result.columns:
            if col.startswith('predicted_') or col.lower() == 'predicted':
                return result[col].to_numpy()

        # 兜底：如有概率列优先取最大概率列
        prob_cols = [c for c in result.columns if c.startswith('probability')]
        if prob_cols:
            warnings.warn(
                f'未匹配到 positive_label={positive_label} 的概率列, '
                f'使用首个概率列: {prob_cols[0]}'
            )
            return result[prob_cols[0]].to_numpy()

        # 最后兜底：第一个数值列
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            warnings.warn(f'使用 PMML 输出首个数值列: {numeric_cols[0]}')
            return result[numeric_cols[0]].to_numpy()

        raise ValueError(
            f'无法从 PMML 输出中识别预测列, 输出列: {list(result.columns)}'
        )

    # -------------------- 一致性验证 --------------------

    def verify_consistency(
        self,
        pkl_path: str,
        pmml_path: str,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[Sequence[str]] = None,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        mode: str = 'auto',
        positive_class_index: int = 1,
        positive_label: Any = 1,
        score_field: Optional[str] = None,
        return_arrays: bool = False,
        save_report: Optional[str] = None,
        top_k_diff: int = 20,
    ) -> ConsistencyResult:
        """
        对比 pkl 与 PMML 模型在同一份数据上的预测结果

        Parameters
        ----------
        pkl_path : str
            pkl 模型路径
        pmml_path : str
            pmml 模型路径
        X : DataFrame / ndarray
            验证数据（建议使用真实分布的代表性样本，量级 1k~10w）
        feature_names : list of str, optional
            X 为 ndarray 时必须提供，需与 PMML inputFields 顺序一致
        atol, rtol : float
            判定一致使用的绝对 / 相对容忍度，含义同 ``np.isclose``
        mode : {'auto', 'proba', 'predict'}
            预测模式（见 ``predict_with_pkl``）
        positive_class_index : int
            pkl 端取正类概率的列索引
        positive_label : Any
            pmml 端正类标签（用于定位 ``probability(<label>)`` 字段）
        score_field : str, optional
            手动指定 PMML 输出字段
        return_arrays : bool, default=False
            是否在结果对象中保留两份预测数组及差异明细
        save_report : str, optional
            json 报告输出路径
        top_k_diff : int, default=20
            差异 Top-K 明细行数

        Returns
        -------
        ConsistencyResult
        """
        # 1. 加载模型
        pkl_model = self.load_pkl(pkl_path)
        pmml_model = self.load_pmml(pmml_path)

        # 2. 准备数据
        df = self._to_dataframe(X, feature_names)
        if feature_names is None:
            feature_names = list(df.columns)

        self._log(f'开始一致性校验，样本数={len(df)}')

        # 3. 双端预测
        pkl_pred = self.predict_with_pkl(
            pkl_model, df,
            mode=mode,
            positive_class_index=positive_class_index,
        )
        pmml_pred = self.predict_with_pmml(
            pmml_model, df,
            feature_names=feature_names,
            score_field=score_field,
            positive_label=positive_label,
        )

        # 4. 长度 / NaN 校验
        if len(pkl_pred) != len(pmml_pred):
            raise ValueError(
                f'pkl 与 pmml 预测长度不一致: {len(pkl_pred)} vs {len(pmml_pred)}'
            )

        pkl_pred = np.asarray(pkl_pred, dtype=float).ravel()
        pmml_pred = np.asarray(pmml_pred, dtype=float).ravel()

        nan_mask = np.isnan(pkl_pred) | np.isnan(pmml_pred)
        n_nan = int(nan_mask.sum())
        if n_nan:
            warnings.warn(f'存在 {n_nan} 个 NaN 预测值，将从一致性比较中剔除')

        valid = ~nan_mask
        diff = np.abs(pkl_pred[valid] - pmml_pred[valid])

        if diff.size == 0:
            raise ValueError('有效样本为 0，无法进行一致性比较')

        # 5. 差异统计
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        median_abs = float(np.median(diff))
        p99_abs = float(np.percentile(diff, 99))

        mismatch_mask = ~np.isclose(
            pkl_pred[valid], pmml_pred[valid], atol=atol, rtol=rtol
        )
        n_mismatch = int(mismatch_mask.sum())
        n_valid = int(valid.sum())
        mismatch_ratio = n_mismatch / n_valid if n_valid else 0.0
        is_consistent = (n_mismatch == 0) and (n_nan == 0)

        # 6. Top-K 差异明细
        diff_detail: Optional[pd.DataFrame] = None
        if return_arrays or save_report:
            full_diff = np.full_like(pkl_pred, fill_value=np.nan, dtype=float)
            full_diff[valid] = diff
            order = np.argsort(-np.nan_to_num(full_diff, nan=-1))
            top_idx = order[:max(1, top_k_diff)]
            diff_detail = pd.DataFrame({
                'index': top_idx,
                'pkl_pred': pkl_pred[top_idx],
                'pmml_pred': pmml_pred[top_idx],
                'abs_diff': full_diff[top_idx],
            })

        result = ConsistencyResult(
            is_consistent=is_consistent,
            n_samples=int(len(pkl_pred)),
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            median_abs_diff=median_abs,
            p99_abs_diff=p99_abs,
            n_mismatch=n_mismatch,
            mismatch_ratio=mismatch_ratio,
            atol=atol,
            rtol=rtol,
            pkl_predictions=pkl_pred if return_arrays else None,
            pmml_predictions=pmml_pred if return_arrays else None,
            diff_detail=diff_detail if return_arrays else None,
            extra={
                'pkl_path': os.path.abspath(pkl_path),
                'pmml_path': os.path.abspath(pmml_path),
                'mode': mode,
                'n_nan': n_nan,
                'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
        )

        self._log(result.summary())

        if save_report:
            self._save_report(result, save_report, diff_detail=diff_detail)
        return result

    # -------------------- 报告 --------------------

    def _save_report(
        self,
        result: ConsistencyResult,
        path: str,
        diff_detail: Optional[pd.DataFrame] = None,
    ) -> None:
        """保存 JSON 报告（含 Top-K 差异明细）"""
        self._ensure_dir(path)
        payload = result.to_dict(include_arrays=False)
        if diff_detail is not None:
            payload['top_k_diff'] = diff_detail.to_dict(orient='records')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        self._log(f'一致性报告已保存: {path}')


# ============================================================
# 便捷函数
# ============================================================

def convert_pkl_to_pmml(
    pkl_path: str,
    pmml_path: str,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = 'target',
    with_repr: bool = True,
    debug: bool = False,
    verbose: bool = True,
) -> str:
    """函数式入口：pkl -> pmml 转换"""
    return ModelDeployer(verbose=verbose).convert_pkl_to_pmml(
        pkl_path=pkl_path,
        pmml_path=pmml_path,
        feature_names=feature_names,
        target_name=target_name,
        with_repr=with_repr,
        debug=debug,
    )


def verify_pkl_pmml_consistency(
    pkl_path: str,
    pmml_path: str,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    mode: str = 'auto',
    positive_class_index: int = 1,
    positive_label: Any = 1,
    score_field: Optional[str] = None,
    save_report: Optional[str] = None,
    return_arrays: bool = False,
    top_k_diff: int = 20,
    verbose: bool = True,
) -> ConsistencyResult:
    """函数式入口：pkl 与 pmml 一致性校验"""
    return ModelDeployer(verbose=verbose).verify_consistency(
        pkl_path=pkl_path,
        pmml_path=pmml_path,
        X=X,
        feature_names=feature_names,
        atol=atol,
        rtol=rtol,
        mode=mode,
        positive_class_index=positive_class_index,
        positive_label=positive_label,
        score_field=score_field,
        save_report=save_report,
        return_arrays=return_arrays,
        top_k_diff=top_k_diff,
    )


__all__ = [
    'ModelDeployer',
    'ConsistencyResult',
    'convert_pkl_to_pmml',
    'verify_pkl_pmml_consistency',
]
