#!/usr/bin/env python3
"""
自动化建模流水线 — 配置驱动 + CLI

使用方式:
    # 方式1: 用配置文件
    python3 auto_modeling_pipeline.py --config pipeline_config.yaml

    # 方式2: 用配置文件 + 覆盖单个参数
    python3 auto_modeling_pipeline.py --config pipeline_config.yaml --n_trials 50

    # 方式3: 跳过调参, 复用已有参数
    python3 auto_modeling_pipeline.py --config pipeline_config.yaml --skip-tuning --reuse_params output/best_params.json

    # 方式4: 仅运行特定步骤 (1=数据分析 2=特征筛选 3=调参 5=模型评估 6=报告)
    python3 auto_modeling_pipeline.py --config pipeline_config.yaml --steps 1,2,5,6

流程: 数据分析 → 特征筛选 → 自动调参 → 模型评估 → HTML 报告生成
"""

import sys
import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')


# ============================================================
# 进度可视化
# ============================================================
class ProgressBar:
    """终端进度条"""

    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        pct = self.current / self.total * 100 if self.total > 0 else 100
        filled = int(self.width * self.current / self.total) if self.total > 0 else self.width
        bar = '█' * filled + '░' * (self.width - filled)
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  {self.desc} |{bar}| {pct:5.1f}% [{self.current}/{self.total}] {rate:.1f}it/s ETA:{eta:.0f}s"
        )
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write('\n')

    def finish(self):
        if self.current < self.total:
            self.current = self.total
            self.update(0)


def step_header(step_num: int, title: str, total_steps: int = 7):
    """打印步骤头部"""
    print(f"\n{'━' * 60}")
    print(f"  ● 步骤 {step_num}/{total_steps}: {title}")
    print(f"{'━' * 60}")


def step_done(step_num: int, title: str, elapsed: float, detail: str = ""):
    """打印步骤完成"""
    status = f"✓ {title} ({elapsed:.1f}s)"
    if detail:
        status += f" — {detail}"
    print(f"  {status}")


# ============================================================
# 内存优化的 Tuner 子类
# ============================================================
def _create_tuner_class():
    """延迟导入, 避免 toolkit 路径未配置时报错"""
    from model_tools.optimization.anti_overfitting_tuner import AntiOverfittingTuner

    class PipelineTuner(AntiOverfittingTuner):
        def _suggest_parameters(self, trial):
            params = super()._suggest_parameters(trial)
            params['n_jobs'] = self._pipeline_n_jobs
            return params
        _pipeline_n_jobs = 1

    return PipelineTuner


# ============================================================
# 工具函数
# ============================================================
def calc_ks(y_true, y_pred):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return max(tpr - fpr)
    except:
        return 0.0


def load_yaml_config(path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return flatten_config(cfg)


def flatten_config(cfg: dict) -> dict:
    """将嵌套配置展平为单层 dict (兼容旧代码)"""
    flat = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[k] = v
        else:
            flat[section] = values
    return flat


def serialize_params(params: dict) -> dict:
    """将 numpy 类型转为 Python 原生类型"""
    result = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            result[k] = bool(v)
        else:
            result[k] = v
    return result


# ============================================================
# 流水线主类
# ============================================================
class ModelingPipeline:
    """自动化建模流水线"""

    def __init__(self, config: dict, steps: list = None):
        """
        Args:
            config: 展平后的配置 dict
            steps: 要执行的步骤列表, None 表示全部执行
                   可选: [1, 2, 3, 5, 6, 7]
        """
        self.cfg = config
        self.steps = steps or [1, 2, 3, 5, 6, 7]
        self.data = None
        self.selected_features = None
        self.best_params = None
        self.tuning_history = None
        self.final_model = None
        self.y_pred_train = None
        self.y_pred_oos = None
        self.delivery_report = None
        self.feature_importance = None
        self.pipeline_log = {}

        # 路径
        self.output_dir = config.get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化 toolkit
        toolkit_path = os.path.expanduser(config.get('toolkit_path', '~/Job/Job_xyd/code/工具包/model_tools_new'))
        if toolkit_path not in sys.path:
            sys.path.insert(0, toolkit_path)

    def run(self) -> dict:
        """执行流水线"""
        start_time = datetime.now()
        self.pipeline_log['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'自动化建模流水线':^42}{'启动':^16}║")
        print(f"{'╚' + '═' * 58 + '╝'}")
        print(f"  配置: output={self.output_dir}")
        print(f"  步骤: {self.steps}")

        total_steps = len(self.steps)
        step_idx = 0

        # 数据加载 (始终执行)
        self._load_data()

        if 1 in self.steps:
            step_idx += 1
            self._step_data_analysis(step_idx, total_steps)

        if 2 in self.steps:
            step_idx += 1
            self._step_feature_selection(step_idx, total_steps)

        if 3 in self.steps:
            step_idx += 1
            self._step_tuning(step_idx, total_steps)
        else:
            # 跳过调参, 加载已有参数
            reuse_path = self.cfg.get('reuse_params_path')
            if reuse_path:
                # 尝试相对路径和绝对路径
                candidates = [
                    reuse_path,
                    os.path.join(self.output_dir, reuse_path),
                    os.path.join(os.path.dirname(self.cfg.get('feature_path', '.')), reuse_path),
                ]
                found = False
                for p in candidates:
                    if os.path.exists(p):
                        with open(p) as f:
                            self.best_params = json.load(f)
                        print(f"\n  ⏭ 跳过调参, 复用参数: {p}")
                        found = True
                        break
                if not found:
                    # 也检查 output_dir 下的 best_params.json
                    fallback = os.path.join(self.output_dir, 'best_params.json')
                    if os.path.exists(fallback):
                        with open(fallback) as f:
                            self.best_params = json.load(f)
                        print(f"\n  ⏭ 跳过调参, 复用参数: {fallback}")
                    else:
                        print(f"\n  ⚠ 未找到参数文件: {reuse_path}")

        if 5 in self.steps:
            step_idx += 1
            self._step_train_and_evaluate(step_idx, total_steps)

        if 6 in self.steps or 7 in self.steps:
            step_idx += 1
            self._step_reports(step_idx, total_steps)

        # 总结
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.pipeline_log['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
        self.pipeline_log['duration_seconds'] = duration

        summary_path = os.path.join(self.output_dir, 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.pipeline_log, f, indent=2, ensure_ascii=False)

        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'流水线执行完成':^50}║")
        print(f"{'╠' + '═' * 58 + '╣'}")
        print(f"║  耗时: {duration:.0f} 秒 ({duration / 60:.1f} 分钟){' ' * 30}║")
        if self.pipeline_log.get('train_auc'):
            print(f"║  训练集 AUC: {self.pipeline_log['train_auc']:.4f}{' ' * 38}║")
            print(f"║  OOS AUC:    {self.pipeline_log['oos_auc']:.4f}{' ' * 40}║")
            gap = abs(self.pipeline_log['train_auc'] - self.pipeline_log['oos_auc'])
            print(f"║  AUC 差距:   {gap:.4f}{' ' * 40}║")
        print(f"║  输出目录: {self.output_dir:<46}║")
        print(f"{'╚' + '═' * 58 + '╝'}")

        return self.pipeline_log

    # ====== 数据加载 ======
    def _load_data(self):
        cfg = self.cfg
        print(f"\n  📂 加载数据...")

        ym = pd.read_csv(cfg['feature_path'])
        df_label = pd.read_csv(cfg['label_path'])
        merge_cols = [cfg['key_col'], cfg['target_col'], cfg['sample_type_col']]
        if cfg.get('date_col') and cfg['date_col'] in df_label.columns:
            merge_cols.append(cfg['date_col'])

        self.data = ym.merge(
            df_label[merge_cols], on=cfg['key_col'], how='inner',
            suffixes=('', '_label')
        )

        # 确定特征列
        exclude = set(cfg.get('exclude_cols', []))
        exclude.add(cfg['key_col'])
        exclude.add(cfg['target_col'])
        exclude.add(cfg['sample_type_col'])
        # exclude 添加 '_label'结尾的col
        exclude.add(f for f in self.data.columns if f.endswith('_label'))
        if cfg.get('date_col'):
            exclude.add(cfg['date_col'])
        all_features = [c for c in self.data.columns if c not in exclude]
        self.numeric_features = self.data[all_features].select_dtypes(include=[np.number]).columns.tolist()

        print(f"     合并后: {self.data.shape[0]:,} 行 × {self.data.shape[1]} 列")
        print(f"     数值特征: {len(self.numeric_features)}")
        print(f"     坏样本率: {self.data[cfg['target_col']].mean():.4f}")

        self.pipeline_log['initial_shape'] = str(self.data.shape)
        self.pipeline_log['initial_features'] = len(self.numeric_features)

    # ====== 步骤 1: 数据分析 ======
    def _step_data_analysis(self, step_idx, total_steps):
        step_header(step_idx, "数据分析", total_steps)
        t0 = time.time()

        from model_tools.features.data_analysis_optimized import DataAnalyzer
        from html_report_generator import generate_data_analysis_html

        analyzer = DataAnalyzer(self.data, n_jobs=1)
        html_path = os.path.join(self.output_dir, 'step1_data_analysis.html')
        generate_data_analysis_html(
            analyzer, features=self.numeric_features,
            save_path=html_path, title="数据分析报告"
        )
        elapsed = time.time() - t0
        step_done(step_idx, "数据分析", elapsed, f"→ {html_path}")

    # ====== 步骤 2: 特征筛选 ======
    def _step_feature_selection(self, step_idx, total_steps):
        step_header(step_idx, "特征筛选", total_steps)
        t0 = time.time()
        cfg = self.cfg
        data = self.data
        feas = self.numeric_features

        # 2a. 高缺失率
        missing_rates = data[feas].isna().mean()
        high_missing = missing_rates[missing_rates > cfg['missing_threshold']].index.tolist()
        print(f"  缺失率 > {cfg['missing_threshold']:.0%}: {len(high_missing)} 个剔除")

        # 2b. 零方差
        vt = VarianceThreshold(threshold=0)
        vt.fit(data[feas].fillna(-999))
        zero_var = [feas[i] for i in range(len(feas)) if not vt.get_support()[i]]
        print(f"  零方差: {len(zero_var)} 个剔除")

        remaining = [f for f in feas if f not in high_missing and f not in zero_var]
        print(f"  预筛选后: {len(remaining)} 个特征")

        # 2c. IV + 单一值占比
        train_mask = data[cfg['sample_type_col']] == cfg['train_label']
        X_train_all = data.loc[train_mask, remaining]
        y_train_all = data.loc[train_mask, cfg['target_col']]

        from model_tools.features.selection import FeatureSelector

        selector = FeatureSelector(
            method='iv',
            single_value_threshold=cfg['single_value_threshold'],
            iv_threshold=cfg['iv_threshold']
        )
        selector.fit(X_train_all, y_train_all)
        self.selected_features = selector.selected_features_ or []

        # 如果为 0, 降级重试
        if len(self.selected_features) == 0:
            print(f"  ⚠ 筛选后 0 特征, 降级重试 (iv={cfg.get('iv_threshold_fallback', 0.01)})")
            selector = FeatureSelector(
                method='iv',
                single_value_threshold=cfg.get('single_value_fallback', 0.99),
                iv_threshold=cfg.get('iv_threshold_fallback', 0.01)
            )
            selector.fit(X_train_all, y_train_all)
            self.selected_features = selector.selected_features_ or remaining[:50]

        # 保存日志
        iv_log = selector.get_iv_log()
        iv_log.to_csv(os.path.join(self.output_dir, 'step2_feature_selection_log.csv'), index=False)
        with open(os.path.join(self.output_dir, 'selected_features.json'), 'w') as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)

        self.pipeline_log['features_after_selection'] = len(self.selected_features)
        elapsed = time.time() - t0
        step_done(step_idx, "特征筛选", elapsed, f"{len(self.selected_features)} 个特征保留")

    # ====== 步骤 3: 自动调参 ======
    def _step_tuning(self, step_idx, total_steps):
        step_header(step_idx, "自动调参", total_steps)
        t0 = time.time()
        cfg = self.cfg

        import lightgbm as lgb
        PipelineTuner = _create_tuner_class()
        PipelineTuner._pipeline_n_jobs = cfg.get('n_jobs', 1)

        train_mask = self.data[cfg['sample_type_col']] == cfg['train_label']
        X_train = self.data.loc[train_mask, self.selected_features].copy()
        y_train = self.data.loc[train_mask, cfg['target_col']].copy()

        n_trials = cfg['n_trials']
        print(f"  Optuna TPE 搜索: {n_trials} trials × {cfg['cv_folds']} 折 CV = {n_trials * cfg['cv_folds']} 次训练")

        tuner = PipelineTuner(
            model_class=lgb.LGBMClassifier,
            model_type='lgb',
            direction='maximize',
            max_auc_gap=cfg['max_auc_gap'],
            max_ks_gap=cfg['max_ks_gap'],
            overfitting_penalty_weight=cfg.get('overfitting_penalty_weight', 2.0)
        )

        result = tuner.optimize_anti_overfitting(
            X_train, y_train,
            n_trials=n_trials,
            cv_folds=cfg['cv_folds'],
            n_jobs=1,
            show_progress_bar=False
        )

        valid = result['results_summary']['valid_trials']
        total = result['results_summary']['total_trials']
        print(f"  完成! 有效 trial: {valid}/{total}, 平均 AUC gap: {result['results_summary']['avg_auc_gap']:.4f}")

        # 选最优参数
        history = pd.DataFrame(result['performance_history'])
        history.to_csv(os.path.join(self.output_dir, 'step3_tuning_raw_history.csv'), index=False)
        history['auc_gap_abs'] = (history['train_auc'] - history['val_auc']).abs()

        if len(history) > 0:
            val_weight = cfg.get('val_auc_weight', 0.5)
            val_range = history['val_auc'].max() - history['val_auc'].min() + 1e-10
            gap_range = history['auc_gap_abs'].max() - history['auc_gap_abs'].min() + 1e-10
            history['score'] = (
                (history['val_auc'] - history['val_auc'].min()) / val_range * val_weight
                + (1 - (history['auc_gap_abs'] - history['auc_gap_abs'].min()) / gap_range) * (1 - val_weight)
            )
            best_row = history.loc[history['score'].idxmax()]
            self.best_params = best_row['params']
            print(f"  最优 Trial #{best_row['trial']}: val_auc={best_row['val_auc']:.4f}, gap={best_row['auc_gap_abs']:.4f}")
        else:
            self.best_params = result.get('recommended_params') or result.get('best_params', {})

        self.tuning_history = history
        self.best_params = serialize_params(self.best_params)

        # 保存
        history_export = history.drop(columns=['params'], errors='ignore')
        history_export.to_csv(os.path.join(self.output_dir, 'step3_tuning_history.csv'), index=False)
        with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        self.pipeline_log['best_val_auc'] = float(best_row['val_auc']) if len(history) > 0 else None
        elapsed = time.time() - t0
        step_done(step_idx, "自动调参", elapsed, f"best val_auc={self.pipeline_log.get('best_val_auc', 'N/A')}")

    # ====== 步骤 5: 训练 + 评估 ======
    def _step_train_and_evaluate(self, step_idx, total_steps):
        step_header(step_idx, "模型训练与评估", total_steps)
        t0 = time.time()
        cfg = self.cfg

        import lightgbm as lgb

        # 如果跳过了调参, 从文件加载参数
        if self.best_params is None:
            params_path = os.path.join(self.output_dir, 'best_params.json')
            if os.path.exists(params_path):
                with open(params_path) as f:
                    self.best_params = json.load(f)
                print(f"  从文件加载参数: {params_path}")
            else:
                raise FileNotFoundError(f"未找到参数文件 {params_path}, 请先运行调参步骤")

        self.best_params['n_jobs'] = cfg.get('n_jobs', 1)
        train_mask = self.data[cfg['sample_type_col']] == cfg['train_label']
        oos_mask = self.data[cfg['sample_type_col']] == cfg['oos_label']

        # 如果跳过了特征筛选, 从文件加载
        if self.selected_features is None:
            sf_path = os.path.join(self.output_dir, 'selected_features.json')
            if os.path.exists(sf_path):
                with open(sf_path) as f:
                    self.selected_features = json.load(f)
            else:
                self.selected_features = self.numeric_features

        X_train = self.data.loc[train_mask, self.selected_features].copy()
        y_train = self.data.loc[train_mask, cfg['target_col']].copy()
        X_oos = self.data.loc[oos_mask, self.selected_features].copy()
        y_oos = self.data.loc[oos_mask, cfg['target_col']].copy()

        print(f"  训练模型: {len(self.selected_features)} 特征, {len(X_train):,} 样本")
        self.final_model = lgb.LGBMClassifier(**self.best_params)
        self.final_model.fit(X_train, y_train)

        self.y_pred_train = self.final_model.predict_proba(X_train)[:, 1]
        self.y_pred_oos = self.final_model.predict_proba(X_oos)[:, 1]

        train_auc = roc_auc_score(y_train, self.y_pred_train)
        oos_auc = roc_auc_score(y_oos, self.y_pred_oos)
        train_ks = calc_ks(y_train, self.y_pred_train)
        oos_ks = calc_ks(y_oos, self.y_pred_oos)

        print(f"  {'指标':<10} {'训练集':>10} {'OOS':>10} {'差距':>10}")
        print(f"  {'AUC':<10} {train_auc:>10.4f} {oos_auc:>10.4f} {abs(train_auc - oos_auc):>10.4f}")
        print(f"  {'KS':<10} {train_ks:>10.4f} {oos_ks:>10.4f} {abs(train_ks - oos_ks):>10.4f}")

        gap = abs(train_auc - oos_auc)
        tag = "✅ 通过" if gap <= cfg['max_auc_gap'] else "⚠️ 需关注"
        print(f"  过拟合检查 ({tag}): AUC gap={gap:.4f} 阈值={cfg['max_auc_gap']}")

        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 保存 OOS 预测
        oos_pred_df = pd.DataFrame({
            cfg['key_col']: self.data.loc[oos_mask, cfg['key_col']].values,
            cfg['target_col']: y_oos.values,
            'pred_proba': self.y_pred_oos
        })
        oos_pred_df.to_csv(os.path.join(self.output_dir, 'oos_predictions.csv'), index=False)

        self.pipeline_log.update({
            'train_auc': float(train_auc), 'oos_auc': float(oos_auc),
            'train_ks': float(train_ks), 'oos_ks': float(oos_ks),
        })
        elapsed = time.time() - t0
        step_done(step_idx, "模型训练与评估", elapsed, f"AUC={oos_auc:.4f}")

    # ====== 步骤 6+7: 报告生成 ======
    def _step_reports(self, step_idx, total_steps):
        step_header(step_idx, "报告生成", total_steps)
        t0 = time.time()
        cfg = self.cfg

        from model_tools.evaluation.delivery_report import ModelDeliveryReport
        from html_report_generator import generate_model_report_html

        train_mask = self.data[cfg['sample_type_col']] == cfg['train_label']
        oos_mask = self.data[cfg['sample_type_col']] == cfg['oos_label']
        y_train = self.data.loc[train_mask, cfg['target_col']]
        y_oos = self.data.loc[oos_mask, cfg['target_col']]

        # 6a. ModelDeliveryReport
        print("  生成交付报告 (generate_full_report)...")
        report_data = self.data.copy()
        report_data['pred_score'] = 0.0
        report_data.loc[train_mask, 'pred_score'] = self.y_pred_train
        report_data.loc[oos_mask, 'pred_score'] = self.y_pred_oos

        type_map = {cfg['train_label']: 'train', cfg['oos_label']: 'test'}
        report_data['sample_type_report'] = report_data[cfg['sample_type_col']].map(type_map).fillna('all')
        if cfg.get('date_col') and cfg['date_col'] in report_data.columns:
            report_data[cfg['date_col']] = pd.to_datetime(report_data[cfg['date_col']], errors='coerce')

        try:
            reporter = ModelDeliveryReport(
                data=report_data,
                target_col=cfg['target_col'],
                score_col='pred_score',
                date_col=cfg.get('date_col', 'date'),
                sample_type_col=cfg['sample_type_col'],
                score_direction='higher_is_bad'
            )
            top_k = cfg.get('top_k_features', 20)
            self.delivery_report = reporter.generate_full_report(
                features=self.selected_features[:top_k],
                model=self.final_model,
                importance_type=cfg.get('importance_type', 'gain'),
                n_bins=cfg.get('n_bins', 10),
                save_path=os.path.join(self.output_dir, 'step4_model_delivery_report.xlsx')
            )
            n_reports = len([k for k in self.delivery_report if not k.startswith('_')])
            print(f"  交付报表: {n_reports} 个")
        except Exception as e:
            print(f"  交付报告失败 (非致命): {e}")
            self.delivery_report = {}

        # 6b. HTML 报告
        print("  生成 HTML 报告...")
        html_path = os.path.join(self.output_dir, 'model_report.html')
        generate_model_report_html(
            report_data=self.delivery_report,
            y_true_train=y_train.values,
            y_score_train=self.y_pred_train,
            y_true_oos=y_oos.values,
            y_score_oos=self.y_pred_oos,
            best_params=self.best_params,
            tuning_history=self.tuning_history,
            feature_importance=self.feature_importance,
            save_path=html_path,
            title="模型交付报告"
        )

        elapsed = time.time() - t0
        step_done(step_idx, "报告生成", elapsed, f"→ {html_path}")


# ============================================================
# CLI
# ============================================================
def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='自动化建模流水线: 数据分析 → 特征筛选 → 调参 → 评估 → 报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标准全流程
  python3 auto_modeling_pipeline.py --config pipeline_config.yaml

  # 覆盖调参次数
  python3 auto_modeling_pipeline.py --config pipeline_config.yaml --n_trials 50

  # 跳过调参, 复用已有参数
  python3 auto_modeling_pipeline.py --config pipeline_config.yaml --skip-tuning --reuse_params output/best_params.json

  # 仅运行数据分析 + 特征筛选
  python3 auto_modeling_pipeline.py --config pipeline_config.yaml --steps 1,2
        """
    )
    parser.add_argument('--config', '-c', type=str, default='pipeline_config.yaml',
                        help='配置文件路径 (YAML)')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='覆盖调参试验次数')
    parser.add_argument('--cv_folds', type=int, default=None,
                        help='覆盖交叉验证折数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='覆盖输出目录')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='跳过调参步骤')
    parser.add_argument('--reuse_params', type=str, default=None,
                        help='复用已有参数文件路径 (配合 --skip-tuning)')
    parser.add_argument('--steps', type=str, default=None,
                        help='仅运行特定步骤, 逗号分隔 (1=数据分析 2=特征筛选 3=调参 5=训练评估 6=报告)')
    parser.add_argument('--feature_path', type=str, default=None,
                        help='覆盖特征文件路径')
    parser.add_argument('--label_path', type=str, default=None,
                        help='覆盖标签文件路径')
    return parser


def main():
    parser = build_cli()
    args = parser.parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print(f"   请先创建配置文件, 参考 pipeline_config.yaml")
        sys.exit(1)

    config = load_yaml_config(args.config)

    # CLI 覆盖
    if args.n_trials is not None:
        config['n_trials'] = args.n_trials
    if args.cv_folds is not None:
        config['cv_folds'] = args.cv_folds
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.feature_path is not None:
        config['feature_path'] = args.feature_path
    if args.label_path is not None:
        config['label_path'] = args.label_path
    if args.reuse_params is not None:
        config['reuse_params_path'] = args.reuse_params

    # 步骤控制
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(',')]
    elif args.skip_tuning:
        steps = [1, 2, 5, 6, 7]  # 跳过 3 (调参)
        if args.reuse_params:
            config['reuse_params_path'] = args.reuse_params
    else:
        steps = None  # 全部

    # 执行
    pipeline = ModelingPipeline(config, steps=steps)
    result = pipeline.run()

    return result


if __name__ == '__main__':
    main()
