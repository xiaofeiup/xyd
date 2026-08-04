"""
数据分析模块

提供数据质量分析功能，包括：
- 缺失值分析
- 按分组统计缺失率
- 数据分布分析
- 异常值检测
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, ProcessPoolExecutor

warnings.filterwarnings('ignore')


class DataAnalyzer:
    """数据分析器，提供全面的数据质量分析功能"""

    def __init__(self, data: pd.DataFrame, n_jobs: int = 1, min_parallel_features: int = 20):
        """
        初始化数据分析器

        Args:
            data: 待分析的数据框
        """
        self.data = data.copy()
        self.n_jobs = max(1, n_jobs)
        self.min_parallel_features = min_parallel_features
        self._numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

    def _parallel_feature_map(self, features: List[str], func, n_jobs: Optional[int] = None, use_process: bool = False) -> List[Any]:
         """按特征并行执行函数，并保持输入顺序。
         
         Args:
             features: 特征列表
             func: 处理函数
             n_jobs: 并行数
             use_process: 是否使用进程池（绕过GIL）
         """
         effective_jobs = max(1, n_jobs if n_jobs is not None else self.n_jobs)
         if effective_jobs <= 1 or len(features) < self.min_parallel_features:
             return [func(col) for col in features]

         max_workers = min(effective_jobs, len(features), os.cpu_count() or effective_jobs)
         
         # 对于数据密集操作，使用进程池绕过GIL
         executor_class = ProcessPoolExecutor if use_process else ThreadPoolExecutor
         with executor_class(max_workers=max_workers) as executor:
             return list(executor.map(func, features))

    # ==================== 缺失值分析 ====================

    def missing_summary(self, features: List[str] = None, n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        生成缺失值汇总统计

        Args:
            features: 要分析的特征列表，None表示分析所有列

        Returns:
            缺失值汇总表，包含缺失数量、缺失率等信息
        """
        if features is None:
            features = self.data.columns.tolist()

        total_rows = len(self.data)

        def _analyze_missing_col(col: str) -> Optional[Dict[str, Any]]:
            if col not in self.data.columns:
                return None

            missing_count = self.data[col].isna().sum()
            missing_rate = missing_count / total_rows if total_rows > 0 else 0
            non_missing_count = total_rows - missing_count
            dtype = str(self.data[col].dtype)

            # 计算非缺失值的统计信息
            non_missing_data = self.data[col].dropna()
            if len(non_missing_data) > 0:
                if col in self._numeric_cols:
                    unique_count = non_missing_data.nunique()
                    min_val = non_missing_data.min()
                    max_val = non_missing_data.max()
                    mean_val = non_missing_data.mean()
                else:
                    unique_count = non_missing_data.nunique()
                    min_val = None
                    max_val = None
                    mean_val = None
            else:
                unique_count = 0
                min_val = None
                max_val = None
                mean_val = None

            return {
                '特征名': col,
                '数据类型': dtype,
                '总样本数': total_rows,
                '非缺失数': non_missing_count,
                '缺失数': missing_count,
                '缺失率': f"{missing_rate:.4f}",
                '缺失率%': f"{missing_rate * 100:.2f}%",
                '唯一值数': unique_count,
                '最小值': min_val,
                '最大值': max_val,
                '均值': f"{mean_val:.4f}" if mean_val is not None else None
            }

        results = [item for item in self._parallel_feature_map(features, _analyze_missing_col, n_jobs) if item is not None]

        df_result = pd.DataFrame(results)
        # 按缺失率降序排列
        df_result['_missing_rate'] = df_result['缺失率'].astype(float)
        df_result = df_result.sort_values('_missing_rate', ascending=False).drop('_missing_rate', axis=1)
        df_result = df_result.reset_index(drop=True)

        return df_result

    def missing_by_group(self, features: List[str] = None,
                         group_col: str = None,
                         date_col: str = None,
                         date_freq: str = 'M') -> pd.DataFrame:
        """
        按分组统计缺失率

        Args:
            features: 要分析的特征列表，None表示分析所有数值列
            group_col: 分组列名（如渠道、来源等）
            date_col: 日期列名，用于按时间分组
            date_freq: 日期分组频率，'D'(日), 'W'(周), 'M'(月), 'Q'(季), 'Y'(年)

        Returns:
            按分组的缺失率统计表
            - 如果只指定group_col: 按该列分组
            - 如果只指定date_col: 按时间分组
            - 如果同时指定: 按group_col和时间双重分组
        """
        if features is None:
            features = self._numeric_cols

        if group_col is None and date_col is None:
            raise ValueError("必须指定 group_col 或 date_col 至少其一")

        data_copy = self.data.copy()

        # 处理日期列
        if date_col is not None:
            if date_col not in data_copy.columns:
                raise ValueError(f"日期列 {date_col} 不存在")

            data_copy[date_col] = pd.to_datetime(data_copy[date_col])

            freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
            if date_freq not in freq_map:
                raise ValueError(f"不支持的日期频率: {date_freq}，支持: {list(freq_map.keys())}")

            data_copy['_date_group'] = data_copy[date_col].dt.to_period(freq_map[date_freq]).astype(str)

        # 处理分组列
        if group_col is not None:
            if group_col not in data_copy.columns:
                raise ValueError(f"分组列 {group_col} 不存在")
            data_copy['_cat_group'] = data_copy[group_col].astype(str)

        # 确定分组方式
        if group_col is not None and date_col is not None:
            # 双重分组：按分类 + 时间
            data_copy['_group'] = data_copy['_cat_group'] + ' | ' + data_copy['_date_group']
            group_name = f"{group_col} | {date_col}({date_freq})"
            # 排序：先按分类，再按时间
            groups = sorted(data_copy['_group'].unique(), 
                          key=lambda x: (x.split(' | ')[0], x.split(' | ')[1]))
        elif date_col is not None:
            data_copy['_group'] = data_copy['_date_group']
            group_name = f"{date_col}({date_freq})"
            groups = sorted(data_copy['_group'].unique())
        else:
            data_copy['_group'] = data_copy['_cat_group']
            group_name = group_col
            groups = sorted(data_copy['_group'].unique())

        # 按分组统计
        results = []
        for group in groups:
            group_data = data_copy[data_copy['_group'] == group]
            group_size = len(group_data)

            row = {
                '分组': group,
                '分组名': group_name,
                '样本数': group_size
            }

            for col in features:
                if col not in self.data.columns:
                    continue
                missing_count = group_data[col].isna().sum()
                missing_rate = missing_count / group_size if group_size > 0 else 0
                row[f'{col}_缺失数'] = missing_count
                row[f'{col}_缺失率'] = f"{missing_rate:.4f}"

            results.append(row)

        # 添加总计行
        total_row = {
            '分组': '总计',
            '分组名': group_name,
            '样本数': len(data_copy)
        }
        for col in features:
            if col not in self.data.columns:
                continue
            missing_count = data_copy[col].isna().sum()
            missing_rate = missing_count / len(data_copy) if len(data_copy) > 0 else 0
            total_row[f'{col}_缺失数'] = missing_count
            total_row[f'{col}_缺失率'] = f"{missing_rate:.4f}"

        results.append(total_row)

        return pd.DataFrame(results)

    def missing_by_group_cross(self, features: List[str] = None,
                               group_col: str = None,
                               date_col: str = None,
                               date_freq: str = 'M',
                               value_type: str = 'rate',
                               n_jobs: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        交叉分组统计缺失率（每个特征生成一个透视表，行=分类，列=时间）

        Args:
            features: 要分析的特征列表
            group_col: 分组列名（必须指定）
            date_col: 日期列名（必须指定）
            date_freq: 日期分组频率
            value_type: 'rate'(缺失率) 或 'count'(缺失数)

        Returns:
            字典，key为特征名，value为该特征的交叉透视表
        """
        if features is None:
            features = self._numeric_cols

        if group_col is None or date_col is None:
            raise ValueError("交叉分组必须同时指定 group_col 和 date_col")

        if group_col not in self.data.columns:
            raise ValueError(f"分组列 {group_col} 不存在")
        if date_col not in self.data.columns:
            raise ValueError(f"日期列 {date_col} 不存在")

        # 只复制必要的列，而不是整个DataFrame
        required_cols = list(set(features + [group_col, date_col]))
        data_copy = self.data[required_cols].copy()
        data_copy[date_col] = pd.to_datetime(data_copy[date_col])

        freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
        if date_freq not in freq_map:
            raise ValueError(f"不支持的日期频率: {date_freq}")

        data_copy['_date_group'] = data_copy[date_col].dt.to_period(freq_map[date_freq]).astype(str)
        data_copy['_cat_group'] = data_copy[group_col].astype(str)

        def _build_single_feature_pivot(feature: str) -> Optional[Tuple[str, pd.DataFrame]]:
            if feature not in data_copy.columns:
                return None

            # 计算每个分组的缺失数/缺失率
            if value_type == 'count':
                pivot = data_copy.groupby(['_cat_group', '_date_group'])[feature].apply(
                    lambda x: x.isna().sum()
                ).unstack(fill_value=0)
            else:  # rate
                pivot = data_copy.groupby(['_cat_group', '_date_group'])[feature].apply(
                    lambda x: x.isna().mean()
                ).unstack(fill_value=0)
                pivot = pivot.applymap(lambda x: f"{x:.4f}")

            # 添加行总计
            if value_type == 'count':
                pivot['总计'] = data_copy.groupby('_cat_group')[feature].apply(lambda x: x.isna().sum())
            else:
                pivot['总计'] = data_copy.groupby('_cat_group')[feature].apply(
                    lambda x: f"{x.isna().mean():.4f}"
                )

            # 添加列总计
            col_totals = {}
            for col in pivot.columns:
                if col == '总计':
                    continue
                col_data = data_copy[data_copy['_date_group'] == col][feature]
                if value_type == 'count':
                    col_totals[col] = col_data.isna().sum()
                else:
                    col_totals[col] = f"{col_data.isna().mean():.4f}"

            # 总体缺失
            if value_type == 'count':
                col_totals['总计'] = data_copy[feature].isna().sum()
            else:
                col_totals['总计'] = f"{data_copy[feature].isna().mean():.4f}"

            pivot.loc['总计'] = col_totals

            pivot.index.name = group_col
            return feature, pivot

        # 使用进程池处理数据密集操作，绕过GIL
        rows = self._parallel_feature_map(features, _build_single_feature_pivot, n_jobs, use_process=True)
        result: Dict[str, pd.DataFrame] = {}
        for item in rows:
            if item is None:
                continue
            feature_name, pivot_df = item
            result[feature_name] = pivot_df

        return result

    def missing_by_group_pivot(self, features: List[str] = None,
                               group_col: str = None,
                               date_col: str = None,
                               date_freq: str = 'M',
                               value_type: str = 'rate') -> pd.DataFrame:
        """
        按分组统计缺失率（透视表格式，行是特征，列是分组）

        Args:
            features: 要分析的特征列表
            group_col: 分组列名
            date_col: 日期列名
            date_freq: 日期分组频率
            value_type: 'rate'(缺失率) 或 'count'(缺失数)

        Returns:
            透视表格式的缺失率统计
        """
        if features is None:
            features = self._numeric_cols

        # 获取分组统计
        df_grouped = self.missing_by_group(features, group_col, date_col, date_freq)

        # 转换为透视表格式
        groups = df_grouped['分组'].tolist()
        suffix = '_缺失率' if value_type == 'rate' else '_缺失数'

        pivot_data = {'特征': features}
        for group in groups:
            group_row = df_grouped[df_grouped['分组'] == group].iloc[0]
            pivot_data[group] = [group_row.get(f'{f}{suffix}', None) for f in features]

        return pd.DataFrame(pivot_data)

    def missing_correlation(self, features: List[str] = None,
                           threshold: float = 0.5) -> pd.DataFrame:
        """
        分析缺失值之间的相关性（哪些特征同时缺失）

        Args:
            features: 要分析的特征列表
            threshold: 相关性阈值，只返回相关性大于该值的特征对

        Returns:
            缺失值相关性矩阵
        """
        if features is None:
            features = [col for col in self.data.columns if self.data[col].isna().any()]

        if len(features) < 2:
            return pd.DataFrame()

        # 创建缺失值指示矩阵
        missing_matrix = self.data[features].isna().astype(int)

        # 计算相关性
        corr_matrix = missing_matrix.corr()

        # 提取高相关的特征对
        results = []
        for i, col1 in enumerate(features):
            for j, col2 in enumerate(features):
                if i < j:  # 只取上三角
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        results.append({
                            '特征1': col1,
                            '特征2': col2,
                            '缺失相关性': f"{corr:.4f}",
                            '特征1缺失率': f"{self.data[col1].isna().mean():.4f}",
                            '特征2缺失率': f"{self.data[col2].isna().mean():.4f}"
                        })

        df_result = pd.DataFrame(results)
        if len(df_result) > 0:
            df_result['_corr'] = df_result['缺失相关性'].astype(float).abs()
            df_result = df_result.sort_values('_corr', ascending=False).drop('_corr', axis=1)
            df_result = df_result.reset_index(drop=True)

        return df_result

    # ==================== 数据分布分析 ====================

    def distribution_summary(self, features: List[str] = None, n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        生成数值特征分布统计

        Args:
            features: 要分析的特征列表，None表示分析所有数值列

        Returns:
            分布统计表
        """
        if features is None:
            features = self._numeric_cols

        def _analyze_distribution_col(col: str) -> Optional[Dict[str, Any]]:
            if col not in self.data.columns or col not in self._numeric_cols:
                return None

            data = self.data[col].dropna()
            if len(data) == 0:
                return None

            return {
                '特征名': col,
                '样本数': len(data),
                '均值': f"{data.mean():.4f}",
                '标准差': f"{data.std():.4f}",
                '最小值': f"{data.min():.4f}",
                '25%分位': f"{data.quantile(0.25):.4f}",
                '中位数': f"{data.median():.4f}",
                '75%分位': f"{data.quantile(0.75):.4f}",
                '最大值': f"{data.max():.4f}",
                '偏度': f"{data.skew():.4f}",
                '峰度': f"{data.kurtosis():.4f}"
            }

        results = [item for item in self._parallel_feature_map(features, _analyze_distribution_col, n_jobs) if item is not None]

        return pd.DataFrame(results)

    def distribution_by_group(self, feature: str,
                              group_col: str = None,
                              date_col: str = None,
                              date_freq: str = 'M') -> pd.DataFrame:
        """
        按分组统计单个特征的分布

        Args:
            feature: 要分析的特征
            group_col: 分组列名
            date_col: 日期列名
            date_freq: 日期分组频率

        Returns:
            按分组的分布统计表
        """
        if feature not in self.data.columns:
            raise ValueError(f"特征 {feature} 不存在")

        if group_col is not None and date_col is not None:
            raise ValueError("group_col 和 date_col 只能指定其一")

        if group_col is None and date_col is None:
            raise ValueError("必须指定 group_col 或 date_col")

        data_copy = self.data.copy()

        if date_col is not None:
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
            data_copy['_group'] = data_copy[date_col].dt.to_period(date_freq).astype(str)
            group_name = f"{date_col}({date_freq})"
        else:
            data_copy['_group'] = data_copy[group_col].astype(str)
            group_name = group_col

        results = []
        groups = sorted(data_copy['_group'].unique())

        for group in groups:
            group_data = data_copy[data_copy['_group'] == group][feature].dropna()

            if len(group_data) == 0:
                continue

            results.append({
                '分组': group,
                '分组名': group_name,
                '特征': feature,
                '样本数': len(group_data),
                '缺失率': f"{1 - len(group_data) / len(data_copy[data_copy['_group'] == group]):.4f}",
                '均值': f"{group_data.mean():.4f}",
                '标准差': f"{group_data.std():.4f}",
                '最小值': f"{group_data.min():.4f}",
                '中位数': f"{group_data.median():.4f}",
                '最大值': f"{group_data.max():.4f}"
            })

        return pd.DataFrame(results)

    # ==================== 分类特征分析 ====================

    def categorical_summary(self, features: List[str] = None,
                            top_n: int = 10,
                            n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        生成分类特征统计

        Args:
            features: 要分析的特征列表，None表示分析所有分类列
            top_n: 显示前N个最常见的值

        Returns:
            分类特征统计表
        """
        if features is None:
            features = self._categorical_cols

        total_rows = len(self.data)

        def _analyze_categorical_col(col: str) -> Optional[Dict[str, Any]]:
            if col not in self.data.columns:
                return None

            data = self.data[col].dropna()
            value_counts = data.value_counts()

            return {
                '特征名': col,
                '总样本数': total_rows,
                '非缺失数': len(data),
                '缺失率': f"{1 - len(data) / total_rows:.4f}",
                '唯一值数': data.nunique(),
                '最常见值': value_counts.index[0] if len(value_counts) > 0 else None,
                '最常见值占比': f"{value_counts.iloc[0] / len(data):.4f}" if len(value_counts) > 0 else None,
                f'Top{top_n}值': ', '.join(map(str, value_counts.index[:top_n].tolist()))
            }

        results = [item for item in self._parallel_feature_map(features, _analyze_categorical_col, n_jobs) if item is not None]

        return pd.DataFrame(results)

    def categorical_by_group(self, feature: str,
                             group_col: str = None,
                             date_col: str = None,
                             date_freq: str = 'M') -> pd.DataFrame:
        """
        按分组统计分类特征的分布

        Args:
            feature: 要分析的分类特征
            group_col: 分组列名
            date_col: 日期列名
            date_freq: 日期分组频率

        Returns:
            按分组的分类分布统计
        """
        if feature not in self.data.columns:
            raise ValueError(f"特征 {feature} 不存在")

        if group_col is not None and date_col is not None:
            raise ValueError("group_col 和 date_col 只能指定其一")

        if group_col is None and date_col is None:
            raise ValueError("必须指定 group_col 或 date_col")

        data_copy = self.data.copy()

        if date_col is not None:
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
            data_copy['_group'] = data_copy[date_col].dt.to_period(date_freq).astype(str)
        else:
            data_copy['_group'] = data_copy[group_col].astype(str)

        # 获取所有类别值
        all_categories = data_copy[feature].dropna().unique()

        results = []
        groups = sorted(data_copy['_group'].unique())

        for group in groups:
            group_data = data_copy[data_copy['_group'] == group]
            group_size = len(group_data)

            row = {
                '分组': group,
                '样本数': group_size,
                '缺失数': group_data[feature].isna().sum()
            }

            # 统计每个类别的占比
            value_counts = group_data[feature].value_counts()
            for cat in all_categories:
                count = value_counts.get(cat, 0)
                row[f'{cat}_数量'] = count
                row[f'{cat}_占比'] = f"{count / group_size:.4f}" if group_size > 0 else "0.0000"

            results.append(row)

        return pd.DataFrame(results)

    # ==================== 异常值分析 ====================

    def outlier_summary(self, features: List[str] = None,
                        method: str = 'iqr',
                        threshold: float = 1.5,
                        n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        异常值统计

        Args:
            features: 要分析的特征列表
            method: 检测方法，'iqr'(四分位距法) 或 'zscore'(Z分数法)
            threshold: 阈值，IQR法默认1.5，Z分数法默认3

        Returns:
            异常值统计表
        """
        if features is None:
            features = self._numeric_cols

        def _analyze_outlier_col(col: str) -> Optional[Dict[str, Any]]:
            if col not in self.data.columns or col not in self._numeric_cols:
                return None

            data = self.data[col].dropna()
            if len(data) == 0:
                return None

            if method == 'iqr':
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            elif method == 'zscore':
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                outliers = data[z_scores > threshold]
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                raise ValueError(f"不支持的方法: {method}，支持: 'iqr', 'zscore'")

            return {
                '特征名': col,
                '样本数': len(data),
                '异常值数': len(outliers),
                '异常值率': f"{len(outliers) / len(data):.4f}",
                '下界': f"{lower_bound:.4f}",
                '上界': f"{upper_bound:.4f}",
                '低于下界数': len(data[data < lower_bound]),
                '高于上界数': len(data[data > upper_bound])
            }

        results = [item for item in self._parallel_feature_map(features, _analyze_outlier_col, n_jobs) if item is not None]

        return pd.DataFrame(results)

    # ==================== 综合报告 ====================

    def generate_full_report(self, features: List[str] = None,
                             group_col: str = None,
                             date_col: str = None,
                             date_freq: str = 'M',
                             save_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        生成完整的数据分析报告

        Args:
            features: 要分析的特征列表
            group_col: 分组列名
            date_col: 日期列名
            date_freq: 日期分组频率
            save_path: 保存路径

        Returns:
            包含所有分析结果的字典
        """
        report = {}

        # 1. 缺失值汇总
        try:
            report['缺失值汇总'] = self.missing_summary(features)
            print("✓ 缺失值汇总完成")
        except Exception as e:
            print(f"✗ 缺失值汇总失败: {e}")
            report['缺失值汇总'] = pd.DataFrame()

        # 2. 按分组统计缺失率
        if group_col or date_col:
            try:
                report['按分组缺失率'] = self.missing_by_group(
                    features, group_col=group_col, date_col=date_col, date_freq=date_freq
                )
                print("✓ 按分组缺失率统计完成")
            except Exception as e:
                print(f"✗ 按分组缺失率统计失败: {e}")
                report['按分组缺失率'] = pd.DataFrame()

            try:
                report['按分组缺失率透视'] = self.missing_by_group_pivot(
                    features, group_col=group_col, date_col=date_col, date_freq=date_freq
                )
                print("✓ 按分组缺失率透视表完成")
            except Exception as e:
                print(f"✗ 按分组缺失率透视表失败: {e}")
                report['按分组缺失率透视'] = pd.DataFrame()

        # 3. 缺失值相关性
        try:
            report['缺失值相关性'] = self.missing_correlation(features)
            print("✓ 缺失值相关性分析完成")
        except Exception as e:
            print(f"✗ 缺失值相关性分析失败: {e}")
            report['缺失值相关性'] = pd.DataFrame()

        # 4. 数值分布统计
        try:
            report['数值分布统计'] = self.distribution_summary(features)
            print("✓ 数值分布统计完成")
        except Exception as e:
            print(f"✗ 数值分布统计失败: {e}")
            report['数值分布统计'] = pd.DataFrame()

        # 5. 分类特征统计
        try:
            report['分类特征统计'] = self.categorical_summary()
            print("✓ 分类特征统计完成")
        except Exception as e:
            print(f"✗ 分类特征统计失败: {e}")
            report['分类特征统计'] = pd.DataFrame()

        # 6. 异常值统计
        try:
            report['异常值统计'] = self.outlier_summary(features)
            print("✓ 异常值统计完成")
        except Exception as e:
            print(f"✗ 异常值统计失败: {e}")
            report['异常值统计'] = pd.DataFrame()

        # 保存报告
        if save_path:
            try:
                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    for sheet_name, df in report.items():
                        if not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"✓ 报告已保存至: {save_path}")
            except Exception as e:
                print(f"✗ 报告保存失败: {e}")

        return report


def analyze_missing(data: pd.DataFrame,
                    features: List[str] = None,
                    group_col: str = None,
                    date_col: str = None,
                    date_freq: str = 'M') -> Dict[str, pd.DataFrame]:
    """
    快捷函数：分析缺失值

    Args:
        data: 数据框
        features: 特征列表
        group_col: 分组列
        date_col: 日期列
        date_freq: 日期分组频率

    Returns:
        缺失值分析结果字典
    """
    analyzer = DataAnalyzer(data)
    result = {
        'summary': analyzer.missing_summary(features)
    }

    if group_col or date_col:
        result['by_group'] = analyzer.missing_by_group(
            features, group_col=group_col, date_col=date_col, date_freq=date_freq
        )
        result['pivot'] = analyzer.missing_by_group_pivot(
            features, group_col=group_col, date_col=date_col, date_freq=date_freq
        )

    return result


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 创建测试数据
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'channel': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['x', 'y', 'z'], n_samples),
    })

    # 添加缺失值
    data.loc[data.index[:100], 'feature1'] = np.nan
    data.loc[data.index[200:350], 'feature2'] = np.nan
    # 让channel=A的缺失率更高
    data.loc[(data['channel'] == 'A') & (data.index < 200), 'feature1'] = np.nan

    print("=" * 50)
    print("数据分析模块测试")
    print("=" * 50)

    analyzer = DataAnalyzer(data)

    print("\n1. 缺失值汇总:")
    print(analyzer.missing_summary())

    print("\n2. 按渠道统计缺失率:")
    print(analyzer.missing_by_group(['feature1', 'feature2'], group_col='channel'))

    print("\n3. 按月统计缺失率:")
    print(analyzer.missing_by_group(['feature1', 'feature2'], date_col='date', date_freq='M'))

    print("\n4. 缺失率透视表（按月）:")
    print(analyzer.missing_by_group_pivot(['feature1', 'feature2'], date_col='date', date_freq='M'))

    print("\n5. 数值分布统计:")
    print(analyzer.distribution_summary())

    print("\n6. 分类特征统计:")
    print(analyzer.categorical_summary())

    print("\n7. 异常值统计:")
    print(analyzer.outlier_summary())

    print("\n测试完成!")
