"""
性能对比脚本：原始版本 vs 优化版本

使用方法：
python compare_performance.py
"""

import pandas as pd
import numpy as np
import time
import sys

# 创建测试数据
def create_test_data(n_rows=50000):
    """创建测试数据集"""
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'feature3': np.random.randn(n_rows),
        'feature4': np.random.randn(n_rows),
        'feature5': np.random.randn(n_rows),
        'partner_name': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'apply_date': pd.date_range('2023-01-01', periods=n_rows, freq='H')
    })
    
    # 添加缺失值
    data.loc[data.index[:5000], 'feature1'] = np.nan
    data.loc[data.index[10000:15000], 'feature2'] = np.nan
    data.loc[data.index[20000:22000], 'feature3'] = np.nan
    
    return data

def benchmark_original(data, features, n_jobs=1):
    """测试原始版本"""
    try:
        from data_analysis import DataAnalyzer
        analyzer = DataAnalyzer(data, n_jobs=n_jobs)
        
        start = time.time()
        result = analyzer.missing_by_group_cross(
            features=features,
            group_col='partner_name',
            date_col='apply_date',
            date_freq='M',
            value_type='rate',
            n_jobs=n_jobs
        )
        elapsed = time.time() - start
        return elapsed, len(result)
    except Exception as e:
        print(f"原始版本错误: {e}")
        return None, None

def benchmark_optimized(data, features):
    """测试优化版本"""
    try:
        from data_analysis_optimized import DataAnalyzer
        analyzer = DataAnalyzer(data)
        
        start = time.time()
        result = analyzer.missing_by_group_cross(
            features=features,
            group_col='partner_name',
            date_col='apply_date',
            date_freq='M',
            value_type='rate'
        )
        elapsed = time.time() - start
        return elapsed, len(result)
    except Exception as e:
        print(f"优化版本错误: {e}")
        return None, None

if __name__ == '__main__':
    print("=" * 60)
    print("并行逻辑性能对比")
    print("=" * 60)
    
    # 创建测试数据
    print("\n创建测试数据...")
    data = create_test_data(n_rows=50000)
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    print(f"数据大小: {data.shape}")
    print(f"特征数: {len(features)}")
    print(f"分组数: {data['partner_name'].nunique()}")
    print(f"时间周期数: {data['apply_date'].dt.to_period('M').nunique()}")
    
    # 测试原始版本（单线程）
    print("\n" + "=" * 60)
    print("测试原始版本 (n_jobs=1)...")
    time1, count1 = benchmark_original(data, features, n_jobs=1)
    if time1:
        print(f"✓ 耗时: {time1:.2f}秒, 结果特征数: {count1}")
    
    # 测试原始版本（多线程）
    print("\n测试原始版本 (n_jobs=10)...")
    time2, count2 = benchmark_original(data, features, n_jobs=10)
    if time2:
        print(f"✓ 耗时: {time2:.2f}秒, 结果特征数: {count2}")
        if time1:
            print(f"  多线程 vs 单线程: {time2/time1:.2f}x (应该更快，但实际更慢)")
    
    # 测试优化版本
    print("\n" + "=" * 60)
    print("测试优化版本...")
    time3, count3 = benchmark_optimized(data, features)
    if time3:
        print(f"✓ 耗时: {time3:.2f}秒, 结果特征数: {count3}")
    
    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    
    if time1 and time3:
        speedup = time1 / time3
        print(f"\n优化版本 vs 原始版本 (n_jobs=1):")
        print(f"  原始: {time1:.2f}秒")
        print(f"  优化: {time3:.2f}秒")
        print(f"  加速比: {speedup:.2f}x")
    
    if time2 and time3:
        speedup = time2 / time3
        print(f"\n优化版本 vs 原始版本 (n_jobs=10):")
        print(f"  原始: {time2:.2f}秒")
        print(f"  优化: {time3:.2f}秒")
        print(f"  加速比: {speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- 原始版本使用 ThreadPoolExecutor 处理 CPU 密集操作")
    print("- GIL 导致多线程反而比单线程更慢")
    print("- 优化版本使用向量化操作，消除并行开销")
    print("- 预期优化版本快 3-5 倍")
    print("=" * 60)
