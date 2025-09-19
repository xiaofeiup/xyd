#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的iv_xy函数
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加模块路径
sys.path.append('model_tools')

from model_tools.feature_select.scorecardpy.info_value import iv_xy

def create_test_data():
    """
    创建测试数据集
    """
    np.random.seed(42)
    
    # 测试案例1：正常情况 - 所有分组都有好坏样本
    print("="*60)
    print("测试案例1：正常情况")
    print("="*60)
    
    x1 = ['A'] * 100 + ['B'] * 200 + ['C'] * 150 + ['D'] * 80
    y1 = [0] * 80 + [1] * 20 + [0] * 150 + [1] * 50 + [0] * 110 + [1] * 40 + [0] * 50 + [1] * 30
    
    print("数据分布:")
    df1 = pd.DataFrame({'x': x1, 'y': y1})
    print(df1.groupby('x')['y'].agg(['count', 'sum', lambda x: (x==0).sum()]))
    print("count=总样本, sum=坏样本, <lambda>=好样本")
    
    # 测试案例2：问题情况 - 某个分组坏样本为0
    print("\n" + "="*60)
    print("测试案例2：问题情况 - B组坏样本为0")
    print("="*60)
    
    x2 = ['A'] * 100 + ['B'] * 200 + ['C'] * 150 + ['D'] * 80
    y2 = [0] * 80 + [1] * 20 + [0] * 200 + [0] * 110 + [1] * 40 + [0] * 50 + [1] * 30  # B组全为好样本
    
    print("数据分布:")
    df2 = pd.DataFrame({'x': x2, 'y': y2})
    print(df2.groupby('x')['y'].agg(['count', 'sum', lambda x: (x==0).sum()]))
    
    # 测试案例3：极端情况 - 多个分组有0值
    print("\n" + "="*60)
    print("测试案例3：极端情况 - 多个分组有0值")
    print("="*60)
    
    x3 = ['A'] * 100 + ['B'] * 50 + ['C'] * 30 + ['D'] * 20 + ['E'] * 10
    y3 = [0] * 80 + [1] * 20 + [0] * 50 + [1] * 0 + [0] * 0 + [1] * 30 + [0] * 20 + [1] * 0 + [0] * 10 + [1] * 0  # B,D,E组有0值
    
    print("数据分布:")
    df3 = pd.DataFrame({'x': x3, 'y': y3})
    print(df3.groupby('x')['y'].agg(['count', 'sum', lambda x: (x==0).sum()]))
    
    return (x1, y1), (x2, y2), (x3, y3)

def test_iv_methods(x, y, case_name):
    """
    测试不同的IV计算方法
    """
    print(f"\n{case_name} - IV计算结果对比:")
    print("-" * 50)
    
    # 转换为numpy array
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    try:
        # 方法1: 合并小样本组 (推荐)
        iv_merge = iv_xy(x_arr, y_arr, method='merge_small', min_samples=10)
        print(f"合并小样本组法:     {iv_merge:.4f}")
    except Exception as e:
        print(f"合并小样本组法:     错误 - {e}")
    
    try:
        # 方法2: 跳过小样本组
        iv_skip = iv_xy(x_arr, y_arr, method='skip_small', min_samples=10)
        print(f"跳过小样本组法:     {iv_skip:.4f}")
    except Exception as e:
        print(f"跳过小样本组法:     错误 - {e}")
    
    try:
        # 方法3: 平滑因子 0.9 (原始方法)
        print("警告: 使用平滑因子0.9可能导致IV值失真")
        iv_smooth_09 = iv_xy(x_arr, y_arr, method='smooth', smooth_factor=0.9)
        print(f"平滑因子0.9 (原始): {iv_smooth_09:.4f}")
    except Exception as e:
        print(f"平滑因子0.9 (原始): 错误 - {e}")
    
    try:
        # 方法4: 平滑因子 1.0 (保守)
        print("警告: 使用平滑因子1.0可能导致IV值失真")
        iv_smooth_10 = iv_xy(x_arr, y_arr, method='smooth', smooth_factor=1.0)
        print(f"平滑因子1.0 (保守): {iv_smooth_10:.4f}")
    except Exception as e:
        print(f"平滑因子1.0 (保守): 错误 - {e}")
    
    try:
        # 方法5: 平滑因子 0.5 (激进)
        print("警告: 使用平滑因子0.5可能导致IV值失真")
        iv_smooth_05 = iv_xy(x_arr, y_arr, method='smooth', smooth_factor=0.5)
        print(f"平滑因子0.5 (激进): {iv_smooth_05:.4f}")
    except Exception as e:
        print(f"平滑因子0.5 (激进): 错误 - {e}")

def interpret_iv_values():
    """
    解释IV值的含义
    """
    print("\n" + "="*60)
    print("IV值解释标准")
    print("="*60)
    print("< 0.02  : 无预测力")
    print("0.02-0.1: 弱预测力") 
    print("0.1-0.3 : 中等预测力")
    print("0.3-0.5 : 强预测力")
    print("> 0.5   : 过强预测力 (可能过拟合)")
    print("> 1.0   : 异常高 (通常表示计算错误)")

def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "="*60)
    print("边界情况测试")
    print("="*60)
    
    # 边界情况1: 单一分组
    print("\n1. 单一分组:")
    x_single = ['A'] * 100
    y_single = [0] * 60 + [1] * 40
    test_iv_methods(x_single, y_single, "单一分组")
    
    # 边界情况2: 完全分离 (某个分组全是好样本或坏样本)
    print("\n2. 完全分离:")
    x_sep = ['A'] * 50 + ['B'] * 50
    y_sep = [0] * 50 + [1] * 50  # A组全好，B组全坏
    test_iv_methods(x_sep, y_sep, "完全分离")
    
    # 边界情况3: 包含缺失值
    print("\n3. 包含缺失值:")
    x_missing = ['A'] * 50 + ['B'] * 30 + [None] * 20
    y_missing = [0] * 30 + [1] * 20 + [0] * 20 + [1] * 10 + [0] * 10 + [1] * 10
    test_iv_methods(x_missing, y_missing, "包含缺失值")

def test_custom_case(x, y, case_name="自定义测试"):
    """
    测试自定义数据
    """
    print(f"\n" + "="*60)
    print(f"{case_name}")
    print("="*60)
    
    # 显示数据分布
    df = pd.DataFrame({'x': x, 'y': y})
    print("数据分布:")
    print(df.groupby('x')['y'].agg(['count', 'sum', lambda x: (x==0).sum()]))
    print("count=总样本, sum=坏样本, <lambda>=好样本")
    
    # 测试各种方法
    test_iv_methods(x, y, case_name)

def main():
    """
    主测试函数
    """
    print("测试改进后的iv_xy函数")
    print("="*60)
    
    # 创建测试数据
    (x1, y1), (x2, y2), (x3, y3) = create_test_data()
    
    # 测试各种情况
    test_iv_methods(x1, y1, "正常情况")
    test_iv_methods(x2, y2, "问题情况")  
    test_iv_methods(x3, y3, "极端情况")
    
    # 测试边界情况
    test_edge_cases()
    
    # 解释IV值
    interpret_iv_values()
    
    print(f"\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("1. 推荐使用 method='merge_small' (默认)")
    print("2. 避免使用平滑因子，特别是小于1.0的值")
    print("3. 当IV > 1.0时需要检查数据质量")
    print("4. 合并或跳过小样本组是更好的策略")
    print("\n使用说明:")
    print("- 可以调用 test_custom_case(x, y, '描述') 测试自定义数据")
    print("- 可以直接调用 iv_xy(x, y, method='merge_small') 计算IV值")

if __name__ == "__main__":
    main()