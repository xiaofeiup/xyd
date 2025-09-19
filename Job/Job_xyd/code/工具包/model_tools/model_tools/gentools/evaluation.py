import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def calc_auc(y_true, y_pred):
    """计算AUC值
    
    Parameters:
    y_true: 真实标签
    y_pred: 预测概率
    
    Returns:
    float: AUC值
    """
    return roc_auc_score(y_true, y_pred)


def calculate_ks(y_true, y_prob):
    """
    计算KS值和相关统计量
    
    Parameters:
    y_true: 真实标签 (0/1)
    y_prob: 预测概率
    
    Returns:
    ks_value: KS值
    df_ks: KS统计表
    ks_index: 最大KS值对应的索引
    """
    # 创建数据框
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # 按预测概率降序排列
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    # 计算累积统计
    df['bad'] = df['y_true']
    df['good'] = 1 - df['y_true']
    
    # 总的good和bad数量
    total_good = df['good'].sum()
    total_bad = df['bad'].sum()
    
    # 累积计算
    df['cum_good'] = df['good'].cumsum()
    df['cum_bad'] = df['bad'].cumsum()
    
    # 计算累积率
    df['cum_good_rate'] = df['cum_good'] / total_good  # TPR
    df['cum_bad_rate'] = df['cum_bad'] / total_bad    # FPR
    
    # 计算KS值
    df['ks'] = df['cum_bad_rate'] - df['cum_good_rate']
    
    # 找到最大KS值
    ks_value = df['ks'].max()
    ks_index = df['ks'].idxmax()
    
    return ks_value, df, ks_index


def calculate_lift(y_true, y_prob, n_bins=10):
    """
    计算Lift值和相关统计量
    
    Parameters:
    y_true: 真实标签 (0/1)
    y_prob: 预测概率
    n_bins: 分箱数量
    
    Returns:
    df_lift_summary: Lift统计表
    df_detail: 详细数据
    baseline_rate: 基准正样本率
    """
    # 创建数据框
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_prob
    })
    
    # 按预测概率降序排列
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    # 总体统计
    total_samples = len(df)
    total_positive = df['y_true'].sum()
    baseline_rate = total_positive / total_samples  # 基准正样本率
    
    # 计算累积统计
    df['cum_positive'] = df['y_true'].cumsum()
    df['cum_samples'] = np.arange(1, len(df) + 1)
    
    # 计算累积精确率和提升度
    df['cum_precision'] = df['cum_positive'] / df['cum_samples']
    df['cum_lift'] = df['cum_precision'] / baseline_rate
    
    # 计算召回率
    df['cum_recall'] = df['cum_positive'] / total_positive
    
    # 按分位数分箱
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop') + 1
    
    # 计算分箱统计
    lift_summary = []
    for decile in sorted(df['decile'].unique()):
        bin_data = df[df['decile'] == decile]
        
        bin_stats = {
            '分箱': decile,
            '样本数': len(bin_data),
            '正样本数': bin_data['y_true'].sum(),
            '负样本数': len(bin_data) - bin_data['y_true'].sum(),
            '正样本率': bin_data['y_true'].mean(),
            'Lift': bin_data['y_true'].mean() / baseline_rate if baseline_rate > 0 else 0,
            '概率范围': f"{bin_data['y_prob'].min():.3f}-{bin_data['y_prob'].max():.3f}"
        }
        lift_summary.append(bin_stats)
    
    df_lift_summary = pd.DataFrame(lift_summary)
    
    # 计算累积Lift统计
    df_lift_summary['累积样本数'] = df_lift_summary['样本数'].cumsum()
    df_lift_summary['累积正样本数'] = df_lift_summary['正样本数'].cumsum()
    df_lift_summary['累积正样本率'] = df_lift_summary['累积正样本数'] / df_lift_summary['累积样本数']
    df_lift_summary['累积Lift'] = df_lift_summary['累积正样本率'] / baseline_rate
    df_lift_summary['累积召回率'] = df_lift_summary['累积正样本数'] / total_positive
    
    return df_lift_summary, df, baseline_rate


def calc_gain(y_true, y_pred, n_bins=10):

    """
    计算gain
    
    Parameters:
    y_true: 真实值，list
    y_pred: 预测值，list
    n_bins: 分箱数，int
    
    Returns:
    gain: gain值，Series
    
    Example:
    >>> y_true = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    >>> y_pred = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6]
    >>> gain = calc_gain(y_true, y_pred, n_bins=10)
    >>> print(gain)
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False)
    grouped = df.groupby('bin')
    total_positives = df['y_true'].sum()
    gain = grouped['y_true'].cumsum() / total_positives
    return gain 



def calculate_psi(base_list, test_list, bins=10, min_sample=10):
    """
    计算PSI
    :param base_list: 基准分布，list
    :param test_list: 测试分布，list
    :param bins: 分箱数，int
    :param min_sample: 最小样本数，int
    :return: PSI值
    
    for example:
        psi, stat_df = calculate_psi(base_list=list(df[df['draw_month'] == '2020-05'][var]), 
                                    test_list=list(df[df['draw_month'] == '2021-02'][var]), 
                                    bins=20, min_sample=10)
    """
    try:
        base_df = pd.DataFrame(base_list, columns=['score'])
        test_df = pd.DataFrame(test_list, columns=['score']) 
        
        base_notnull_cnt = len(list(base_df['score'].dropna()))
        test_notnull_cnt = len(list(test_df['score'].dropna()))
        if base_notnull_cnt == 0 or test_notnull_cnt == 0:
            return np.nan, None

        base_null_cnt = len(base_df) - base_notnull_cnt
        test_null_cnt = len(test_df) - test_notnull_cnt
        
        q_list = []
        if type(bins) == int:
            bin_num = min(bins, int(base_notnull_cnt / min_sample))
            q_list = [x / bin_num for x in range(1, bin_num)]
            break_list = []
            for q in q_list:
                bk = base_df['score'].quantile(q)
                break_list.append(bk)
            break_list = sorted(list(set(break_list))) # 去重复后排序
            score_bin_list = [-np.inf] + break_list + [np.inf]
        else:
            score_bin_list = bins
        
        base_cnt_list = [base_null_cnt]
        test_cnt_list = [test_null_cnt]
        bucket_list = ["MISSING"]
        for i in range(len(score_bin_list)-1):
            left  = round(score_bin_list[i+0], 4)
            right = round(score_bin_list[i+1], 4)
            bucket_list.append("(" + str(left) + ',' + str(right) + ']')
            
            base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0]
            base_cnt_list.append(base_cnt)
            
            test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
            test_cnt_list.append(test_cnt)
         
        stat_df = pd.DataFrame({"bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
        stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
        stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)
        
        def sub_psi(row):
            base_list = row['base_dist']
            test_dist = row['test_dist']
            # 处理某分箱内样本量为0的情况
            if base_list == 0 and test_dist == 0:
                return 0
            elif base_list == 0 and test_dist > 0:
                base_list = 1 / base_notnull_cnt   
            elif base_list > 0 and test_dist == 0:
                test_dist = 1 / test_notnull_cnt
                
            return (test_dist - base_list) * np.log(test_dist / base_list)
        
        stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
        stat_df = stat_df[['bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]
        psi = stat_df['psi'].sum()
        
    except:
        print('error!!!')
        psi = np.nan 
        stat_df = None
    return psi, stat_df
