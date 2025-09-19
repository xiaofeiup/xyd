import pandas as pd


def balance_sampling_by_category(df, cat_col, target_col):
    """
    按照类别列对数据进行采样，使每个类别的目标变量分布接近总体分布
    
    参数:
    df: DataFrame, 输入数据
    cat_col: str, 类别列名称
    target_col: str, 目标变量列名称
    
    返回:
    DataFrame: 采样后的数据
    """
    # 计算总体目标变量分布
    label_percent_all = df[target_col].value_counts(normalize=True)
    
    # 初始化结果DataFrame
    sampled_data = pd.DataFrame()
    
    # 对每个类别进行采样
    for category in df[cat_col].unique():
        category_data = df[df[cat_col] == category]
        category_label_dist = category_data[target_col].value_counts(normalize=True)
        
        # 如果该类别的label=0占比大于总体label=0占比,需要减少label=0的样本
        if category_label_dist[0.0] > label_percent_all[0.0]:
            # 计算需要保留的label=0样本数量
            n_zeros = int(category_data[category_data[target_col]==1].shape[0] * 
                         label_percent_all[0.0]/label_percent_all[1.0])
            # 随机采样label=0的样本
            zeros_sample = category_data[category_data[target_col]==0].sample(n=n_zeros, random_state=42)
            ones_sample = category_data[category_data[target_col]==1]
            category_sampled = pd.concat([zeros_sample, ones_sample])
        else:
            # 如果label=1占比过高,增加label=0的样本数量
            n_ones = int(category_data[category_data[target_col]==0].shape[0] * 
                        label_percent_all[1.0]/label_percent_all[0.0])
            zeros_sample = category_data[category_data[target_col]==0]
            ones_sample = category_data[category_data[target_col]==1].sample(n=n_ones, random_state=42)
            category_sampled = pd.concat([zeros_sample, ones_sample])
            
        sampled_data = pd.concat([sampled_data, category_sampled])

    # 重置索引
    return sampled_data.reset_index(drop=True)