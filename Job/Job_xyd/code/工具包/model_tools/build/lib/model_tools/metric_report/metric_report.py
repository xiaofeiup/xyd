import pandas as pd
import numpy as np
from sklearn import metrics

# 从gentools模块导入评估函数
from ..gentools import calc_auc, calculate_ks

# ---------------------- Matplotlib 中文字体支持 -------------------------
# 指定常见中文字体，按顺序查找系统中已安装的字体名称。
# 若首选字体不存在，matplotlib 会退回到后续字体。有必要时用户应自行安装或调整这里的字体列表。


def numerical_univerate(df, var_name, target, labels=None, bins=10, lamb=0.001, max_categories=30):
    """

    :param df:  dataframe
    :param var_name:  特征名称
    :param target:  目标变量
    :param labels:  分箱标签
    :param bins:   分箱数
    :param lamb:  平滑系数
    :param max_categories: 最大类别数
    :return:  dataframe
    """
    new_col = var_name + "_bin"
    if df[var_name].dtype.name == 'category' or df[var_name].dtype.name == 'object' or df[
            var_name].dtype.name == "dtype('O')":
        # 处理分类特征
        value_counts = df[var_name].value_counts()
        top_categories = value_counts.head(max_categories).index
        df[new_col] = df[var_name].apply(lambda x: x if x in top_categories else 'Other')
    else:
        # 处理连续型特征
        if labels is None:
            df[new_col] = pd.qcut(df[var_name], bins, duplicates='drop')
        else:
            df[new_col] = pd.cut(df[var_name], labels)

    dti = pd.crosstab(df[new_col], df[target]).sort_values(by=new_col, ascending=False)
    # print(dti)
    dti.rename(
        {1: "positive", 0: "negative"},
        axis=1,
        inplace=True,
    )
    # spearman_corr = df[var_name].corr(df[target],method='spearman')
    dti["positive"] = dti["positive"].astype(int)
    dti["negative"] = dti["negative"].astype(int)
    p_t = dti["positive"].sum()
    n_t = dti["negative"].sum()
    t_t = p_t + n_t
    r_t = p_t / t_t
    dti["total"] = dti["positive"] + dti["negative"]
    dti["total_rate"] = dti["total"] / t_t
    dti["positive_rate"] = dti["positive"] / dti["total"]

    dti["negative_cum"] = dti["negative"].cumsum()
    dti["positive_cum"] = dti["positive"].cumsum()

    dti['positive_rate_cum'] = dti["positive_cum"] / (dti["negative_cum"] + dti["positive_cum"])

    dti["woe"] = np.log(
        ((dti["negative"] / n_t) + lamb) / ((dti["positive"] / p_t) + lamb)
    )

    dti["LIFT"] = dti["positive_rate"] / r_t
    dti["KS"] = np.abs((dti["positive_cum"] / p_t) - (dti["negative_cum"] / n_t))
    dti["IV"] = (dti["negative"] / n_t - dti["positive"] / p_t) * dti["woe"]

    IV = dti["IV"].sum()
    dti['IV'] = IV

    dti = dti.reset_index()
    dti.columns.name = None

    def _cum_calc_auc(n):
        df[new_col] = df[new_col].astype('category')
        max_codes = df[new_col].cat.codes.max()

        codes_list = sorted([n] + list(range(max_codes, n, -1)))
        df_new = df[df[new_col].cat.codes.isin(codes_list)]
        df_new[var_name] = df_new[var_name].astype('category').cat.codes
        unique_targets = df_new[target].unique()
        if len(unique_targets) == 1:
            auc = np.nan
        else:
            auc = metrics.roc_auc_score(df_new[target], df_new[var_name])
            auc = auc if auc > 0.5 else 1 - auc
        return auc

    dti[new_col] = dti[new_col].astype('category')
    dti['auc_cum'] = dti[new_col].cat.codes.map(_cum_calc_auc)

    dti.rename({new_col: 'bin'}, axis=1, inplace=True)
    dti.insert(0, "target", [target] * dti.shape[0])
    dti.insert(0, "var", [var_name] * dti.shape[0])
    dti.drop(columns=["negative_cum", "positive_cum"], inplace=True)
    return dti


def generate_metric_table(y_true_dict, y_pred_dict):
    """
    生成模型评估表，包含样本总计、0/1样本数、占比、AUC、KS等。
    返回pandas.DataFrame。
    :param y_true_dict: 真实值，dict
    :param y_pred_dict: 预测值，dict
    :param sample_name_dict: 样本名称，dict
    :return: pandas.DataFrame
    for example:
        df = generate_metric_table(y_true_dict={'train': y_true_train, 'test': y_true_test, 'oot': y_true_oot}, 
                     y_pred_dict={'train': y_pred_train, 'test': y_pred_test, 'oot': y_pred_oot})
    """
    rows = []
    for key in y_true_dict:
        y_true = pd.Series(y_true_dict[key])
        y_pred = pd.Series(y_pred_dict[key])
        total = len(y_true)
        n_0 = (y_true == 0).sum()
        n_1 = (y_true == 1).sum()
        pct_0 = f"{n_0 / total * 100:.2f}%"
        pct_1 = f"{n_1 / total * 100:.2f}%"
        auc = calc_auc(y_true, y_pred)
        ks, _, _ = calculate_ks(y_true, y_pred)
        row = [
            key, total, n_0, n_1, pct_0, pct_1, f"{auc:.4f}", f"{ks:.4f}"
        ]
        rows.append(row)
    columns = [
        "数据类别", "样本总计", "0-样本计数", "1-样本计数", "0-样本占比", "1-样本占比", "AUC值", "KS值"
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df


def analyze_lift_performance(df_lift_summary, baseline_rate):
    """
    分析Lift性能表现
    """
    print("=" * 60)
    print("Lift分析详细报告")
    print("=" * 60)
    
    # 基本统计
    max_lift = df_lift_summary['Lift'].max()
    max_lift_decile = df_lift_summary.loc[df_lift_summary['Lift'].idxmax(), '分箱']
    avg_lift = df_lift_summary['Lift'].mean()
    
    print(f"基准正样本率: {baseline_rate:.4f}")
    print(f"最大Lift值: {max_lift:.2f} (第{max_lift_decile}分箱)")
    print(f"平均Lift值: {avg_lift:.2f}")
    
    # Top 10%分析
    top10_lift = df_lift_summary.iloc[0]['Lift']
    top10_precision = df_lift_summary.iloc[0]['正样本率']
    top10_recall = df_lift_summary.iloc[0]['累积召回率']
    
    print("\nTop 10%表现:")
    print(f"  Lift值: {top10_lift:.2f}")
    print(f"  精确率: {top10_precision:.4f}")
    print(f"  召回率: {top10_recall:.4f}")
    
    # Top 20%分析
    if len(df_lift_summary) >= 2:
        top20_lift = df_lift_summary.iloc[1]['累积Lift']
        top20_precision = df_lift_summary.iloc[1]['累积正样本率']
        top20_recall = df_lift_summary.iloc[1]['累积召回率']
        
        print("\nTop 20%表现:")
        print(f"  累积Lift值: {top20_lift:.2f}")
        print(f"  累积精确率: {top20_precision:.4f}")
        print(f"  累积召回率: {top20_recall:.4f}")
    
    # 性能评价
    print("\n模型性能评价:")
    if top10_lift >= 3:
        评价 = "优秀"
    elif top10_lift >= 2:
        评价 = "良好"
    elif top10_lift >= 1.5:
        评价 = "一般"
    else:
        评价 = "较差"
    
    print(f"  整体评价: {评价}")
    
    # 业务建议
    print("\n业务建议:")
    if top10_lift >= 2:
        print(f"  - 建议重点关注前{int(100 / len(df_lift_summary))}%的高分客户")
        print(f"  - 该部分客户的转化率是平均水平的{top10_lift:.1f}倍")
    else:
        print("  - 模型区分度有限，建议优化特征工程或尝试其他算法")
    
    print("=" * 60)
    
    # 显示详细表格
    print("\nLift分析详细表格:")
    print(df_lift_summary.round(4))


