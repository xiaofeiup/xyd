import pandas as pd
import numpy as np
from sklearn import metrics

def numerical_univerate(df, var_name, target, labels=None, bins=10, lamb=0.001, max_categories=30):
    """
    计算特征的IV和lift值，并将空值单独分为一箱

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

    # 创建一个副本避免修改原始数据
    df_temp = df.copy()

    # 处理空值 - 将空值单独分为一箱
    missing_mask = df_temp[var_name].isnull()

    if df_temp[var_name].dtype.name == 'category' or df_temp[var_name].dtype.name == 'object' or df_temp[
            var_name].dtype.name == "dtype('O')":
        # 处理分类特征
        # 如果是category类型，先转换为object类型以避免类别限制
        if df_temp[var_name].dtype.name == 'category':
            df_temp[var_name] = df_temp[var_name].astype('object')

        # 先处理非空值
        non_missing_data = df_temp[~missing_mask]
        if len(non_missing_data) > 0:
            value_counts = non_missing_data[var_name].value_counts()
            top_categories = value_counts.head(max_categories).index
            df_temp.loc[~missing_mask, new_col] = df_temp.loc[~missing_mask, var_name].apply(
                lambda x: x if x in top_categories else 'Other')

        # 处理空值
        if missing_mask.any():
            df_temp.loc[missing_mask, new_col] = 'Missing'
    else:
        # 处理连续型特征
        # 先处理非空值
        non_missing_data = df_temp[~missing_mask]
        if len(non_missing_data) > 0:
            if labels is None:
                # 使用 pd.qcut 进行等频分箱，转为字符串避免分类类型问题
                bins_result = pd.qcut(non_missing_data[var_name], bins, duplicates='drop')
                df_temp.loc[~missing_mask, new_col] = bins_result.astype(str)
            else:
                # 使用用户定义的标签，转为字符串
                bins_result = pd.cut(non_missing_data[var_name], labels)
                df_temp.loc[~missing_mask, new_col] = bins_result.astype(str)

        # 处理空值
        if missing_mask.any():
            df_temp.loc[missing_mask, new_col] = 'Missing'

    # 如果所有值都是空值，则全部标记为Missing
    if df_temp[new_col].isnull().all():
        df_temp[new_col] = 'Missing'

    dti = pd.crosstab(df_temp[new_col], df_temp[target]).sort_values(by=new_col, ascending=False)

    dti.rename(
        {1: "positive", 0: "negative"},
        axis=1,
        inplace=True,
    )

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
        df_temp[new_col] = df_temp[new_col].astype('category')
        max_codes = df_temp[new_col].cat.codes.max()

        codes_list = sorted([n] + list(range(max_codes, n, -1)))
        df_new = df_temp[df_temp[new_col].cat.codes.isin(codes_list)]
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
