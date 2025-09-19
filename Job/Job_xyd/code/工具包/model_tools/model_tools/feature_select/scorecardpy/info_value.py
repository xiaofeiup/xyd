# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from .condition_fun import *
from tqdm import tqdm


def iv(dt, y, x=None, positive='bad|1', order=True, method='merge_small', min_samples=10, smooth_factor=1.0):
    '''
    Information Value
    ------
    This function calculates information value (IV) for multiple x variables. 
    It treats each unique x value as a group and counts the number of y 
    classes. If there is a zero number of y class, it will be replaced by 
    0.99 to make sure it calculable.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and 
      y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is NULL. If x is NULL, then 
      all variables except y are counted as x variables.
    positive: Value of positive class, default is "bad|1".
    order: Logical, default is TRUE. If it is TRUE, the output 
      will descending order via iv.
    
    Returns
    ------
    DataFrame
        Information Value
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # information values
    dt_info_value = sc.iv(dat, y = "creditability")
    '''
    
    dt = dt.copy(deep=True)
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str) and x is not None:
        x = [x]
    if x is not None: 
        dt = dt[y+x]
    # remove date/time col
#    dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
#    dt = rep_blank_na(dt)
    # check y
    dt = check_y(dt, y, positive)
    # x variable names
    xs = x_variable(dt, y, x)
    # info_value


    # 可以使用 joblib 并行加速 IV 计算
    from joblib import Parallel, delayed

    def _iv_single(feature):
        return iv_xy(dt[feature], dt[y[0]], method, min_samples, smooth_factor)

    iv_values = Parallel(n_jobs=-1)(
        delayed(_iv_single)(feature) for feature in tqdm(xs, desc="IV计算进度", ncols=80)
    )
    
    ivlist = pd.DataFrame({
        'variable': xs,
        'info_value': iv_values
    }, columns=['variable', 'info_value'])
    # sorting iv
    if order: 
        ivlist = ivlist.sort_values(by='info_value', ascending=False)
    return ivlist
# ivlist = iv(dat, y='creditability')

#' @import data.table
def iv_xy(x, y, method='merge_small', min_samples=10, smooth_factor=1.0):
    """
    改进的IV计算方法
    
    Parameters:
    -----------
    x : array-like
        特征值
    y : array-like  
        标签值 (0/1)
    method : str, default 'merge_small'
        处理0值的方法:
        - 'merge_small': 合并小样本组 (推荐)
        - 'skip_small': 跳过小样本组
        - 'smooth': 使用平滑因子 (不推荐，但保持向后兼容)
    min_samples : int, default 10
        最小样本量阈值
    smooth_factor : float, default 1.0
        平滑因子 (仅在method='smooth'时使用)
    """
    # good bad func
    def goodbad(df):
        names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
        return pd.Series(names)
    
    # iv calculation
    df_counts = pd.DataFrame({'x':x.astype('str'),'y':y}) \
      .fillna('missing') \
      .groupby('x') \
      .apply(goodbad)
    
    if method == 'skip_small':
        # 跳过样本量过小的组
        mask = (df_counts['good'] + df_counts['bad']) >= min_samples
        df_filtered = df_counts[mask]
        # 同时跳过任一类别为0的组
        mask_nonzero = (df_filtered['good'] > 0) & (df_filtered['bad'] > 0)
        df_final = df_filtered[mask_nonzero]
        
    elif method == 'merge_small':
        # 简化的合并方法：将小样本组合并到"其他"类别
        df_final = df_counts.copy()
        small_mask = (df_final['good'] + df_final['bad']) < min_samples
        if small_mask.any():
            # 将小样本组合并
            small_good = df_final[small_mask]['good'].sum()
            small_bad = df_final[small_mask]['bad'].sum()
            df_final = df_final[~small_mask]
            if small_good > 0 or small_bad > 0:
                # 添加合并后的组
                df_final.loc['small_groups'] = [small_good, small_bad]
        
        # 处理仍然为0的情况
        zero_mask = (df_final['good'] == 0) | (df_final['bad'] == 0)
        if zero_mask.any():
            print(f"警告: 仍有{zero_mask.sum()}个分组存在0值，将被跳过")
            df_final = df_final[~zero_mask]
            
    else:  # method == 'smooth'
        # 使用平滑因子 (保持向后兼容，但不推荐)
        print(f"警告: 使用平滑因子{smooth_factor}可能导致IV值失真")
        df_final = df_counts.copy()
        df_final['good'] = np.where(df_final['good'] == 0, smooth_factor, df_final['good'])
        df_final['bad'] = np.where(df_final['bad'] == 0, smooth_factor, df_final['bad'])
    
    if len(df_final) == 0:
        return 0.0
        
    iv_total = df_final \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv.sum()
    # return iv
    return iv_total

# print(iv_xy(x,y))


# #' Information Value
# #'
# #' calculating IV of total based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @examples
# #' # iv_01(good, bad)
# #' dtm = melt(dt, id = 'creditability')[, .(
# #' good = sum(creditability=="good"), bad = sum(creditability=="bad")
# #' ), keyby = c("variable", "value")]
# #'
# #' dtm[, .(iv = lapply(.SD, iv_01, bad)), by="variable", .SDcols# ="good"]
# #'
# #' @import data.table
#' @import data.table
#'
def iv_01(good, bad, smooth_factor=0.5):
    # iv calculation - 使用改进的平滑方法
    df = pd.DataFrame({'good':good,'bad':bad})
    
    # 使用更合理的平滑因子替换0值
    df_smoothed = df.copy()
    df_smoothed['good'] = np.where(df_smoothed['good'] == 0, smooth_factor, df_smoothed['good'])
    df_smoothed['bad'] = np.where(df_smoothed['bad'] == 0, smooth_factor, df_smoothed['bad'])
    
    iv_total = df_smoothed \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv.sum()
    # return iv
    return iv_total


# #' miv_01
# #'
# #' calculating IV of each bin based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @import data.table
# #'
#' @import data.table
#'
def miv_01(good, bad, smooth_factor=0.5):
    # iv calculation - 使用改进的平滑方法
    df = pd.DataFrame({'good':good,'bad':bad})
    
    # 使用更合理的平滑因子替换0值
    df_smoothed = df.copy()
    df_smoothed['good'] = np.where(df_smoothed['good'] == 0, smooth_factor, df_smoothed['good'])
    df_smoothed['bad'] = np.where(df_smoothed['bad'] == 0, smooth_factor, df_smoothed['bad'])
    
    infovalue = df_smoothed \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv
    # return iv
    return infovalue


# #' woe_01
# #'
# #' calculating WOE of each bin based on good and bad vectors
# #'
# #' @param good vector of good numbers
# #' @param bad vector of bad numbers
# #'
# #' @import data.table
#' @import data.table
#'
def woe_01(good, bad, smooth_factor=0.5):
    # woe calculation - 使用改进的平滑方法
    df = pd.DataFrame({'good':good,'bad':bad})
    
    # 使用更合理的平滑因子替换0值
    df_smoothed = df.copy()
    df_smoothed['good'] = np.where(df_smoothed['good'] == 0, smooth_factor, df_smoothed['good'])
    df_smoothed['bad'] = np.where(df_smoothed['bad'] == 0, smooth_factor, df_smoothed['bad'])
    
    woe = df_smoothed \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood)) \
      .woe
    # return woe
    return woe
