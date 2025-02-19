# -*- coding: utf-8 -*-

from threading import main_thread
import numpy as np
import pandas as pd
import warnings
import time

# from torch import P
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# from sc_var_filter.condition_fun import *
# from sc_var_filter.info_value import *
from .condition_fun import *
from .info_value import *


def var_filter(dt, y, x=None, iv_limit=0.02, missing_limit=0.95, 
               identical_limit=0.95, var_rm=None, var_kp=None, 
               return_rm_reason=False, positive='bad|1', iv_binning_method=None, iv_n_bins=5):
    '''
    Variable Filter
    ------
    This function filter variables base on specified conditions, such as 
    information value, missing rate, identical value rate.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and y 
      (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is NULL. If x is NULL, then all 
      variables except y are counted as x variables.
    iv_limit: The information value of kept variables should>=iv_limit. 
      The default is 0.02.
    missing_limit: The missing rate of kept variables should<=missing_limit. 
      The default is 0.95.
    identical_limit: The identical value rate (excluding NAs) of kept 
      variables should <= identical_limit. The default is 0.95.
    var_rm: Name of force removed variables, default is NULL.
    var_kp: Name of force kept variables, default is NULL.
    return_rm_reason: Logical, default is FALSE.
    positive: Value of positive class, default is "bad|1".
    
    Returns
    ------
    DataFrame
        A data.table with y and selected x variables
    Dict(if return_rm_reason == TRUE)
        A DataFrame with y and selected x variables and 
          a DataFrame with the reason of removed x variable.
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # variable filter
    dt_sel = sc.var_filter(dat, y = "creditability")
    '''
    # start time
    start_time = time.time()
    print('[INFO] filtering variables ...')
    
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
    x = x_variable(dt,y,x)
    
    # force removed variables
    if var_rm is not None: 
        if isinstance(var_rm, str):
            var_rm = [var_rm]
        x = list(set(x).difference(set(var_rm)))
    # check force kept variables
    if var_kp is not None:
        if isinstance(var_kp, str):
            var_kp = [var_kp]
        var_kp2 = list(set(var_kp) & set(x))
        len_diff_var_kp = len(var_kp) - len(var_kp2)
        if len_diff_var_kp > 0:
            warnings.warn("Incorrect inputs; there are {} var_kp variables are not exist in input data, which are removed from var_kp. \n {}".format(len_diff_var_kp, list(set(var_kp)-set(var_kp2))) )
        var_kp = var_kp2 if len(var_kp2) > 0 else None
  
    ################### 这是原来的代码 ###################
    """
    # -iv
    iv_list = iv(dt, y, x, order=False)
    # -na percentage
    nan_rate = lambda a: a[a.isnull()].size/a.size
    na_perc = dt[x].apply(nan_rate).reset_index(name='missing_rate').rename(columns={'index':'variable'})
    # -identical percentage
    idt_rate = lambda a: a.value_counts().max() / a.size
    identical_perc = dt[x].apply(idt_rate).reset_index(name='identical_rate').rename(columns={'index':'variable'})
    
    # dataframe iv na idt
    dt_var_selector = iv_list.merge(na_perc,on='variable').merge(identical_perc,on='variable')
    # remove na_perc>95 | ele_perc>0.95 | iv<0.02
    # variable datatable selected
    dt_var_sel = dt_var_selector.query('(info_value >= {}) & (missing_rate <= {}) & (identical_rate <= {})'.format(iv_limit,missing_limit,identical_limit))  
    
    # add kept variable
    x_selected = dt_var_sel.variable.tolist()
    if var_kp is not None: 
        x_selected = np.unique(x_selected+var_kp).tolist()
    # data kept
    dt_kp = dt[x_selected+y]

    # runingtime
    runingtime = time.time() - start_time
    if (runingtime >= 10):
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Variable filtering on {} rows and {} columns in {} \n{} variables are removed'.format(dt.shape[0], dt.shape[1], time.strftime("%H:%M:%S", time.gmtime(runingtime)), dt.shape[1]-len(x_selected+y)))
    # return remove reason
    if return_rm_reason:
        dt_var_rm = dt_var_selector.query('(info_value < {}) | (missing_rate > {}) | (identical_rate > {})'.format(iv_limit,missing_limit,identical_limit)) \
          .assign(
            info_value = lambda x: ['info_value<{}'.format(iv_limit) if i else np.nan for i in (x.info_value < iv_limit)], 
            missing_rate = lambda x: ['missing_rate>{}'.format(missing_limit) if i else np.nan for i in (x.missing_rate > missing_limit)],
            identical_rate = lambda x: ['identical_rate>{}'.format(identical_limit) if i else np.nan for i in (x.identical_rate > identical_limit)]
          )
        dt_rm_reason = pd.melt(dt_var_rm, id_vars=['variable'], var_name='iv_mr_ir').dropna()\
        .groupby('variable').apply(lambda x: ', '.join(x.value)).reset_index(name='rm_reason')
        
        if var_rm is not None: 
            dt_rm_reason = pd.concat([
              dt_rm_reason, 
              pd.DataFrame({'variable':var_rm,'rm_reason':"force remove"}, columns=['variable', 'rm_reason'])
            ])
        if var_kp is not None:
            dt_rm_reason = dt_rm_reason.query('variable not in {}'.format(var_kp))
        
        dt_rm_reason = pd.merge(dt_rm_reason, dt_var_selector, how='outer', on = 'variable')
        return {'dt': dt_kp, 'rm':dt_rm_reason}
    else:
      return dt_kp
    """
    ##########################################################

    ##################### 这是修改的地方 #######################
    # 串行计算，na_perc, identical_perc，iv，减少计算iv的特征数
    # -na percentage
    print('[INFO] calculating missing rate ...')

    def nan_rate(a):
        return a[a.isnull()].size/a.size

    na_perc = dt[x].apply(nan_rate).reset_index(name='missing_rate').rename(columns={'index':'variable'})
    # filter by missing rate
    dt_var_selector = na_perc.query('missing_rate <= {}'.format(missing_limit))
    dt_rm_reason_na = na_perc.query('missing_rate > {}'.format(missing_limit))
    x = dt_var_selector['variable'].tolist()
    print('[INFO] end missing rate calculation')
    print('[INFO] length of x:', len(x))
    print("=" * 50)

    # -identical percentage
    print('[INFO] calculating identical rate ...')

    def idt_rate(a):
        return a.value_counts().max() / a.size
    identical_perc = dt[x].apply(idt_rate).reset_index(name='identical_rate').rename(columns={'index':'variable'})
    # filter by identical rate
    dt_var_selector = dt_var_selector.merge(identical_perc, on='variable')
    dt_rm_reason_idt = dt_var_selector.query('identical_rate > {}'.format(identical_limit))
    dt_var_selector = dt_var_selector.query('identical_rate <= {}'.format(identical_limit))
    x = dt_var_selector['variable'].tolist()
    print('after nan and idt selected features')
    print('[INFO] end identical rate calculation')
    print('[INFO] length of x:', len(x))
    print("=" * 50)
    
    # -iv
    print('[INFO] calculating information value ...')
    iv_list = iv(dt, y, x, order=False, binning_method=iv_binning_method, n_bins=iv_n_bins)
    # filter by iv
    dt_var_selector = dt_var_selector.merge(iv_list, on='variable')
    dt_rm_reason_iv = dt_var_selector.query('info_value < {}'.format(iv_limit))
    dt_var_sel = dt_var_selector.query('info_value >= {}'.format(iv_limit))
    x = dt_var_sel['variable'].tolist()
    print('[INFO] end information value calculation')
    print('[INFO] length of x:', len(x))
    print("=" * 50)
    
    # add kept variable
    x_selected = dt_var_sel.variable.tolist()
    if var_kp is not None: 
        x_selected = np.unique(x_selected + var_kp).tolist()
    # data kept
    dt_kp = dt[x_selected + y]
    
    # runingtime
    runingtime = time.time() - start_time
    if (runingtime >= 10):
        print('Variable filtering on {} rows and {} columns in {} \n{} variables are removed'.format(dt.shape[0], dt.shape[1], time.strftime("%H:%M:%S", time.gmtime(runingtime)), dt.shape[1]-len(x_selected+y)))
    
    # return remove reason
    if return_rm_reason:
        dt_rm_reason = pd.concat([dt_rm_reason_na, dt_rm_reason_idt, dt_rm_reason_iv])
        dt_rm_reason = dt_rm_reason.assign(
            info_value = lambda x: ['info_value<{}'.format(iv_limit) if i else np.nan for i in (x.info_value < iv_limit)], 
            missing_rate = lambda x: ['missing_rate>{}'.format(missing_limit) if i else np.nan for i in (x.missing_rate > missing_limit)],
            identical_rate = lambda x: ['identical_rate>{}'.format(identical_limit) if i else np.nan for i in (x.identical_rate > identical_limit)]
        )
        dt_rm_reason = pd.melt(dt_rm_reason, id_vars=['variable'], var_name='iv_mr_ir').dropna()\
            .groupby('variable').apply(lambda x: ', '.join(x.value)).reset_index(name='rm_reason')
        
        if var_rm is not None: 
            dt_rm_reason = dt_rm_reason.append(pd.DataFrame({'variable': var_rm, 'rm_reason': ['force removed']*len(var_rm)}))
        if var_kp is not None:
            dt_rm_reason = dt_rm_reason[~dt_rm_reason['variable'].isin(var_kp)]
        
        dt_rm_reason = pd.merge(dt_rm_reason, dt_var_selector, how='outer', on='variable')
        return {'dt': dt_kp, 'rm': dt_rm_reason}
    else:
        return dt_kp
    ##########################################################


if __name__ == '__main__':

    # test var_filter
    from scorecardpy import germancredit
    dat = germancredit()
    print(dat['creditability'].value_counts())
    dt_sel = var_filter(dat, y="creditability",iv_binning_method='equal_width')
    print(dt_sel.shape)

