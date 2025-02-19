import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from concurrent.futures import ThreadPoolExecutor
from sc_var_filter.var_filter import var_filter
from tqdm import tqdm
pd.set_option('display.max_columns', None)


class feature_selection:
    def __init__(self, dt, y, x=None, var_rm=None, var_kp=None, 
                 return_rm_reason=False, positive='bad|1', num_process=1):
        """
        Parameters
        ----------
        dt : DataFrame
            A data.table with y and x variables
        y : str
            Target variable name
        x : list
            A list of x variables
        var_rm : list
            A list of variables that should be removed
        var_kp : list
            A list of variables that should be kept
        return_rm_reason : bool
            Whether to return the reason of removed x variable
        positive : str
            The positive value of target variable
        num_process : int
            The number of process to parallel computing
        """
        self.dt = dt
        self.y = y
        self.x = x if x is not None else [col for col in dt.columns if col != y]
        self.var_rm = var_rm
        self.var_kp = var_kp
        self.return_rm_reason = return_rm_reason
        self.positive = positive
        self.num_process = num_process
        
        # 存储筛选结果
        self.dt_sc_filtered = None
        self.rm_reason_sc = None

    def var_filter_sc(self, iv_limit=0.02, missing_limit=0.95, 
                      identical_limit=0.95,iv_binning_method='equal_width', iv_n_bins=5):
        """
        Variable Filter
        
        This function filter variables based on specified conditions, such as
        information value, missing rate, identical value rate.

        Returns
        ------
        DataFrame
            A data.table with y and selected x variables
        Dict(if return_rm_reason == TRUE)
            A DataFrame with y and selected x variables and
            a DataFrame with the reason of removed x variable.
        """
        # 执行变量筛选
        dt_sel = var_filter(self.dt, y=self.y, x=self.x, iv_limit=iv_limit, 
                            missing_limit=missing_limit, identical_limit=identical_limit, 
                            var_rm=self.var_rm, var_kp=self.var_kp, 
                            return_rm_reason=self.return_rm_reason, positive=self.positive,  iv_binning_method= iv_binning_method , iv_n_bins=iv_n_bins)
        
        self.dt_sc_filtered = dt_sel['dt'] if 'dt' in dt_sel else dt_sel
        self.rm_reason_sc = dt_sel.get('rm', None)

        if self.return_rm_reason:
            return {'dt': self.dt_sc_filtered, 'rm_reason': self.rm_reason_sc}
        else:
            return self.dt_sc_filtered

    def var_filter_permutation(self, test_size=0.3, score_type='auc', score_threshold=0.01,
                               max_keep=100, cv=5, random_state=2024, use_sc_filter=True, 
                               iv_limit=0.02, missing_limit=0.95, identical_limit=0.95,iv_binning_method='equal_width', iv_n_bins=5):
        """
        Permutation Feature Importance Selection
        
        Parameters
        ----------
        use_sc_filter : bool
            Whether to use var_filter_sc before permutation
        iv_limit : float, optional
            Information value limit for filtering
        missing_limit : float, optional
            Missing value limit for filtering
        identical_limit : float, optional
            Identical value limit for filtering
        """
        # 如果选择使用 var_filter_sc，并且未筛选过，进行筛选
        if use_sc_filter:
            self.var_filter_sc(iv_limit=iv_limit, missing_limit=missing_limit, identical_limit=identical_limit, iv_binning_method= iv_binning_method , iv_n_bins=iv_n_bins)

        dt_sel = self.dt_sc_filtered if self.dt_sc_filtered is not None else self.dt
        y = dt_sel[self.y]
        x = dt_sel.drop(columns=[self.y])
        x = x.fillna(-9999)
        # print(x.columns.to_list())

        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        cv = ShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)
        scores = defaultdict(list)

        def compute_permutation_importance(train_idx, test_idx):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # rf不支持类别型变量，所以这里需要将类别型变量进行one-hot编码
            for col in x.columns:
                if x[col].dtype == 'object':
                    x_train = pd.get_dummies(x_train, columns=[col], drop_first=True)
                    x_test = pd.get_dummies(x_test, columns=[col], drop_first=True)
            
            rf.fit(x_train, y_train)

            # 计算基线分数
            if score_type == 'auc':
                baseline_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:, 1])
            elif score_type == 'ks':
                fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])
                baseline_score = max(tpr - fpr)
            else:
                raise ValueError('score_type should be "auc" or "ks"')

            # 计算排列重要性
            def compute_score_for_feature(col):
                x_permuted = x.copy()
                x_permuted[col] = np.random.permutation(x_permuted[col])
                x_train_permuted, x_test_permuted = x_permuted.iloc[train_idx], x_permuted.iloc[test_idx]
                # rf.fit(x_train_permuted, y_train)

                if score_type == 'auc':
                    score = roc_auc_score(y_test, rf.predict_proba(x_test_permuted)[:, 1])
                elif score_type == 'ks':
                    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(x_test_permuted)[:, 1])
                    score = max(tpr - fpr)
                else:
                    raise ValueError('score_type should be "auc" or "ks"')

                # 返回重要性分数
                return (baseline_score - score) / baseline_score

            with ThreadPoolExecutor(self.num_process) as executor:
                results = list(executor.map(compute_score_for_feature, x.columns))
            for col, score in zip(x.columns, results):
                scores[col].append(score)
        for train_idx, test_idx in tqdm(cv.split(x,y),total=cv.get_n_splits()):
            compute_permutation_importance(train_idx, test_idx)

        permutation_importances = pd.Series(
            [np.mean(score) for score in scores.values()], index=scores.keys()
        ).sort_values(ascending=False)
        print("==== permutation_importances ====")
        print(permutation_importances)
        if score_threshold:
            permutation_importances = permutation_importances[permutation_importances > score_threshold]

        if max_keep:
            permutation_importances = permutation_importances[:max_keep][permutation_importances > 0]

        permutation_importances = permutation_importances.reset_index()
        permutation_importances.columns = ['variable', 'importance']

        dt_final = dt_sel[permutation_importances['variable'].tolist() + [self.y]]

        if self.return_rm_reason:
            rm_reason = self.rm_reason_sc  # 假设这里保留了原来的理由
            if rm_reason is None:
                rm_reason = pd.DataFrame({'variable': dt_sel.columns.drop(self.y)})
                rm_reason['reason'] = 'Not Selected'
            else:
                # rm_reason = rm_reason.merge(permutation_importances, left_on='variable', right_index=True, how='left')
                rm_reason = rm_reason.merge(permutation_importances, left_on='variable', right_on='variable', how='left')
                # 将rm_reason['rm_reason']为空且importance为空的变量的原因改为'shuffle_impt < score_threshold'
                rm_reason.loc[(rm_reason['rm_reason'].isnull()) & (rm_reason['importance'].isnull()), 'rm_reason'] = 'shuffle_impt < score_threshold'

            return {'dt': dt_final, 'rm_reason': rm_reason}
        else:
            return dt_final



if __name__ == '__main__':
    # test feature_selection
    from scorecardpy import germancredit

    dat = germancredit()
    print(dat.dtypes)
    # print(dat['creditability'].value_counts())
    # print(dat.describe())
    # 哪些不是数值型变量
    print(dat.dtypes[(dat.dtypes == 'object') | (dat.dtypes == 'category')])
    x = dat.columns[(dat.dtypes != 'category') & (dat.dtypes != 'object')].tolist()
    y = 'creditability'
    # 去除非数值型变量
    dat = dat[x + [y]]


    dt_sel = feature_selection(dat, y='creditability', return_rm_reason=True)
    _ = dt_sel.var_filter_permutation()
    print(_)
