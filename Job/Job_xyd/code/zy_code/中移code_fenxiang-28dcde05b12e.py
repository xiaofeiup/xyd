# 共有特征筛选
def calcu_missing_rate_grouped(df, var_list, dt_name):
    results = []
    group_sizes = df.groupby(dt_name).size()
    valid_groups = group_sizes[group_sizes >= 50].index
    mask = df[var_list].isnull() | df[var_list].isin([-9999.0,-9998.0,-9997.0,-9996.0])
    
    for var in var_list:
        grouped_average = mask[var].groupby(df[dt_name]).mean()
        display(grouped_average)
        grouped_average = grouped_average[valid_groups].reset_index(name='null_ratio')
        grouped_average['feature'] = var
        display(grouped_average)
        results.append(grouped_average)
        break
        
    final_result = pd.concat(results, ignore_index=True)
    null_feature = list(set(final_result[final_result['null_ratio'] >= 1]['feature']))
    # print(null_feature)
    final_result = final_result[~ final_result['feature'].isin(null_feature)]
    return final_result
    


dict_feature_every_data = {}
for name, f_tag in f_tag_dict.items():
    print(name, f_tag)
    if f_tag in df_web_train['f_tag'].value_counts().index.tolist():
        missing_rate_month = calcu_missing_rate_grouped(df_web_train[df_web_train['f_tag'] == f_tag], var_list,'apply_date')
    if  f_tag in df_web_oot['f_tag'].value_counts().index.tolist():
        missing_rate_month = calcu_missing_rate_grouped(df_web_oot[df_web_oot['f_tag'] == f_tag], var_list,'apply_date')
    # display(missing_rate_month)
    dict_feature_every_data[f_tag] = set(missing_rate_month['feature'].tolist())
    print(len(dict_feature_every_data[f_tag]))


feature_set_0 = set(dict_feature_every_data[0])
feature_set_1 = set(dict_feature_every_data[1])
feature_set_3 = set(dict_feature_every_data[3])
feature_set_6 = set(dict_feature_every_data[6])
feature_set_9 = set(dict_feature_every_data[9])
feature_set_10 = set(dict_feature_every_data[10])

feature_all_have = feature_set_0 & feature_set_1 & feature_set_3 & feature_set_6 & feature_set_9 & feature_set_10
len(feature_all_have)




# 特征 重要性筛选
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def feature_selection_with_lightgbm(X, y, params, threshold=0, n_iterations=10, test_size=0.2,random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state = random_state)
    selected_features = X.columns.tolist()
    
    for _ in tqdm(range(n_iterations)):
        model = lgb.LGBMClassifier(objective="binary", **params,class_weight=label_weight)
        model.fit(
            X_train[selected_features],
            y_train,
            eval_set=[(X_val[selected_features], y_val)],
            eval_metric="logloss",
            callbacks=[
                log_evaluation(50)
                # LightGBMPruningCallback(trial,'ks')
            ]
        )
        
        
        importance = model.booster_.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({'feature':selected_features, 'importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        new_selected_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature'].tolist()
        print(len(new_selected_features))
        
        if len(new_selected_features) == 0 or set(new_selected_features) == set(selected_features) :
            break
        selected_features = new_selected_features
    return selected_features, model




# 调参
def calculate_ks(y_true, y_pred_proba):
    """
    计算KS统计量

    参数:
    y_true: 真实标签
    y_pred_proba: 预测概率

    返回:
    KS统计量
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks

def objective(trial, X_train, y_train, score_type='auc',X_valid=None,y_valid=None):
    # 超参数搜索范围
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50,400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 6, 20, step=1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, step=50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.9, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "subsample":trial.suggest_float("feature_fraction", 0.5, 1, step=0.05),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1, step=0.1),
        "random_state": 2024
    }
    
    if X_valid is not None and y_valid is not None:
        model = lgb.LGBMClassifier(objective="binary", **param_grid,class_weight=label_weight)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[
                # LightGBMPruningCallback(trial, "binary_logloss"),
                log_evaluation(50)
                # LightGBMPruningCallback(trial,'binary_logloss')
            ]
        )
         # 模型预测
        preds_valid = model.predict_proba(X_valid)[: ,1]
        preds_train =  model.predict_proba(X_train)[: ,1]
        if score_type == 'auc':  # 优化目标为auc最大
            score = roc_auc_score(y_valid, preds_valid)
        elif score_type == 'ks': # 优化目标为ks最大
            score = calculate_ks(y_valid, preds_valid)
        elif score_type == 'logloss': # 优化目标为logloss最小
            score = log_loss(y_valid, preds_valid)
        elif score_type == 'train_valid_ks_diff': # 优化目标为训练集和测试集ks差值最小
            valid_ks = calculate_ks(y_valid, preds_valid)
            train_ks = calculate_ks(y_train,preds_train)
            print("valid_ks:",valid_ks)
            print("train_ks:",train_ks)
            score = abs(valid_ks - train_ks)
        else:
            raise ValueError('score_type should be "auc" or "ks"')
                    
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return score

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

    cv_scores = np.empty(5)
    for idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # LGBM建模
        model = lgb.LGBMClassifier(objective="binary", **param_grid,class_weight=label_weight)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[
                # LightGBMPruningCallback(trial, "binary_logloss"),
                log_evaluation(50)
                # LightGBMPruningCallback(trial,'binary_logloss')
            ]
        )
        # 模型预测
        preds_valid = model.predict_proba(X_valid)[: ,1]
        preds_train =  model.predict_proba(X_train)[: ,1]
        if score_type == 'auc':  # 优化目标为auc最大
            score = roc_auc_score(y_valid, preds_valid)
        elif score_type == 'ks': # 优化目标为ks最大
            score = calculate_ks(y_valid, preds_valid)
        elif score_type == 'logloss': # 优化目标为logloss最小
            score = log_loss(y_valid, preds_valid)
        elif score_type == 'train_valid_ks_diff': # 优化目标为训练集和测试集ks差值最小
            valid_ks = calculate_ks(y_valid, preds_valid)
            train_ks = calculate_ks(y_train,preds_train)
            if valid_ks == train_ks == 0 or valid_ks < 0.15:
                continue
            print("valid_ks:",valid_ks)
            print("train_ks:",train_ks)
            score = abs(valid_ks - train_ks)
        else:
            raise ValueError('score_type should be "auc" or "ks"')
            
        trial.report(score, idx)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        cv_scores[idx] = score
    return np.mean(cv_scores)


start_time = time.time()
study = optuna.create_study(direction='minimize')
func = lambda trial: objective(trial,X_train ,y_train , score_type='train_valid_ks_diff',X_valid=X_test, y_valid=y_test)
study.optimize(func, n_trials=30)

# 打印最优参数
end_time = time.time()
print('耗时(m):',(end_time - start_time) /60 )
print(f'Best trial: {study.best_trial.params}')
# 打印最优目标值
print(f'Best value: {study.best_value}')
