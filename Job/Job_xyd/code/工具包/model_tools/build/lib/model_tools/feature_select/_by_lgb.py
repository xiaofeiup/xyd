import pandas as pd
import lightgbm as lgb


def recursive_lgb_feature_selection(X, y, random_state=42, verbose=True):
    """
    递归筛选特征，直到所有特征重要性都大于0。
    返回筛选后的DataFrame和特征名列表。
    """
    X_selected = X.copy()
    iteration = 0
    while True:
        iteration += 1
        model = lgb.LGBMClassifier(random_state=random_state)
        model.fit(X_selected, y)
        feature_importances = pd.Series(model.feature_importances_, index=X_selected.columns)
        zero_importance_features = feature_importances[feature_importances == 0].index.tolist()
        if verbose:
            print(f'第{iteration}轮，特征数: {X_selected.shape[1]}，重要性为0的特征数: {len(zero_importance_features)}')
        if len(zero_importance_features) == 0:
            break
        X_selected = X_selected.drop(columns=zero_importance_features)
    return X_selected, feature_importances[feature_importances > 0].index.tolist() 