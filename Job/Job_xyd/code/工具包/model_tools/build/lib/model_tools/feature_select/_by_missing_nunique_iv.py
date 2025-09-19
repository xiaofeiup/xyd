# 使用 scorecardpy 的 var_filter 函数，筛选缺失值、唯一值、IV

"""var_filter(dt, y, x=None, iv_limit=0.02, missing_limit=0.95,
identical_limit=0.95, var_rm=None, var_kp=None,
return_rm_reason=False, positive='bad|1')"""

from .scorecardpy.var_filter import var_filter as _sp_var_filter  # noqa: E501

__all__ = [
    "by_missing_nunique_iv",
]


def by_missing_nunique_iv(dt, y, **kwargs):
    """同 :func:`var_filter`，按缺失率、唯一值、IV 进行特征筛选。"""
    return _sp_var_filter(dt, y, **kwargs)