"""
兼容历史调用方式的通用工具入口。

示例：
    import gentools as glts
    glts.whos(globals(), top=5)
"""

from model_tools.utils.memory_tools import whos

__all__ = ["whos"]
