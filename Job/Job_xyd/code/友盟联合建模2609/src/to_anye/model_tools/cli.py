"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import get_module_info, get_version, print_welcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-tools",
        description="Model Tools 2.0 命令行工具",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="显示当前版本",
    )
    parser.add_argument(
        "--auto-modeling-config",
        metavar="PATH",
        help="运行配置驱动的自动建模流水线（YAML）",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="以 JSON 形式输出模块信息",
    )
    parser.add_argument(
        "--welcome",
        action="store_true",
        help="打印欢迎信息",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(get_version())
        return 0

    if args.auto_modeling_config:
        from .auto import run_auto_modeling
        result = run_auto_modeling(args.auto_modeling_config)
        print(result.html_report_path)
        return 0

    if args.info:
        print(json.dumps(get_module_info(), ensure_ascii=False, indent=2))
        return 0

    if args.welcome or not any((args.version, args.info, args.auto_modeling_config)):
        print_welcome()
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
