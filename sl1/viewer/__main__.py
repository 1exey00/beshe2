#!/usr/bin/env python3
"""策略查看器模块入口点。

允许通过 `python -m viewer` 命令启动策略查看器。

使用方法:
    python -m viewer
    python -m viewer --checkpoint path/to/checkpoint.pt
    python -m viewer -c path/to/checkpoint.pt
"""

import sys
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        prog="python -m viewer",
        description="策略查看器 - Texas Hold'em AI 策略分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m viewer                            # 启动查看器
  python -m viewer -c checkpoint.pt           # 启动并加载检查点
  python -m viewer --checkpoint model.pt      # 指定检查点路径

快捷键:
  Ctrl+O      加载检查点
  Ctrl+Q      退出
  F5          刷新显示
  Ctrl+E      展开所有节点
  Ctrl+Home   重置到根节点
        """
    )
    
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="检查点文件路径（.pt 或 .pth 文件）"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="显示版本信息"
    )
    
    return parser.parse_args()


def main() -> int:
    """主入口函数。
    
    Returns:
        退出码（0表示成功，非0表示失败）
    """
    args = parse_args()
    
    # 显示版本信息
    if args.version:
        from viewer import __version__
        print(f"策略查看器 版本 {__version__}")
        return 0
    
    # 检查PyQt6依赖
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("错误: 未安装PyQt6库。", file=sys.stderr)
        print("请运行以下命令安装:", file=sys.stderr)
        print("  pip install PyQt6", file=sys.stderr)
        return 1
    
    # 验证检查点文件（如果指定）
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"错误: 检查点文件不存在: {checkpoint_path}", file=sys.stderr)
            return 1
        if not checkpoint_path.is_file():
            print(f"错误: 指定的路径不是文件: {checkpoint_path}", file=sys.stderr)
            return 1
        checkpoint_path = str(checkpoint_path.resolve())
    
    # 启动查看器
    from viewer import run_viewer
    
    try:
        return run_viewer(checkpoint_path=checkpoint_path)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
