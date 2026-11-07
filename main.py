"""兼容入口，等价于 `python run.py`。

保留此文件是为了兼容 `uv run main.py` 等既有调用方式，
实际逻辑全部在 run.py 与 scripts/ 下。
"""
from run import main

if __name__ == "__main__":
    main()
