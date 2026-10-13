"""CLI 编排面：把库侧 `endfield` 的 module 拼成可执行命令。

只保留训练与导出 ONNX 两个入口：`train` 读 `data/` 下的 raw 名单划分与 C++ 预处理
条带，`export-onnx` 把 run 目录的 checkpoint 落成交付图。`endfield/` 保持为纯库，
import 它不会带出 argparse 与子进程副作用。
"""
