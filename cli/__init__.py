"""CLI 编排面：把库侧（`endfield` / `placement`）的 module 拼成可执行命令。

编排天然横跨两个包（数据生成既消费采样又写缓存戳，交付导出同时用结构断言与模型导出），
因此集中在这一层；`endfield/` 与 `placement/` 保持为纯库，import 它们不会带出 argparse
与子进程副作用（ADR 0004）。
"""
