# 项目改为可安装发行版，训练入口收进 endfield 包

仓库原本是纯脚本平铺布局（pyproject 声明不构建、不安装为包），训练脚本 train.py 膨胀到 550+ 行后，单文件作用域承载不下配置/数据/指标/循环/产物/编排六类职责，决定按职责拆包；同时训练要能以 `uv run train` 一键启动，这要求项目提供可执行入口。

决定：

- 引入构建系统（hatchling），项目变为可安装发行版。
- 共享模块 data_utils / model / polar 收进唯一的顶级包 `endfield/`；训练管线拆为子包 `endfield/train/`，控制台入口 `[project.scripts] train`。
- prepare_data.py / predict.py / live.py 保持根级平铺脚本，从 `endfield` 导入共享模块——形成「单顶级包 + 根脚本」的混合布局。

权衡：`uv run python -m train` 可零打包达成同一调用，被否——打包路线让共享模块获得正牌命名空间，入口命令也更短。独立包名 `training/` 可避免与训练集目录同名，被否——train 一词保留给训练集划分这一正典含义（见 CONTEXT.md），包从属 `endfield` 命名空间后歧义可接受。`endfield` 与 `train` 双顶级包平级，被否——单发行版配单顶级包是标准形态，`endfield.train` 的层次自解释。

Considered options：

- 保持平铺、仅拆函数：被否，职责块已多到模块级导航比单文件内函数级导航更清晰。
- `uv run python -m train`：被否，见权衡。
- 共享模块作为顶级散件随 wheel 安装：被否，`model`、`data_utils` 这类通用名会污染 site-packages，且与根目录副本并存有漂移风险。
