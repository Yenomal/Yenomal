# TODO

## 当前优先事项

1. 配置 `src` 的 `pyproject.toml`
- 明确 `src` 的包名与安装方式
- 通过 `pip install -e .` 提供统一 import

- [x]

2. 统一 `scripts` 的入口规范
- 每个入口脚本自行解析 `ROOT`
- 使用 `pushd/popd` 切换到对应 `third_party` repo 运行
- 统一管理 `datasets`、`outputs`、`checkpoints` 路径

- [x]

3. 清理 `third_party` 的历史路径依赖
- 减少对 Yenomal 根目录的反推
- 清理不必要的 `Path(__file__)` 多层回溯
- 优先通过 shell 传参而不是在 Python 内猜路径

- [x]

4. 清理旧结构遗留
- 移除 `src/projects` 相关假设
- 清理写死的绝对路径
- 减少全局 `PYTHONPATH` 依赖

- [x]

5. 同步文档
- 更新当前 monorepo 结构说明
- 补充 Python 脚本与 bash 脚本的路径规范

- []

6. 补充最小复现检查
- 确认入口脚本、工作目录切换、路径解析可以正常工作

- []