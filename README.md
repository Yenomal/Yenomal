# Yenorepo

这是一个个人的 Monorepo 架构，我给他取了一个独立的IP**Yenorepo**，用于存档平时实验过程中的各种代码

## 目录



## 总述

Yenorepo的主要目的是提供一个个人的科研工作站，其包含以下一些内容：

``` text
- datasets：存放原始数据、数据的相关产物、原始权重

- docs：存放项目进展过程中的各种内容，具体参考[docs要求.md](./docs/architecture/docs要求.md)

- outputs：存放各种实验的输出内容，包括其权重、log等

- scripts：存放各种脚本，包括实验的入口脚本（train，eval，manifest）

- src：存放在实验过程中可以沉淀下来，经常进行复用的一些封装好的模型、函数等

- third_party：第三方repo，主要的工作场所
```
文件具体结构参考[文件结构.md](./docs/architecture/文件结构.md)

## 文件结构

``` text
Yenomal/
├── third_party/
│   ├── AutoGaze/
│   ├── RMBench/
│   ├── SimVLA/
│   ├── hdf5-to-lerobot-converter/
│   └── openpi/
├── datasets/
│   ├── raw/            原始数据
│   ├── product/        经过处理后的数据
│   ├── checkpoints/    下载下来的 pre-train 权重
│   └── lerobot/        转换后的 LeRobot 数据
├── src/
│   ├── pyproject.toml
│   └── yenomal/        经过沉淀并可安装的个人 Python 包
│       ├── nn/
│       └── vision/
├── scripts/
│   ├── data/           数据制作、转换、产品构建入口
│   └── projects/       不同 project / experiment 的入口 bash 脚本
├── outputs/
│   ├── runs/           不同 proj/exp 的训练后权重
│   ├── eval/           不同 proj/exp 的 eval 结果
│   └── logs/           不同 proj/exp 的日志
├── docs/
│   ├── architecture/   当前 monorepo 的部分信息
│   └── notes/          不同 proj/exp 的过程文件
├── tempwork/           临时工作区，每次结束实验务必清理
├── .gitignore
├── .gitmodule
└── README.md           总信息文件
```

## Environment

Yenorepo通过多级环境管理，保证实验复现过程中，环境管理包括：
``` text
1. 文件管理：主要依赖git tag + manifest方案回放主repo和submodule的文件内容

2. 系统环境：在特殊情况下需要一些特殊的系统环境，通过bash脚本存放在scripts对应exp下（少见）

3. 二进制环境：一般建议通过conda或者通过docker等方式配置二进制依赖（开源前）

4. python环境：src通过pip install -e .方式安装，third_party的环境依赖通过在bash脚本下配置pushd-popd+PYTHONPATH解决，**推荐通过uv+pyproject.toml进行python环境管理**
```
## Protocol

- 为了本Monorepo的持续运行，需要统一一些Protocol

### uv & pyproject.toml

参考模板[pyproject.toml](./docs/template/pyproject.toml)*等待开发*

### git tag

tag需要满足命名规范：

``` text
<origin repo>-<project>-<experiment>-v<version>

e.g. rmbench-maevla-simvla_autogaze-v1
```

建议配置vscode插件git graph可视化tag+commit

### datasets

product需要满足命名规范：

``` text
<benchmark>_<task_config>_train<train_n>_eval<eval_n>

e.g. rmbench_demo_clean_train48_eval2
```

### src

src需满足一系列规范：

1. 外部规范

```text
src需要配置pyproject.toml以提供editable能力，需要使用src内容通过 pip install -e ./src（根目录下进行）
```

2. 内部规范

```text
使用相对索引避免出现问题，通过config提供配置接口，简洁、轻便（尽量通过单个config能够完成配置）
```

### bash

scripts下脚本需要能够在任何环境，任何配置下实现一键复现，主要脚本包含train、eval、manifest，参考模板[train.sh](./docs/template/exp_bash/train.sh)、[eval.sh](./docs/template/exp_bash/eval.sh)、[manifest.sh](./docs/template/exp_bash/manifest.sh)

其他规范：

```text
1. bash脚本需要按照类似config的方案，将超参数写在内部并配置默认参数

2. 尽量自动索引ROOT

3. 减少PYTHONPATH的使用，而是通过pushd-popd切换工作目录解决
```

### python for third_party

third_party的python依赖主要问题在于cross-repo，repo内，在外部bash脚本通过PYTHONPATH进行解决

### Smoke

需要根据不同的情况设计Smoke脚本，一般划分模块测试，测试内容包括：

1. 输入输出维度

2. 模型通路前向通路（前向通一般反向就通了）

3. 数据集输入

4. train，eval通路

可以通过随机数-数据集两阶段测试，分别测试Model、Train、eval通路以及数据集导入通路

### docs

## 使用方法