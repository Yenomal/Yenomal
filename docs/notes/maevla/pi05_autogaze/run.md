# run

## 说明

当前已经补好一套最简可运行的 `pi05 + common AutoGaze/SigLIP` PyTorch 训练与评测脚本：

- 训练脚本：`/home/rui/Yenomal/tempwork/train_common_autogaze_pytorch.sh`
- 评测脚本：`/home/rui/Yenomal/tempwork/eval_common_autogaze_pytorch.sh`

使用方式是：

1. 打开对应的 `bash` 脚本
2. 只修改脚本顶部的参数默认值
3. 直接运行 `./xxx.sh`

不需要在命令行额外传参。

## 训练

进入目录：

```bash
cd /home/rui/Yenomal/tempwork
```

编辑训练脚本里的这些变量：

```bash
TRAIN_CONFIG_NAME="pi05_aloha_common_autogaze_base"
REPO_ID="你的数据集路径或 LeRobot repo_id"
ASSET_ID="你的 asset id"
MODEL_NAME="common_autogaze_stage_a"
STAGE="stage_a"
HISTORY_LEN=8
DEVICE="cuda"
INIT_WEIGHT_PATH=""
WANDB_PROJECT="pi05-common-autogaze"
```

如果你要跑不同阶段，建议这样改：

- `stage_a`：先对齐新的视觉桥接层
- `stage_b`：打开 action expert / action head
- `stage_c`：再打开 SigLIP 顶层

运行训练：

```bash
./train_common_autogaze_pytorch.sh
```

训练脚本会自动做两件事：

- 如果 `norm_stats.json` 不存在，先自动调用 `compute_norm_stats.py`
- 按 `save_interval` 保存 checkpoint，同时维护一个 `latest/` 目录方便直接评测

wandb 会记录：

- `loss`
- `grad_norm`

其中 `grad_norm` 只统计 `common_visual_prefix.stack.projector.net.*` 这几层，也就是 projector MLP 的梯度范数。

## 评测

编辑评测脚本里的这些变量：

```bash
TASK_NAME="battery_try"
TASK_CONFIG="demo_clean"
TRAIN_CONFIG_NAME="pi05_aloha_common_autogaze_base"
MODEL_NAME="common_autogaze_stage_a"
CHECKPOINT_ID="latest"
SEED=0
GPU_ID=0
PI0_STEP=50
```

运行评测：

```bash
./eval_common_autogaze_pytorch.sh
```

如果你想评测某个固定 step 的 checkpoint，把：

```bash
CHECKPOINT_ID="latest"
```

改成：

```bash
CHECKPOINT_ID="1000"
```

它会先在后台启动 `RMBench/script/policy_model_server.py`，再通过 `RMBench/script/eval_policy_client.py` 发起 benchmark 评测。

它会读取：

```text
/home/rui/Yenomal/tempwork/RMBench/policy/pi05/checkpoints/<TRAIN_CONFIG_NAME>/<MODEL_NAME>/<CHECKPOINT_ID>/
```

## 关键路径

训练输出目录默认是：

```text
/home/rui/Yenomal/tempwork/RMBench/policy/pi05/checkpoints/<TRAIN_CONFIG_NAME>/<MODEL_NAME>/
```

其中每个 step checkpoint 下至少包含：

- `model.safetensors`
- `assets/<ASSET_ID>/norm_stats.json`
- `train_meta.json`

## 前提条件

运行前需要保证环境里有这些依赖：

- `torch`
- `wandb`
- `safetensors`
- `lerobot`
- `openpi` 当前依赖链

另外需要确保：

- `PYTHONPATH` 能访问 `yenomal/` 和 `RMBench/policy/pi05/src/`
- `MAEVLA_AUTOGAZE_STACK_CONFIG` 指向 `yenomal/projects/maevla/simvla_autogaze/config/autogaze_siglip_simvla.yaml`

这两个环境变量已经在 `bash` 脚本内部默认设置好了。

## 当前边界

这次已经打通的是：

- `pi05` 的 PyTorch 推理链
- `common visual encoder` 接 prefix token 的模型结构
- `head_history` 的数据流
- 部分权重加载
- `RMBench` benchmark 的 client/server adapter 解耦

还没有完整验证的是：

- 真实训练环境中 `torch + safetensors + wandb + dataset` 全部齐备时的端到端训练
- 多阶段训练是否比原始 `pi05` 更稳定
- 最终策略效果是否优于原视觉塔

所以推荐先做一次很小规模 smoke run，再开始正式长训。
