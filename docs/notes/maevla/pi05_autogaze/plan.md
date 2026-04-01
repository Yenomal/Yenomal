# `plan.md` 内容草案：`pi05` 接入 `common AutoGaze + SigLIP` Visual Encoder

## 摘要

本次改动只做 `PyTorch pi05` 路线，不做 `JAX/Flax` 对齐。总体思路是把 `pi05` 里原本“逐张图调用 `PaliGemma vision tower`”的前缀视觉分支替换成 `common/vision_stacks/autogaze_siglip`，让 `head_history` 先经 `AutoGaze` 稀疏选择，再与当前三视角一起经 `SigLIP` 编码，最后直接作为 `pi05` 的 prefix visual tokens。`pi05` 原有的语言模型、action expert、采样逻辑、prefix/suffix 总体结构保留。

本期默认策略是：`common` 继续使用当前三视角输入约定和 `attn_type=bidirectional`；不回加复杂 observation mask；不强求兼容原视觉塔的权重分布；先做一个“可训练、可推理、可部分加载旧 checkpoint”的 `pi05` 分支，验证训练是否收敛。

## 关键改动

### 1. 模型接口与输入数据

- 给 `pi05` 的模型输入新增可选字段 `head_history`，用于承载头相机历史帧序列；字典形态统一为 `[*b, t, h, w, c]`，PyTorch 路线进入模型前转成 `[B, T, C, H, W]`。
- 保留现有三路当前图像输入键，不改 `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb` 这套外部接口；`common` 适配器内部把它们映射为 `current_views`。
- `pi05` 的 prefix 侧不再逐张图调用 `embed_image()`，而是一次性调用一个新的 `common` 视觉适配器，输出 `visual_tokens` 和 `visual_pad_mask`。
- prefix 级 attention 语义保持现状：visual prefix tokens 与 language prefix tokens 之间仍然全互相可见；这次只替换 visual token producer，不改 `pi05` prefix/suffix 的总 mask 语义。
- `common` 的 projector 输出维度强制对齐 `pi05 / PaliGemma` 当前 prefix hidden size；不新增第二个 projector，直接把 `common` 输出当成 `image tokens` 使用。

### 2. 需要改动的文件

- `RMBench/policy/pi05/src/openpi/models/pi0_config.py`
增加 `use_common_visual_encoder: bool` 和 `common_visual_history_len: int`。默认关闭；新实验配置打开。
- `RMBench/policy/pi05/src/openpi/models/model.py`
扩展 `Observation` / `from_dict()` / `to_dict()` 支持 `head_history`；重写 `load_pytorch()` 为手动 `state_dict` 过滤加载，允许跳过旧视觉塔和旧 multimodal projector。
- `RMBench/policy/pi05/src/openpi/models_pytorch/preprocessing_pytorch.py`
让 `SimpleProcessedObservation` 保留 `head_history`，并在 PyTorch 预处理链中传递该字段，不做图像增强。
- `RMBench/policy/pi05/src/openpi/models_pytorch/pi0_pytorch.py`
新增 `use_common_visual_encoder` 分支；在 `__init__` 中挂接新的视觉适配器；重写 `embed_prefix()`，改成直接吃 `common` visual tokens；原 `embed_image()` 循环保留为 fallback 路径。
- 新增 `RMBench/policy/pi05/src/openpi/models_pytorch/common_visual_prefix.py`
封装从 `Observation` 中读取 `head_history + 3 current views`，调用 `common.vision_stacks.autogaze_siglip`，并返回 `visual_tokens / visual_pad_mask`；这层只做 `pi05` 适配，不改 `common`。
- `RMBench/policy/pi05/src/openpi/policies/aloha_policy.py`
接受可选 `head_history` 输入并保留到模型输入字典；当前三视角映射逻辑不变。
- `RMBench/policy/pi05/pi_model.py`
推理时增加头相机历史缓存；每一步更新滚动窗口，并把 `head_history` 放入 `observation_window`。
- `RMBench/policy/pi05/deploy_policy.py`
`encode_obs()` 继续取当前三路图和 state；推理缓存交给 `pi_model.py` 维护，不在这里单独处理历史。
- `RMBench/policy/pi05/scripts/process_data.py`
在数据转换时直接物化 `head_history`，为每个 step 写一个长度为 `history_len` 的头相机历史张量，左侧用首帧补齐；建议写到 `observation.head_history` 对应的 HDF5 路径，避免塞进 `images` 子树。
- `RMBench/policy/pi05/src/openpi/training/config.py`
新增一个独立训练配置，例如 `pi05_aloha_common_autogaze_base`，其 `RepackTransform` 需要额外映射 `head_history`；原有 `pi05` 配置不动。
- 训练/推理启动脚本需要显式设置 `PYTHONPATH=<repo>/src` 和 `MAEVLA_AUTOGAZE_STACK_CONFIG=<.../autogaze_siglip_simvla.yaml>`，保证 `pi05` 侧能导入 `common`。

### 3. 权重加载策略

- 默认不完全导入旧 checkpoint。
- 需要保留并加载的参数：`paligemma_with_expert.paligemma.language_model.*`、`paligemma_with_expert.gemma_expert.*`、`action_in_proj.*`、`action_out_proj.*`、其它 suffix/action 相关层。
- 需要跳过并重新初始化的参数：`paligemma_with_expert.paligemma.model.vision_tower.*`、`paligemma_with_expert.paligemma.model.multi_modal_projector.*`、新增的 `common` 视觉适配器和 `common` encoder 自身参数。
- 加载方式改为：手动读 `model.safetensors`，过滤旧视觉塔和旧 projector 的 key，`load_state_dict(..., strict=False)`，并打印 `missing/unexpected keys`；如果缺失的 key 超出“新视觉分支 + 旧视觉分支”范围，直接报错。
- 默认不尝试把旧 multimodal projector 权重迁移到 `common` projector；这一步留到第二轮实验再做。

### 4. 训练流程与学习率调度

- `Stage A`：只对齐新视觉桥接层，训练 `common projector`、`source/age embeddings`、`pi05` 视觉适配器里新增的小型桥接层；冻结 `AutoGaze`、`SigLIP backbone`、`PaliGemma language_model`、`action expert`。默认 `1000` step，学习率 `1e-4`，`5% warmup + cosine decay`。
- `Stage B`：打开 `action expert` 和 `action_out_proj`，继续冻结 `PaliGemma language_model`、`AutoGaze`、`SigLIP backbone`。默认 `1000` step。参数组学习率：新视觉桥接层 `5e-5`，`action expert/action_out_proj` `2e-5`，同样 `5% warmup + cosine decay`。
- `Stage C`：如果 `Stage B` 收敛但性能仍不足，再打开 `SigLIP` 顶层若干层，默认只开最高 `4` 层；`AutoGaze` 继续冻结。默认 `1000` step。学习率：`SigLIP top layers = 5e-6`，桥接层 `2e-5`，action expert/head `1e-5`。
- `Stage D`：本版默认不打开 `PaliGemma language_model`。只有在前三阶段已经稳定收敛、且需要进一步逼近旧分布时，才开最后若干层 LM，学习率固定在 `1e-6` 级别，作为后续迭代，不放进第一版实现。
- 默认训练顺序是“桥接层先对齐，再 action expert，再 SigLIP，高层 LM 延后”。不采用“一开始全开 SigLIP 或全开 VLM”的策略。

## 测试流程

### 1. 静态与单元检查

- 配置检查：新训练配置能正确实例化 `Pi0Config(use_common_visual_encoder=True)`，并且 `common_visual_history_len` 与 `common` YAML 中的 `history_len` 一致。
- 假数据前向：构造带 `head_history` 的 fake observation，验证 `PI0Pytorch.embed_prefix()` 能输出合法的 `prefix tokens / pad mask / att mask`，shape 与旧路径一致，hidden size 与 `PaliGemma` 前缀维度一致。
- 权重加载检查：partial load 后，`missing/unexpected keys` 只包含预期的旧视觉塔和新视觉适配器相关项。

### 2. 推理 smoke test

- 在 `RMBench` 中用 `pi05` 跑单 episode，确认 `pi_model.py` 的头相机缓存正常滚动，`head_history` shape 正确。
- 执行一次完整 `Policy.infer()`，确认不因 `head_history`、`common` encoder、prefix token 拼接而报错。
- 检查动作输出维度、`pi0_step` 截断行为和 `TASK_ENV.take_action()` 链路保持不变。

### 3. 训练 smoke test

- 用极小配置跑 `10~50` 个 step，确认 `compute_loss()`、反向传播、优化器 step 都正常。
- 在 `Stage A` 检查只有新视觉桥接层参数有梯度；在 `Stage B` 检查 action expert/head 有梯度；冻结层梯度为零。
- 用 `1` 个 episode 或极小子集做 tiny overfit，验证 `Stage A + B` 的训练 loss 至少有显著下降趋势。

### 4. 验收标准

- `common` encoder 替换后，`pi05` PyTorch 路线能正常训练与推理。
- 旧 checkpoint 可以部分加载，且不会因视觉塔替换导致整模加载失败。
- `Stage A` 和 `Stage B` 能稳定跑完，不出现 shape mismatch、NaN、attention mask 错误或 prefix/suffix 对齐错误。
- 第一版不要求优于原 `pi05`，只要求“链路完整、loss 可降、推理可出动作”。

## 假设与默认值

- 本版只做 `PyTorch pi05`，不要求 `JAX/Flax` 路线同步。
- `common` 继续沿用当前输入设计：历史头相机走 `AutoGaze`，当前三视角走 `SigLIP`，`attn_type=bidirectional`。
- `common` 的复杂 observation mask 暂不启用，后续如果需要，再作为独立 `mask policy` 层加入，不和 `SigLIP` 深耦合。
- 训练数据侧默认通过 `process_data.py` 物化 `head_history`，不依赖 `LeRobotDataset` 运行时拼历史。
- 第一版默认不复用旧视觉塔权重，也不尝试做旧 projector 到新 projector 的参数映射。
