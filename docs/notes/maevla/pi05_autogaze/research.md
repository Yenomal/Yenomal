# Research

## 框架图

```text
                           [AutoGaze 训练数据]
     InternVid / Ego4D / 100DoH / scanning_* / gazing_labels.json
                                         |
                                         v
                         [VideoFolder Dataset + collate_fn]
                                         |
                      video / video_for_task / gt_gazing_info
                                         |
                                         v
                                [AutoGaze Model]
      shallow video conv + connector + llama-style gaze decoder + generate()
                                         |
              gazing_pos / if_padded_gazing / num_gazing_each_frame
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
      [VideoMAEReconstruction Task]                [SigLIP / downstream ViT]
   reconstruction loss / reward / metrics        只编码被 gaze 命中的 patch
                    |                                         |
                    v                                         v
           [NTP or GRPO Algorithm]                   更便宜的视频视觉特征
                    |
                    v
                 [Trainer]
          checkpoint / wandb / validate


                              [RMBench Task Env]
         task setup -> scene / robot / camera -> get_obs() -> observation
                                                          |
                                                          v
                                      [policy client_adapter]
                                                          |
                                                          v
                                      [policy_model_server]
                                                          |
                            SimVLA / SimVLA_AutoGaze / 其他策略后端
                                                          |
                                                          v
                                         action chunk (qpos trajectory)
                                                          |
                                                          v
                                             TASK_ENV.take_action()
```

## 总述

### 一句话总结

`AutoGaze` 是一个把视频 patch 选择问题做成自回归 gaze 生成的高效视觉前端，`RMBench` 是一个强调时序记忆与多步操作的机器人操作基准；前者解决“看哪里”，后者检验“看了之后能不能做对”。

### 解决问题

`AutoGaze` 要解决的是高分辨率、长时长视频里冗余 patch 太多，直接喂给 ViT/MLLM 代价过大。它把“先决定观察哪些 patch”单独建模，通过 NTP 预训练加 GRPO 后训练，让模型用更少 token 保住下游可用信息。

`RMBench` 要解决的是很多操作任务并不是单步反应型控制，而是必须记住过去观察、阶段状态、目标组合或顺序约束。它把重点放在任务构造、可重复数据采集、专家轨迹验证、评测协议和策略接入标准化上。

### 源码结构

`AutoGaze` 的主干在 `autogaze/`，核心分成五块：

`models/` 负责 gaze 模型本身。

`tasks/` 负责“如何评价 gaze 好不好”，这里主要是 `VideoMAEReconstruction`。

`algorithms/` 负责如何把任务信号变成训练 loss，核心是 `NTP` 和 `GRPO`。

`datasets/` 负责视频读取、采样、ground-truth gaze 对齐。

`vision_encoders/` 负责把标准 ViT 改造成能消费 `gazing_info` 的版本，这里已经实现了 `SigLIP`。

`RMBench` 的主干则是四层：

`envs/` 定义环境基类与具体任务。

`task_config/` 定义相机、随机化、数据采集与 embodiment 配置。

`script/` 提供数据采集、评测、客户端/服务端策略执行入口。

`policy/` 提供不同策略的适配层，当前与本工作区最相关的是 `SimVLA` 和 `SimVLA_AutoGaze`。

### 模型整体输入输出

`AutoGaze` 训练/推理时的核心输入是：

`inputs["video"]`: 形状为 `B x T x C x H x W` 的视频张量。

可选控制项：`gazing_ratio`、`task_loss_requirement`、`target_scales`、`target_patch_size`。

核心输出是：

`gazing_pos`: 被选中的 patch 全局索引。

`if_padded_gazing`: 哪些位置只是 padding。

`num_gazing_each_frame`: 每帧保留多少 gaze token。

训练态额外输出 `log_action_probs`、`task_loss_prediction`。

`RMBench` 评测链路的核心输入输出则是：

环境观测输入策略时，至少包含多路相机 `rgb` 和 `joint_action.vector`。

策略输出是未来一段 `qpos` action chunk。

环境通过 `TASK_ENV.take_action(action, action_type="qpos")` 将 chunk 逐步执行回仿真。

## 分模块讲解

### 模块1

`AutoGaze` 的训练编排非常模块化，入口在 `autogaze/train.py`。它用 `hydra` 实例化 `algorithm`、`model`、`task`、`dataset` 和 `trainer`，并在 DDP 下统一创建 `train_loader`/`val_loader`。

```text
[Hydra config]
      |
      v
[train.py]
  instantiate algorithm/model/task/dataset
      |
      v
[Trainer]
  _one_step():
    gaze_model(inputs)
      -> task(inputs, gaze_outputs)
      -> algorithm(inputs, gaze_outputs, task_outputs)
```

这里最重要的一点是职责边界切得很干净：

`model` 只负责“生成 gaze”。

`task` 只负责“给出 reconstruction loss / reward / metrics”。

`algorithm` 只负责“如何用这些信号优化 gaze model”。

所以你以后如果要换任务目标，理论上主要改 `tasks/` 即可；如果要换 RL 目标，主要改 `algorithms/`。

### 模块2

`AutoGaze` 的 gaze 模型主干在 `autogaze/models/autogaze/modeling_autogaze.py`。它不是直接拿一个巨型视频 Transformer 端到端看完整视频，而是分成：

```text
[video]
   |
   v
[ShallowVideoConvNet]
   |
   v
[Connector]
   |
   v
[LlamaForCausalLM_MultiTokenPred]
   |
   v
 predict next gaze token ids
```

`ShallowVideoConvNet` 先做轻量视频表征，`Connector` 把视觉特征映射到 decoder 维度，最后用类 `LLaMA` 的 decoder 自回归地产生 patch id。生成时还有两个关键约束：

`NoRepeatTokensLogitsProcessor` 禁止重复 gaze 同一 patch。

`NoEosTokenLogitsProcessor` 控制中途不要乱结束，真正的 early stop 由 `task_loss_requirement` 决定。

这个设计的核心价值是：gaze 序列本身被当成“token sequence”建模，所以它天然支持自回归、缓存、流式处理和多 token 并行预测。

### 模块3

`AutoGaze` 的任务模块目前主线是 `VideoMAEReconstruction`。它把“gaze 是否足够好”定义成：给定当前被观察到的 patch，VideoMAE 能不能把目标帧重建得足够好。

```text
[video_for_task + gazing_info]
            |
            v
[ViTMAEForPreTraining]
            |
            +--> reconstruction_loss
            +--> loss_each_reconstruction_frame
            +--> reconstruction
```

这里有两个很重要的衍生量：

`task_losses`: 每个 gaze token 对应的 reconstruction loss 监督，可用于训练 `task_loss_prediction head`。

`reward`: 对 RL 来说，直接取负 reconstruction loss。

所以 `AutoGaze` 的“看哪里”并不是凭空学习的，而是被一个冻结或半冻结的重建任务持续牵引。

### 模块4

`AutoGaze` 的两阶段训练逻辑写得非常清楚：

```text
Stage 1
GT gazing sequence
   -> gaze_model(gazing_info=gt)
   -> NTP loss

Stage 2
sample multiple gazing sequences
   -> task reward
   -> GRPO relative advantage
   -> optimize sampled gaze policy
```

第一阶段 `NTP` 用 `gazing_labels.json` 里的真值 gaze 序列做 next-token prediction，先把模型教会“像人一样看”。

第二阶段 `GRPO` 会把每个输入重复 `group_size` 次，采样多条 gaze 轨迹，再用组内相对优势做优化。这里的 reward 不是最终机器人成功率，而是 reconstruction reward，因此它本质上还是视觉前端后训练，而不是机器人端到端 RL。

### 模块5

`AutoGaze` 与下游 ViT 的集成点是 `gazing_info`。这部分在 `INTEGRATION.md` 和 `vision_encoders/siglip/` 里已经写得很清楚，逻辑可以总结成：

```text
[full patches of all frames and scales]
            |
            v
   mask_with_gazing(gazing_pos)
            |
            v
 [only selected patches remain]
            |
            v
 [SigLIP encoder + custom attention mask]
```

和标准图像 ViT 相比，真正改的只有两件事：

只嵌入被 gaze 到的 patch。

给多帧 token 构造 `block_causal` / `causal` / `bidirectional` attention mask。

这也是为什么 `AutoGaze` 可以作为“前置 patch 选择器”插在已有 ViT/MLLM 前面，而不用改完整个视觉主干。

### 模块6

`RMBench` 的环境基类 `Base_Task` 很厚，但主线很明确：

```text
_init_task_env_
  -> setup_scene()
  -> load_robot()
  -> load_camera()
  -> load_actors()
  -> play_once() / check_success()
```

`Base_Task` 统一封装了：

场景初始化，含光照、桌面、背景随机化。

机器人与相机加载。

观测接口 `get_obs()`。

双臂/单臂动作规划与执行接口 `move()`、`take_action()`。

数据缓存、录像、轨迹保存、HDF5 合并。

具体任务只需要重写 `load_actors()`、`play_once()`、`check_success()`。例如：

`battery_try` 的关键是电池方向组合记忆与仪表盘开关状态。

`cover_blocks` 的关键是按状态机顺序覆盖和打开盖子。

所以 `RMBench` 任务不是简单 pick-and-place，而是把“状态组合”“顺序依赖”“阶段性奖励”都揉进了 task logic。

### 模块7

`RMBench` 的数据采集流程分成两个阶段：

```text
[search valid seed]
      |
      v
 TASK_ENV.setup_demo(..., need_plan=True)
      |
      v
 TASK_ENV.play_once() + check_success()
      |
      v
 save seed + save expert joint path

[replay saved seed/path]
      |
      v
 need_plan=False + save_data=True
      |
      v
 get_obs() -> .pkl cache -> hdf5/video
```

`collect_data.py` 先找到能稳定成功的 seed，再回放专家 joint path 真正录数据。这意味着它不是“在线收集带噪策略数据”，而是“先找可解场景，再稳定重放专家轨迹”。

最终数据除了 `episode*.hdf5`，还有：

`scene_info.json`

`language_annotation.json`

自动生成的任务语言描述

### 模块8

`RMBench` 的评测主线在 `script/eval_policy.py` 和 `script/eval_policy_client.py`，而策略服务在 `script/policy_model_server.py`。

```text
TASK_ENV.get_obs()
   -> client_adapter.encode_obs()
   -> socket/json/base64 numpy transport
   -> server_adapter.get_action()
   -> action chunk
   -> TASK_ENV.take_action()
```

这个设计把仿真环境和模型服务显式解耦了，所以模型可以跑在单独进程，甚至不同 conda 环境里。代价是观测通过 JSON + base64 传输，图像带宽会比较重。

### 模块9

`RMBench/policy/SimVLA_AutoGaze` 是当前两个仓库最直接的接缝。它的 client 侧和 baseline 几乎一致，真正变化在 server 侧：构造 `head_history`，打开 `use_autogaze_obs_encoder`，并把以下配置注入模型：

`autogaze_model_path`

`autogaze_siglip_model_path`

`autogaze_history_len`

`autogaze_projector_hidden_size`

`autogaze_gazing_ratio`

所以在 `RMBench` 的视角里，AutoGaze 不是独立运行的训练器，而是一个被策略 observation encoder 消费的视觉前端选项。

## 训练配方

### 数据使用

`AutoGaze` 训练明确依赖：

`InternVid_res448_250K`

`100DoH_res448_250K`

`Ego4D_res448_250K`

`scanning_SAM_res448_50K`

`scanning_idl_res448_50K`

以及 `gazing_labels.json` 和 `VideoMAE_AutoGaze` 权重。

`RMBench` 自己不是一个统一训练器仓库，它提供的是任务环境、演示数据生成和评测协议。默认采集设置是 `demo_clean`，会保存 `rgb`、`third_view`、`qpos`、`endpose` 等观测。

### Setting

`AutoGaze` 的标准配方是：

第一阶段 `NTP`：8 卡、`batch_size=1024`、`lr=5e-4`、`n_epochs=150`，用 GT gaze 做监督。

第二阶段 `GRPO`：`group_size=12`、`discount_factor=0.995`、`batch_size=64`、从 NTP checkpoint 初始化，继续做 reconstruction-reward RL。

典型推理设置里常见的经验值是：

`gazing_ratio=0.75`

`task_loss_requirement=0.7`

`RMBench` 侧更像部署配方而不是训练配方。`SimVLA` 与 `SimVLA_AutoGaze` 的默认部署参数都包含：

`execute_horizon=5`

`integration_steps=10`

`expected_state_dim=14`

`camera_keys=[head_camera, left_camera, right_camera]`

## 其他

`RMBench` 在评测前会先用专家轨迹做一次可行性过滤，只有“专家可成功”的 seed 才进入正式 policy eval，这一点会显著影响成功率解释。

语言指令不是手写死的，而是由 `description/` 下的模板和 episode info 动态生成。

`AutoGaze` 已经原生支持流式视频推理，缓存形式包括 `past_key_values`、`past_input_embeds`、`past_attention_mask` 和 `past_conv_values`。

## 注意事项

### 实现边界

`AutoGaze` 仓库本身主要实现了 gaze 模型、重建任务、训练算法和一个 `SigLIP` 集成示例，并没有在本仓库里实现完整下游机器人策略训练框架。

`RMBench` 仓库主要实现 benchmark、环境和策略适配，不直接拥有 `SimVLA` 这类策略本体的完整训练代码。`policy/SimVLA*` 更接近“部署适配层”。

本文故意没有把 `yenomal` 展开写进来，因为它在这个工作区里承担的是工程编排与复用封装角色，已单独写入 `yenomal.md`。

### 注意事项

`AutoGaze` 官方训练分布是 16 帧、224 分辨率；跑任意长视频和更高分辨率时，依赖 `chunk + target_scales + target_patch_size` 这套适配逻辑，而不是“模型天然支持任意输入”。

`VideoMAEReconstruction` 当前显式断言 `frame_sampling_rate == 1`，如果你后面想改成更稀疏时间采样，任务侧先会卡住。

`RMBench` 的 `take_action()` 默认按 `qpos` chunk 执行，状态维度当前默认写死为 `14`，基本对应默认 `aloha-agilex` embodiment；换 embodiment 时别忘了同时检查策略适配层。

`SimVLA_AutoGaze` 在 `RMBench` 中只是把 AutoGaze 参数透传给策略模型，并不在这个目录里直接定义完整的 observation encoder 细节，所以你调 bug 时不要只盯着 `policy/SimVLA_AutoGaze` 一层。
