# Research

## 框架图

### 1. 当前仓库真正跑通的主线

```text
RMBench/data/data/<task>/demo_clean/
├─ data/episode*.hdf5
└─ instructions/episode*.json
        │
        ├─ SimVLA/create_rmbench_meta.py
        │      生成训练元数据 meta.json
        │
        └─ SimVLA/compute_rmbench_norm_stats.py
               生成 state/actions 归一化统计
        │
        ▼
SimVLA/datasets/dataset_smolvlm.py
        │
        ▼
SimVLA/datasets/domain_handler/rmbench_hdf5.py
        │
        │  从每个 episode 取：
        │  - 当前时刻 3 路 RGB
        │  - 当前 joint_action/vector 作为 proprio
        │  - 未来 num_actions 步 joint_action/vector 作为监督目标
        ▼
SimVLA/models/processing_smolvlm_vla.py
        │
        ├─ encode_image: resize + normalize
        └─ encode_language: tokenizer(max_length=50)
        ▼
SimVLA/models/modeling_smolvlm_vla.py
        │
        ├─ forward_vlm_efficient:
        │    vision_model -> connector -> text_model
        │
        └─ SmolVLMActionTransformer:
             Flow Matching 预测未来动作速度场
        ▼
velocity_loss(MSE)
        ▼
train_smolvlm.py / accelerate
        ▼
runs/.../ckpt-xxxx
```

### 2. RMBench 双进程评测时的 C/S 数据流

```text
RMBench env
  TASK_ENV.get_obs()
      │
      ▼
policy/SimVLA/client_adapter.py
  只抽取:
  - head_camera.rgb
  - left_camera.rgb
  - right_camera.rgb
  - joint_action.vector
  - instruction
      │
      ▼
ModelClient
  TCP socket
  4-byte length header + JSON
  numpy 用 base64 序列化
      │
      ▼
script/policy_model_server.py
      │
      ▼
policy/SimVLA/server_adapter.py
  SimVLAPolicyRunner.get_action()
      │
      ▼
SimVLA model.generate_actions()
      │
      ▼
返回 [execute_horizon, 14] 绝对 qpos chunk
      │
      ▼
TASK_ENV.take_action(action, action_type="qpos")
  内部做 TOPP / 关节插值 / 仿真推进
      │
      └─ 执行完一小段后重新 get_obs()，继续 replan
```

## 总述

### 一句话总结

当前仓库本质上是把 SimVLA 改造成一个面向 RMBench 的极简 VLA 基线：输入是“当前三视角图像 + 当前 14 维双臂关节状态 + 文本指令”，输出是“未来若干步 14 维绝对关节目标”，训练目标是 Flow Matching 的速度场回归，部署时再通过 Euler 积分生成动作块。

### 解决问题

它想解决的不是“做一个复杂的带长期记忆机器人系统”，而是给出一个足够简单、足够透明、足够容易迁移的基线：

- 感知直接复用 SmolVLM。
- 控制头只做未来动作序列建模。
- 用 LIBERO 版 SimVLA checkpoint 迁移到 RMBench。
- 把 RMBench 的问题改写成当前观测到未来 joint qpos chunk 的预测问题。

所以它的设计哲学非常明确：先把视觉语言 backbone 和动作头解耦，再用尽量少的工程改动把 benchmark 接进来。

### 源码结构

- `SimVLA/`
  这是主角。里面有 RMBench 元数据构造、归一化统计、数据读取、SmolVLM-VLA 模型、训练脚本。
- `RMBench/`
  这是数据、环境、评测和策略接入层。它负责：
  1. 生成/存放 episode 数据；
  2. 提供 SAPIEN 任务环境；
  3. 提供单进程和双进程评测入口；
  4. 用 `policy/SimVLA/` 把 SimVLA 接成一个 RMBench policy。
- `SimCkpt/`
  这是 LIBERO 版 SimVLA 预训练 checkpoint。RMBench 训练默认从它开始做参数迁移。

### 模型整体输入输出

### 训练时

输入：

- `input_ids`: `[B, 50]`
- `image_input`: `[B, 3, 3, H, W]`
- `image_mask`: `[B, 3]`
- `proprio`: `[B, 14]`
- `action`: `[B, num_actions, 14]`

输出：

- `{"velocity_loss": scalar}`

### 部署时

输入：

- `head_camera / left_camera / right_camera`: 当前 RGB 图像
- `state`: 当前 `joint_action/vector`，shape 为 `[14]`
- `instruction`: 当前任务指令

输出：

- `generate_actions(...) -> [num_actions, 14]`
- `server_adapter` 实际只返回前 `execute_horizon` 步，默认是 `[5, 14]`

这 14 维的语义是固定的：

```text
[left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_gripper,
 right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_gripper]
```

## 分模块讲解

### 模块1：RMBench 数据如何被改写成 SimVLA 可训练样本

`SimVLA/create_rmbench_meta.py` 做的事很简单，它不改数据内容，只给训练器一份“去哪里读 episode”的索引。

每个 episode 的实际布局是：

```text
episodeX.hdf5
├─ observation/head_camera/rgb          (T,)
├─ observation/left_camera/rgb          (T,)
├─ observation/right_camera/rgb         (T,)
├─ observation/front_camera/*           存在，但 SimVLA 不用
├─ joint_action/vector                  (T, 14)
├─ endpose/*                            存在，但 SimVLA 不用
├─ pointcloud                           通常为空或不用
└─ third_view_rgb                       用于视频，可视化，不进 SimVLA

episodeX.json
└─ {"seen": [...], "unseen": [...]}
```

它生成的 meta 里最关键的是：

- `dataset_name = "rmbench_hdf5"`
- `observation_key = ["observation/head_camera/rgb", "observation/left_camera/rgb", "observation/right_camera/rgb"]`
- `state_key = "joint_action/vector"`
- `state_dim = 14`
- `action_dim = 14`

也就是说，当前接入根本没有用到 front camera、third view、点云、末端位姿，只拿三路 RGB 和 joint vector。

### 模块2：一个训练样本是怎么切出来的

主链在 `SimVLA/datasets/dataset_smolvlm.py` 和 `SimVLA/datasets/domain_handler/rmbench_hdf5.py`。

真正的样本构造过程是：

```text
选中某个 episode
  │
  ├─ 在时间轴上挑一个 idx
  │
  ├─ 取 idx 时刻的 head/left/right 三张图
  │
  ├─ 取 joint_action/vector[idx] 作为当前 proprio
  │
  └─ 取 joint_action/vector[idx+1 : idx+num_actions]
      作为未来动作目标
      如果到尾部不够长，就重复最后一帧补齐
```

最终 `abs_trajectory` 的组织是：

```text
row 0      = 当前 joint qpos
row 1..N   = 未来 N 步 joint qpos
```

然后 `datasets/utils.py` 里的 `action_slice()` 会把它拆成：

- `proprio = abs_trajectory[0]`
- `action = abs_trajectory[1:]`

这里对 RMBench 非常关键的一点是：

- 它预测的是绝对 joint target，不是 delta end-effector action。
- 它只看当前这一帧，不看历史帧。
- 它没有显式记忆模块。

所以从“记忆 benchmark”的角度看，这个 baseline 本质上是一个无外部记忆、无观测历史缓存的单帧重规划策略。

### 模块3：Processor 和视觉语言编码链路

`SimVLA/models/processing_smolvlm_vla.py` 负责把图像和文字变成模型输入。

图像路径：

```text
numpy / PIL / torch image
  └─> resize 到 image_size
  └─> normalize(mean/std)
  └─> pad 到 3 个视角
  └─> image_input: [B, 3, 3, H, W]
```

文字路径：

```text
instruction string
  └─> tokenizer
  └─> max_length = 50
  └─> input_ids: [B, 50]
```

真正进入主模型的不是 README 里那条“按原生 chat template 直接走 SmolVLM”的慢路径，而是 `modeling_smolvlm_vla.py` 里的 `forward_vlm_efficient()`：

```text
valid images
  └─> self.vlm.model.vision_model(...)
  └─> image patch features
  └─> connector / multi_modal_projector
  └─> 投到 language hidden size

input_ids
  └─> self.vlm.model.text_model.get_input_embeddings()
  └─> text embeddings

[all image tokens | all text embeddings]
  └─> self.vlm.model.text_model(inputs_embeds=...)
  └─> last_hidden_state = vlm_features
```

这意味着当前仓库真正使用的是“手工拼接视觉 token 与文本 token，再过 text model”的自定义快路径，而不是 `forward_vlm()` 里那条基于 `apply_chat_template()` 的原生多模态推理路径。`forward_vlm()` 在当前训练/推理主线上没有被调用。

### 模块4：动作空间、状态空间和归一化

`SimVLA/models/action_hub.py` 里对 RMBench 定义了 `RMBenchJointActionSpace`：

- `dim_action = 14`
- `dim_proprio = 14`
- `gripper_idx = (6, 13)`

归一化统计由 `compute_rmbench_norm_stats.py` 生成：

```text
state stats   <- joint_action/vector[:-1]
action stats  <- joint_action/vector[1:]
```

输出 JSON 里有：

- `mean`
- `std`
- `q01`
- `q99`

默认走 z-score，只有显式开 `use_quantile_norm` 才走分位数归一化。

这里的建模含义非常直接：

- 当前机器人姿态 `q_t` 是状态；
- 未来机器人姿态 `q_{t+1:t+H}` 是监督目标；
- 不是模仿末端轨迹，不是模仿抓取 primitives，也不是 tokenized action。

### 模块5：Flow Matching 动作头到底在做什么

动作头在 `SimVLA/models/transformer_smolvlm.py`，默认是 `use_adaln = false` 的 concat 模式。

训练公式在 `SimVLA/models/modeling_smolvlm_vla.py` 里很清楚：

```text
1. 采样 t ~ Beta(1.5, 1) * 0.999 + 0.001
2. noise ~ N(0, I)
3. x_t = t * noise + (1 - t) * action_norm
4. u_t = noise - action_norm
5. transformer(vlm_features, x_t, proprio_norm, t) -> v_t
6. loss = MSE(v_t, u_t)
```

默认 concat 模式下，action token 的构造是：

```text
[noisy_action_t, repeated_proprio, repeated_time_embedding]
    └─> action_encoder
```

然后再把 `vlm_features` 线性投影后直接拼在 action tokens 后面，一起做 self-attention：

```text
[action tokens | projected vlm tokens]
   └─> Transformer blocks
   └─> 只解码前 num_actions 段
```

可选的 `use_adaln=true` 分支会改成 DiT/AdaLN 风格：

- action 序列单独作为 token 序列；
- time / pooled VLM / proprio 融合成全局条件 `c`；
- 每层通过 AdaLN 注入条件。

但当前 checkpoint 和训练脚本默认都没开这个分支。

### 模块6：推理时为什么会“先预测 10 步，但只执行 5 步”

`generate_actions()` 的推理逻辑是 Euler 积分：

```text
x_1 ~ N(0, I)
for t = 1 -> 0:
    v_t = transformer(...)
    x_t = x_t + dt * v_t
最后得到 x_0
postprocess 反归一化
```

模型本身通常预测 `num_actions = 10` 步未来 qpos。

但 RMBench 侧的 `policy/SimVLA/server_adapter.py` 不会把这 10 步全执行掉，而是：

- 先预测 10 步；
- 只截前 `execute_horizon` 步，默认 5 步；
- 执行 5 步后重新观察，再重规划。

所以实际部署语义是：

```text
单次前向 = 生成一个较长未来块
在线控制 = 只执行前一半，然后再次感知和重规划
```

### 模块7：训练脚本如何复用 LIBERO 版 SimVLA checkpoint

`SimVLA/train_rmbench_joint_all.sh` 的默认做法是：

```text
INIT_CKPT = ../SimCkpt
```

这个 `SimCkpt` 是一个 LIBERO 版 checkpoint，配置是：

- `action_mode = libero_joint`
- `hidden_size = 1024`
- `depth = 24`
- `num_heads = 16`
- `num_actions = 10`
- `image_size = 384`

RMBench 训练不是严格直接加载，而是走 `load_matching_checkpoint()`：

```text
如果 key 存在且 shape 相同 -> 加载
否则 -> 跳过
```

这意味着：

- SmolVLM backbone 大部分参数可复用；
- transformer 里与通道数一致的部分大多可复用；
- 动作头中那些因 `7维 LIBERO 动作` 变成 `14维 RMBench 动作` 而 shape 不同的层，会被跳过并重新初始化。

这就是它迁移到 RMBench 的核心手法：尽量复用视觉语言与共享序列建模部分，只把 RMBench 特有的动作输出层重新学一遍。

## 训练配方

### 数据使用

当前下载好的 RMBench 数据目录实际包含 12 个任务：

- `battery_try`
- `blocks_ranking_try`
- `classify_blocks`
- `cover_blocks`
- `observe_and_pickup`
- `place_block_mat`
- `press_button`
- `put_back_block`
- `rearrange_blocks`
- `storage_blocks`
- `swap_T`
- `swap_blocks`

每个任务在 `demo_clean` 下有：

- `50` 个 `episode*.hdf5`
- `50` 个 `instructions/episode*.json`

默认训练脚本想做的是：

- 每任务前 40 条训练
- 后 10 条留作 eval meta

所以理论上训练集大小是：

```text
12 tasks * 40 episodes = 480 episodes
```

已经给出的 `norm_stats/rmbench_all_train_40of50_joint_norm.json` 里记录的训练统计是：

- `num_episodes = 480`
- `num_steps = 337014`

但要注意，SimVLA 真正使用的字段只有：

- `observation/head_camera/rgb`
- `observation/left_camera/rgb`
- `observation/right_camera/rgb`
- `joint_action/vector`

以下字段虽然在 HDF5 里有，但当前主线不用：

- `observation/front_camera/*`
- `third_view_rgb`
- `endpose/*`
- `pointcloud`

### Setting

`train_rmbench_joint_all.sh` 默认配置可以概括成：

- 模型初始化：`../SimCkpt`
- backbone：`HuggingFaceTB/SmolVLM-500M-Instruct`
- 图像尺寸：`384`
- 未来动作步数：`10`
- batch size：`8`
- learning rate：`1e-4`
- VLM learning coef：`0.1`
- iterations：`100000`
- freeze steps：`1000`
- warmup steps：`0`
- hidden size：`1024`
- depth：`24`
- heads：`16`
- mixed precision：`bf16`
- grad clip：`1.0`
- save interval：`5000`

优化器是 `AdamW`，参数被拆成 3 组：

- `vlm`
- `transformer_core`
- `action_heads`

冻结策略是：

- 前 `freeze_steps`，只训练 `action_heads`
- 之后再放开 `transformer_core`
- `vlm` 也在之后放开，但学习率乘 `learning_coef`

这说明它默认假设：

- 视觉语言 backbone 已经比较成熟；
- RMBench 适配初期主要先让动作输出层对上；
- 再慢慢微调 backbone。

另一个容易忽略的点是：

- 脚本会生成 `eval meta`
- 但 `train_smolvlm.py` 实际只接收 `--train_metas_path`
- 当前训练循环里没有验证集 forward，也没有 early stopping

所以“eval split”在当前脚本里只是被生成出来，并没有真正参与训练过程中的验证。

## 其他

### RMBench 的作用

RMBench 在这个仓库里承担三件事：

1. 提供仿真环境和任务定义。
2. 提供 episode 数据格式和数据收集流程。
3. 提供 policy 评测框架，把 SimVLA 当成一个可插拔 policy 来调用。

从代码上看，RMBench 的数据收集是两阶段：

```text
阶段1：先用环境自带 expert / planner 找到成功 seed，并保存 joint path
阶段2：回放 joint path，把每一步观测存成 pkl，再合并成 hdf5 + mp4
```

这也是为什么最终 HDF5 里会天然带有：

- RGB 序列
- 当前 joint qpos 序列
- 末端位姿序列

### RMBench 的 C/S 数据流动过程

双进程版本入口是 `RMBench/policy/SimVLA/eval_double_env.sh`。

它先启动：

- 服务端：`script/policy_model_server.py`
- 客户端：`script/eval_policy_client.py`

服务端职责：

- 读取 `policy/SimVLA/deploy_policy.yml`
- 通过 `policy/SimVLA/server_adapter.py:get_model()` 加载 SimVLA checkpoint
- 监听 TCP socket
- 接收 `cmd + obs`
- 调用 `get_action()` 或 `reset_model()`

客户端职责：

- 启动 RMBench 环境
- 做 `expert_check`
- 生成当前 episode 的 `seen/unseen` 指令
- 把 `TASK_ENV.get_obs()` 转成 SimVLA 所需的轻量 payload
- 通过 `ModelClient.call()` 向服务端请求动作
- 收到动作块后逐步 `TASK_ENV.take_action(..., action_type="qpos")`

### 关键接口

#### 1. 环境侧观测接口

`TASK_ENV.get_obs()` 返回的是一个很大的字典，但 SimVLA 只取下面这些：

```text
observation.head_camera.rgb
observation.left_camera.rgb
observation.right_camera.rgb
joint_action.vector
instruction
```

#### 2. 传输接口

底层协议不是 HTTP，而是：

- TCP socket
- 4 字节大端长度头
- 后面跟 JSON body
- numpy 数组用 base64 编进 JSON

请求大概长这样：

```json
{
  "cmd": "get_action",
  "obs": {
    "head_camera": "... uint8[H,W,3] ...",
    "left_camera": "... uint8[H,W,3] ...",
    "right_camera": "... uint8[H,W,3] ...",
    "state": "... float32[14] ...",
    "instruction": "..."
  }
}
```

响应大概是：

```json
{
  "res": "... float32[execute_horizon, 14] ..."
}
```

#### 3. 服务端 policy 接口

`policy/SimVLA/server_adapter.py` 实际只暴露两个核心方法：

- `reset_model() -> bool`
- `get_action(obs) -> np.ndarray[execute_horizon, 14]`

其中 `get_action()` 内部流程是：

```text
obs
  └─> processor.encode_image(images)
  └─> processor.encode_language(instruction)
  └─> proprio = 当前 state 向量（14维）
  └─> model.generate_actions(...)
  └─> 截前 execute_horizon 步返回
```

#### 4. 环境执行接口

客户端拿到动作后调用：

```text
TASK_ENV.take_action(action, action_type="qpos")
```

这里不是简单“把 14 维向量直接塞给机器人”，而是：

- 把左右臂 qpos 目标拆开；
- 走 TOPP/规划器得到更密的关节轨迹；
- gripper 线性插值；
- 仿真步进；
- 每一步都检查 `check_success()`

所以 SimVLA 输出的是“目标关节块”，而真正执行层仍然是 RMBench 环境自己的轨迹推进逻辑。

## 注意事项

### 实现边界

1. 当前仓库的训练入口并不是完全可直接运行。`SimVLA/datasets/domain_handler/` 目录里没有 `libero_hdf5.py`，但 `__init__.py` 和 `registry.py` 还在强行 import 它，所以现在 `cd SimVLA && python train_smolvlm.py ...` 会在 `import datasets` 这一步直接报错。也就是说，RMBench 训练主线代码写出来了，但当前仓库缺了一块 LIBERO handler，导致 import 边界没有清干净。不过 RMBench 的推理服务链不依赖这个 `datasets` 包，所以部署链和训练链的可用状态并不一致。

2. RMBench 数据和 norm stats 覆盖 12 个任务，但 `RMBench/envs/` 里实际只实现了 10 个环境类，缺 `classify_blocks.py` 和 `storage_blocks.py`。所以“训练看到的数据范围”和“当前仓库能直接评测的环境范围”并不完全一致。

3. 当前 SimVLA-RMBench 基线没有显式记忆实现。没有历史帧堆叠，没有 recurrent state，没有 KV cache 管理，`SimVLAPolicyRunner` 里虽然有个 `obs_cache`，但实际上完全没用上。

4. `SimVLA/models/modeling_smolvlm_vla.py` 虽然自带 FastAPI `/act` 接口，但 RMBench 接它时走的是 `policy_model_server.py + socket + adapters` 这套链路，不走 FastAPI。

5. 当前主线强绑定默认 `aloha-agilex` 双臂 14 维 joint vector。你如果想换 embodiment，不是只改 config 就行，还要同步改：
   - action space 维度；
   - norm stats；
   - server adapter 的 `expected_state_dim`；
   - 训练数据字段定义；
   - 环境执行时的 action 拆分逻辑。

6. README、注释和部分 docstring 还残留着“512x512”的旧表述，但当前 checkpoint、训练脚本和 config 的实际值是 `384`。这个项目文档和当前代码状态并不完全同步。

### 注意事项

1. `create_rmbench_meta.py` 和 `compute_rmbench_norm_stats.py` 用的是字符串排序 `sorted(glob(...episode*.hdf5))`，不是按 episode id 的数值排序。所以默认所谓“前 40 个 train、后 10 个 eval”在真实代码里其实变成：

```text
train: [0, 1, 10, 11, ..., 44]
eval : [45, 46, 47, 48, 49, 5, 6, 7, 8, 9]
```

这会直接影响训练/评测切分语义。

2. `forward_vlm_efficient()` 当前没有使用 tokenizer 返回的 `attention_mask`，而是把拼好的 `[image tokens | text tokens]` 整段都置成了有效位。也就是说，padding 出来的文本 token 在当前实现里也会进入 text model 的有效上下文。

3. `compute_rmbench_norm_stats.py` 里的 `q01/q99` 不是全量精确统计，而是从 episode 中随机抽样累计，最多保留约 `100000` 个样本，所以分位数统计是近似值，而且不是严格确定性的。

4. 当前下载到的 `instructions/episode*.json` 里，`seen` 和 `unseen` 基本是同一条字符串。训练侧虽然会把两者合并采样，但在这份数据上几乎没有额外语言增广收益。

5. 评测时使用的指令并不是直接读取训练集里的 `instructions/episode*.json`，而是由 `description/utils/generate_episode_instructions.py` 根据 `scene_info` 和任务模板现生成，再按 `instruction_type=seen/unseen` 选择。这意味着训练语言源和评测语言源不是完全同一条路径。

6. `train_rmbench_joint_all.sh` 会生成 eval meta，但当前训练循环根本没消费它。如果你后面想加验证集、best checkpoint、early stopping，需要自己把这一段补起来。

7. RMBench 的几个主脚本在真正执行前都会先跑一次 `Sapien_TEST()`。这相当于强依赖本机的渲染/显示环境先可用，否则脚本会在正式逻辑开始前就退出。
