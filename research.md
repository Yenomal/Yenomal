# AutoGaze + SigLIP 调研记录

## 阅读范围

- `src/common/vision_stacks/autogaze_siglip/config.py`
- `src/common/vision_stacks/autogaze_siglip/stack.py`
- `src/projects/maevla/simvla_autogaze/sandbox/models/observation_encoder_autogaze.py`
- `third_party/SimVLA/models/modeling_smolvlm_vla.py`
- `third_party/SimVLA/datasets/domain_handler/rmbench_hdf5.py`
- `third_party/AutoGaze/autogaze/models/autogaze/autogaze.py`
- `third_party/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py`
- `third_party/AutoGaze/INTEGRATION.md`

## 模块定位

`src/common/vision_stacks/autogaze_siglip` 本质上是在把 `third_party/SimVLA/models/observation_encoder_autogaze.py` 里的实现抽成一个可复用的公共视觉栈。

它当前的职责不是完整的 VLM，而是一个“视觉 token 生产器”：

- 输入：历史 `head` 视频帧 + 当前多视角图像
- 中间：先用 AutoGaze 选历史 patch，再用改造过的 SigLIP 编码
- 输出：一串给下游模型消费的视觉 token、attention mask、以及 gazing metadata

当前仓库里，这个公共 stack 主要通过 `src/projects/maevla/simvla_autogaze/sandbox/models/observation_encoder_autogaze.py` 接回 SimVLA，然后在 `third_party/SimVLA/models/modeling_smolvlm_vla.py` 里和文本 token 拼接，继续走 SmolVLM 的 text model。

## 项目主流程

### 1. 数据侧

RMBench 的数据读取在 `third_party/SimVLA/datasets/domain_handler/rmbench_hdf5.py`。

- 当前时刻会读取最多 3 个视角，组成 `image_input`
- 同时会从第一个视角流里取历史帧，组成 `head_history`
- 历史长度由 `history_len` 控制，不足时会用最早一帧左侧补齐

这意味着当前设计默认把“第一个相机流”当成 head camera，AutoGaze 只看这一路历史序列。

### 2. 训练装配

训练脚本 `scripts/projects/maevla/simvla_autogaze/train.sh` 会打开 `--use_autogaze_obs_encoder`，把 AutoGaze 相关参数传给 SimVLA。

随后：

1. `third_party/SimVLA/train_smolvlm.py` 把这些参数写进 `SmolVLMVLAConfig`
2. `third_party/SimVLA/models/modeling_smolvlm_vla.py` 检测到 `use_autogaze_obs_encoder=True`
3. 模型实例化 `AutoGazeObservationEncoder`
4. 当前项目里的 sandbox wrapper 再去加载 `src/projects/maevla/simvla_autogaze/config/autogaze_siglip_simvla.yaml`
5. wrapper 用 YAML 作为基线配置，但会在运行时覆盖 `autogaze_model_path`、`siglip_model_path`、`history_len`、`output_dim` 等关键参数

所以这里的 YAML 更像“默认合同”，而不是最终唯一配置源。

### 3. `AutoGazeSiglipVisionStack` 前向流程

`src/common/vision_stacks/autogaze_siglip/stack.py` 的主流程可以概括成下面几步。

#### 3.1 初始化

- 从 `third_party/AutoGaze` 动态导入 `AutoGaze`、`AutoGazeImageProcessor`、`SiglipVisionModel`
- 解析本地模型路径；如果给的是 Hugging Face repo id，则只在本地 cache snapshot 里找，不直接联网下载
- 构建一个 MLP projector，把 SigLIP hidden size 投到下游需要的 `output_dim`
- 可选加入两类 embedding：
  - `source_embed`：区分 token 来自 history / head / left / right
  - `age_embed`：区分历史 token 的时间年龄

#### 3.2 输入重归一化

这个 stack 假设上游送进来的图像已经按 ImageNet 均值方差标准化过。

于是它先做一次“反标准化”：

- `_input_to_unit`：把输入从标准化空间还原到 `[0, 1]`

然后分成两路：

- `_prepare_autogaze_pixels`
  - resize 到 AutoGaze 需要的分辨率
  - 再映射到 `[-1, 1]`
  - 最后按 AutoGaze processor 的均值方差再标准化
- `_prepare_siglip_pixels`
  - resize 到 SigLIP 需要的分辨率
  - 按 SigLIP processor 的均值方差再标准化

也就是说，AutoGaze 和 SigLIP 各自吃的是同一批原图，但各自有一套独立的预处理。

#### 3.3 AutoGaze 只处理历史帧

前向时：

1. `history_frames` 先走 `_prepare_autogaze_pixels`
2. 调用 `self.autogaze({"video": history_pixels_for_gaze}, generate_only=True, ...)`
3. 得到 `history_info`

这里的 `history_info` 里最重要的是：

- `gazing_pos`
- `num_gazing_each_frame`
- `if_padded_gazing`

根据 `third_party/AutoGaze/INTEGRATION.md` 和 `modeling_siglip.py`，这些信息描述的是：

- 历史视频所有 frame 拼起来后，哪些 patch 被选中
- 每一帧选了多少 patch
- 哪些位置只是 padding

#### 3.4 当前视角不走稀疏采样，而是走 dense token

公共 stack 并不会让 AutoGaze 给当前三视角做稀疏 patch 选择。

它的做法是：

- 历史帧：用 AutoGaze 给出的 sparse patch 索引
- 当前视角：每个视角直接保留完整 patch 网格

`_build_unified_gazing_info` 做的就是把两类 token 合并成一套统一索引：

- history 部分沿用 AutoGaze 产生的 sparse `gazing_pos`
- 当前 `head / left / right` 则人为构造 dense group

所以最终给 SigLIP 的不是“所有帧都稀疏”，而是“历史稀疏 + 当前全量”。

### 4. SigLIP 编码和下游消费

公共 stack 会把 `history_frames` 和 `current_views` 拼起来，一起送给改造过的 `SiglipVisionModel`。

底层 SigLIP 的改造点在 `third_party/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py`，核心有两件事：

- patch embedding 只保留 `gazing_info` 指向的 patch
- attention mask 不再是普通 image ViT 的全连接，而是按视频 token 序列构造

编码结束后，公共 stack 会：

1. 用 projector 把 SigLIP 输出映射到下游维度
2. 加上 `source_embed`
3. 加上 `age_embed`
4. 返回：
   - `tokens`
   - `attention_mask`
   - `metadata`

再往下，`third_party/SimVLA/models/modeling_smolvlm_vla.py` 的 `forward_vlm_autogaze` 会把这些视觉 token 和文本 embedding 直接拼起来，喂给 `self.vlm.model.text_model`。

所以从系统结构上看，这个 stack 是“把视觉观察转成 token”，而不是直接负责视觉语言融合。

## 关键设计理解

### 1. 这是一个明显带任务假设的“公共模块”

虽然代码放在 `src/common/vision_stacks`，但它其实仍然带着很强的 RMBench / SimVLA 语义：

- 默认历史只看 `head_history`
- 默认当前视角是 `head / left / right`
- 默认下游是 token-based VLM 消费方式
- 默认输入已经按 ImageNet 方式归一化

所以它现在更像“抽取复用后的 SimVLA AutoGaze encoder”，还不是完全去任务假设的通用视觉组件。

### 2. 当前设计的观测策略是“历史稀疏、当前稠密”

这点非常关键，因为它解释了这个模块为什么不是简单地“把 AutoGaze 套在所有图像上”。

当前实现更偏向：

- 用历史信息做记忆压缩
- 用当前三视角保留完整感知能力

这个设计对机器人任务是合理的，但它也意味着 token 数量仍然会明显受当前三视角分辨率影响。

## 目前看到的主要问题

### 1. `group_role_ids` 和 `group_roles` 键名不一致

这是我看到的最关键实现问题。

公共 stack 在 `src/common/vision_stacks/autogaze_siglip/stack.py` 里构造的是：

- `group_role_ids`

但底层 AutoGaze SigLIP 在 `third_party/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py` 里检查的是：

- `group_roles`

这会直接带来一个行为差异：

- 原始 SimVLA 版本会走 `get_observation_mask`
- 当前公共 stack 大概率不会走这条分支，而是退回普通 `get_causal_mask`

影响是当前公共实现的注意力行为可能已经和原始 `third_party/SimVLA/models/observation_encoder_autogaze.py` 不一致，尤其是：

- history 组之间的因果关系
- 当前三视角之间的互相可见性
- `current head` 是否能读 history，而 wrist 视角不能直接读 history 的那套特殊规则

这不是文风问题，而是很可能改变模型行为的真实 bug。

### 2. “可配置视角名”与底层语义并没有完全对齐

`StackConfig` 允许配置 `current_view_names`，看起来像支持任意当前视角命名。

但底层 observation mask 的语义实际上是写死在 `ROLE_HEAD / ROLE_LEFT / ROLE_RIGHT` 上的：

- head 可以读 history
- left/right 不能直接读 history
- 当前三视角之间是双向可见

所以这个“可配置”目前并不完全真实：

- 现在因为键名不匹配，特殊 observation mask 根本没用上
- 即使以后把键名修正，如果视角顺序或语义不是 `head / left / right`，mask 语义也会跟着错位

换句话说，这个 stack 现在对“视角可配置”的支持更多停留在表层接口，还没有完全打通到注意力语义。

### 3. `image_mask` 在 wrapper 里被直接忽略

`src/projects/maevla/simvla_autogaze/sandbox/models/observation_encoder_autogaze.py` 里：

- `image_mask` 被 `del`
- `current_views = image_input[:, :3]`

这意味着当前实现默认：

- 前 3 路一定就是有效视角
- 顺序一定就是 `head / left / right`

只要数据集视角顺序变化、视角数变化、或者存在缺失视角，这里就会静默地产生错误行为，而不是显式失败。

### 4. `history_len` 只是“半约束”，没有被真正校验

`history_len` 现在主要用于两件事：

- 配置期望的历史长度
- 初始化 `age_embed = nn.Embedding(self.history_len + 1, ...)`

但在 `forward` 里并没有显式检查：

- `history_frames.shape[1]` 是否真的等于 `config.stack.history_len`

这会带来两个问题：

- 如果实际 history 比配置更短，语义会漂移，但代码不一定报错
- 如果实际 history 比配置更长，那么 `age_ids` 可能超过 `age_embed` 的上界，出现 embedding index 越界

所以这里更像是“下游大家默认都传对了”，而不是一个稳固的模块边界。

### 5. 默认 `attn_type=bidirectional` 与 AutoGaze 文档推荐值不一致

公共配置里默认写的是：

- `attn_type: bidirectional`

但 `third_party/AutoGaze/INTEGRATION.md` 里明确把：

- `block_causal`

当作推荐默认值。

这不一定是错，因为当前项目的设计目标可能就是让当前观察和历史完全互看。
但问题在于：

- 这个偏离没有在文档里被明确解释
- 再叠加上面 `group_roles` 键名不一致的问题，当前真实 attention 语义会更难推断

所以这里至少是一个需要澄清实验意图的设计点。

### 6. 模型加载强依赖本地 cache，首次部署不够稳

`_resolve_model_path` 的行为是：

- 如果给的是本地路径，就直接用
- 如果给的是 repo id，就只去本地 Hugging Face cache snapshot 里找
- 找不到就直接报错

也就是说当前公共 stack 并不负责下载模型，也不直接走标准的 `from_pretrained(repo_id)` 在线解析路径。

这对离线训练环境是友好的，但对首次部署、换机器、或者 cache 目录变化的情况不够鲁棒。

### 7. smoke 脚本里还有旧路径

`src/projects/maevla/simvla_autogaze/run_train_smoke.sh` 里写的是：

- `src/projects/maevla/experiments/simvla_autogaze`

但当前仓库真实目录是：

- `src/projects/maevla/simvla_autogaze`

这说明公共 stack 接入后的工程脚手架还没有完全收拢干净，至少 smoke 脚本这一层还有路径遗留问题。

## 简短结论

这个模块的整体思路是清楚的，而且工程目标也很明确：把原本耦合在 SimVLA 里的 AutoGaze observation encoder 提成公共 stack，然后继续服务于 SmolVLM 风格的下游。

但从当前实现看，它还处在“已经抽出来了，但还没完全抽干净”的阶段：

- 主体前向链路已经能表达清楚
- 但接口语义仍然强依赖 RMBench / SimVLA 约定
- 还存在一个很关键的 `group_roles` 键名错配问题

如果后面要继续把这块做成真正的 `common` 组件，我觉得优先级最高的事情不是继续加配置，而是先把语义合同补齐：

1. 修正 `group_roles` 键名和 observation mask 行为
2. 明确 `current_view_names` 与底层 attention 语义的对应关系
3. 给 `history_len`、`image_mask`、视角顺序增加显式校验
4. 再考虑把模型加载、脚本路径、实验配置做成更稳的公共接口
