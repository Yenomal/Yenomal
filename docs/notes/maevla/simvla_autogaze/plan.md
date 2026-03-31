# 计划

## 目标

围绕 `SimVLA + RMBench + AutoGaze` 做一个统一的 Observation Encoder，实现下面这套信息流：

- `head_history` 使用 AutoGaze 产生 sparse patch。
- `current_head / current_left / current_right` 使用人工构造的 dense `gazing_info`。
- 四组 token 一起进入同一个改造后的 `SigLIP`。
- 注意力逻辑满足：
  - `head_history -> head_history`：保留原本的 causal / block-causal。
  - `current_head / current_left / current_right`：当前三视角内部 fully bidirectional。
  - `current_head` 可以读取 `head_history`。
  - `current_left / current_right` 不能直接读取 `head_history`。
- `SigLIP` 输出 latent token 后，经新的 projector 映射到 `SmolVLM text hidden`，再送入 `SmolVLM text_model`。

## Observation 设计

统一 observation 结构定义为：

```python
{
    "head_history": Tensor[K, C, H, W],
    "head_current": Tensor[C, H, W],
    "left_current": Tensor[C, H, W],
    "right_current": Tensor[C, H, W],
    "instruction": str,
    "state": Tensor[14],
}
```

约束：

- `head_history` 默认不含当前帧，只取 `t-K ... t-1`
- `current tri-view` 是 `t` 时刻三张图
- 第一版固定 `K=8`

## 整体结构

```text
head_history
  -> AutoGaze
  -> sparse gazing_info

head_current / left_current / right_current
  -> manual dense gazing_info

统一拼接成一个 observation-level gazing_info
  -> customized SigLIP with custom observation mask
  -> new MLP projector
  -> source / age embedding
  -> SmolVLM text_model
  -> SimVLA action transformer
```

## 核心设计决策

### 1. 不复用 SmolVLM 原 connector

原因：

- `SmolVLM` 原 connector 不是单纯 `768 -> 960`
- 它前面还有 `pixel_shuffle`
- `pixel_shuffle` 假设输入 token 是规则网格
- `head_history` 是 sparse token，不适合直接走这套结构

结论：

- 统一使用新的 `MLP projector`
- 全量微调

### 2. 不把三视角 current frame 当作伪时序

原因：

- `current head / left / right` 是同一时刻的不同视角，不是时间帧
- 当前三视角内部必须 fully bidirectional，否则会损失多视角互补信息

结论：

- `SigLIP` 中要做 observation-level custom mask
- 而不是继续沿用默认 temporal mask 语义

### 3. wrist 不直接看 history

逻辑：

- `head` 的历史不是 `wrist` 的历史
- `wrist` 只应通过 `head` 得到历史提示

结论：

- 在 attention mask 上显式约束：
  - `current_head -> history = 允许`
  - `current_left/right -> history = 禁止`

## 自定义 Mask 目标

token 顺序固定为：

```text
[H_hist][H_now][L_now][R_now]
```

mask 关系定义为：

| Query \\ Key | H_hist | H_now | L_now | R_now |
|---|---:|---:|---:|---:|
| H_hist | causal | 0 | 0 | 0 |
| H_now  | 1 | 1 | 1 | 1 |
| L_now  | 0 | 1 | 1 | 1 |
| R_now  | 0 | 1 | 1 | 1 |

语义解释：

- `H_hist` 内部仍然是历史记忆流
- `H_now` 读取历史，并和当前三视角完全双向融合
- `L_now / R_now` 不直接读 raw history
- `L_now / R_now` 通过 `H_now` 间接获得历史信息

## 需要修改 / 新增的文件

### 建议新增

- [observation_encoder_autogaze.py](/home/rui/SimVLA+RMBench/SimVLA/models/observation_encoder_autogaze.py)

### 建议修改

- [rmbench_hdf5.py](/home/rui/SimVLA+RMBench/SimVLA/datasets/domain_handler/rmbench_hdf5.py)
- [modeling_smolvlm_vla.py](/home/rui/SimVLA+RMBench/SimVLA/models/modeling_smolvlm_vla.py)
- [modeling_siglip.py](/home/rui/SimVLA+RMBench/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py)
- [configuration_siglip.py](/home/rui/SimVLA+RMBench/AutoGaze/autogaze/vision_encoders/siglip/configuration_siglip.py)
- [server_adapter.py](/home/rui/SimVLA+RMBench/RMBench/policy/SimVLA/server_adapter.py)

## 分步骤实现思路

### Step 1：扩展 RMBench 样本，带入 `head_history`

改动目标：

- 离线训练样本从“当前单帧三视角”扩成“head_history + 当前三视角”

实施点：

- 在 `rmbench_hdf5.py` 中新增 `history_len`
- 在 `iter_episode()` 中读取 `head_camera` 的历史窗口
- 输出 `head_history / head_current / left_current / right_current`

第一版建议：

- `history_len = 8`
- 历史不足时用最早一帧 pad

### Step 2：在推理侧维护 history 缓存

改动目标：

- `server_adapter.py` 在在线推理时能持续维护 `head_history`

实施点：

- `reset_model()` 清空 history deque
- `get_action()` 中把当前 `head_camera` 推入缓存
- 组装 observation dict 给新的 observation encoder

### Step 3：统一 `gazing_info`

改动目标：

- 保持 AutoGaze 原有输出协议不动
- 额外新增一个 observation-level 打包层

实施点：

- `head_history`：
  - 走 AutoGaze
  - 得到 sparse `gazing_pos / if_padded_gazing / num_gazing_each_frame`
- `current_head / left_current / right_current`：
  - 手工构造 dense `gazing_info`
  - 第一版直接全 patch dense
- 按固定顺序拼成一个 unified `gazing_info`
- 增加 `group_roles`

### Step 4：在 SigLIP 中增加 `get_observation_mask()`

改动目标：

- 尽量保持 AutoGaze 版 SigLIP 主体不变
- 只在 mask 构造层面做最小有效改动

实施点：

- 在 `modeling_siglip.py` 中新增 `get_observation_mask()`
- 当 `gazing_info` 里存在 `group_roles` 时，走新的 observation mask
- 否则继续走原来的 `get_causal_mask()`

技术选择：

- backend 使用 `sdpa`
- 不使用 `flash_attention_2`

### Step 5：新增统一 Observation Encoder

改动目标：

- 把 AutoGaze、人工 gaze、SigLIP、projector、source/age embedding 封装成一个模块

实施点：

- 新建 `observation_encoder_autogaze.py`
- 输入：
  - `head_history / head_current / left_current / right_current`
- 输出：
  - `obs_tokens`
  - `obs_attention_mask`

模块内部职责：

- 调 AutoGaze
- 构造 unified `gazing_info`
- 调 customized SigLIP
- 调新的 MLP projector
- 加 `source embedding`
- 给 history token 加 `age embedding`

### Step 6：在 SimVLA 主模型中替换视觉前端

改动目标：

- 用新的 observation encoder 替换当前 `forward_vlm_efficient()`

实施点：

- `modeling_smolvlm_vla.py`
- 新增 `forward_vlm_obs()`
- 把 observation token 与 text embedding 拼接
- 送入 `self.vlm.model.text_model`

### Step 7：全量微调

改动目标：

- 直接按新 observation encoder 端到端训练

建议：

- 不做外部冻结假设
- 直接全量微调
- 但训练顺序仍建议先跑通离线训练，再接在线 server

## 关键代码片段

### 1. 新 projector

```python
class ObservationProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=960, hidden_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```

### 2. source / age embedding

```python
self.source_embed = nn.Embedding(4, out_dim)   # hist_head / cur_head / cur_left / cur_right
self.age_embed = nn.Embedding(history_len, out_dim)
```

### 3. custom observation mask 伪代码

```python
def get_observation_mask(group_lengths, group_roles, batch_size, num_heads, token_mask=None, dtype=torch.float32):
    ...
    # H_hist -> H_hist: causal
    # H_now/L_now/R_now 内部 fully bidirectional
    # H_now -> H_hist: allowed
    # L_now/R_now -> H_hist: blocked
    ...
    return mask
```

## 风险与验证点

### 风险

1. unified token 数过大，`sdpa` 开销变高
2. history sparse token 和 current dense token 分布不一致
3. custom mask 在 padding batch 上容易出错
4. source / age embedding 不加会导致角色混乱

### 必做验证

1. 验证 `group_roles` 和 offsets 对齐正确
2. 验证 `L_now / R_now` 对 `H_hist` 的 attention 真正被 mask 掉
3. 验证 `H_now` 的确能从 `H_hist` 读到信息
4. 验证 current tri-view block 内 attention 是 full bidirectional
5. 验证在线推理 history cache 的 reset / append 逻辑

## 最小可跑通版本

第一版建议严格收敛到下面这版：

1. `head_history` 用 AutoGaze。
2. `current head / left / right` 全部 dense gaze。
3. 自定义 observation mask。
4. `SigLIP` backend 用 `sdpa`。
5. 新 projector 用 MLP。
6. 全量微调。

这版最符合目标，而且实现复杂度最低。
