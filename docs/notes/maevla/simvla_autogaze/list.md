# 代码修改流程

## 0. 先做约束确认

- 不修改现有训练主线前，先确认新 observation encoder 只替换视觉输入部分。
- `action space / loss / generate_actions` 暂不改。
- 第一版只支持：
  - `head_history`
  - `current head`
  - `current left`
  - `current right`

## 1. 扩展训练数据输入

### 文件

- [rmbench_hdf5.py](/home/rui/SimVLA+RMBench/SimVLA/datasets/domain_handler/rmbench_hdf5.py)

### 目标

- 让一个 sample 输出：
  - `head_history`
  - `head_current`
  - `left_current`
  - `right_current`

### 流程

1. 增加 `history_len` 配置读取。
2. 读取 `head_camera` 历史窗口。
3. 当前三视角独立输出。
4. 保持 `abs_trajectory` 逻辑不变。

### 检查点

- batch 出来的历史 shape 是否固定为 `[K, C, H, W]`
- 历史不足时 padding 是否稳定

## 2. 扩展在线推理缓存

### 文件

- [server_adapter.py](/home/rui/SimVLA+RMBench/RMBench/policy/SimVLA/server_adapter.py)

### 目标

- 在线推理时维护 `head_history`

### 流程

1. 在 runner 中新增 `deque(maxlen=history_len)`。
2. `reset_model()` 清空 deque。
3. `get_action()` 中读取 deque，拼出 `head_history`。
4. 把当前 `head_camera` append 进去。

### 检查点

- 首帧时 history 缺失是否正确 pad
- episode 切换时缓存是否清空

## 3. 新建统一 Observation Encoder

### 文件

- [observation_encoder_autogaze.py](/home/rui/SimVLA+RMBench/SimVLA/models/observation_encoder_autogaze.py)

### 目标

- 统一封装：
  - AutoGaze
  - manual dense gaze
  - customized SigLIP
  - projector
  - source / age embedding

### 流程

1. 定义 observation 输入结构。
2. 调 AutoGaze 处理 `head_history`。
3. 构造 `current head / left / right` 的 dense `gazing_info`。
4. 拼接 unified `gazing_info`。
5. 调 SigLIP。
6. 调 projector。
7. 加 `source / age embedding`。
8. 输出 `obs_tokens / obs_attention_mask`。

### 检查点

- unified `gazing_pos` offsets 是否正确
- `group_roles` 顺序是否固定

## 4. 给 SigLIP 加 observation mask

### 文件

- [modeling_siglip.py](/home/rui/SimVLA+RMBench/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py)
- [configuration_siglip.py](/home/rui/SimVLA+RMBench/AutoGaze/autogaze/vision_encoders/siglip/configuration_siglip.py)

### 目标

- 复用现有 SigLIP 主体，只新增 observation-level mask 逻辑

### 流程

1. 增加 `group_roles` 分组支持。
2. 新增 `get_observation_mask()`。
3. `forward()` 中判断：
   - 有 `group_roles` -> 用 observation mask
   - 没有 -> 用原 mask
4. backend 切到 `sdpa`

### 检查点

- `H_hist -> H_hist` 是否仍是 causal
- `H_now / L_now / R_now` 是否 fully bidirectional
- `L_now / R_now -> H_hist` 是否被完全 mask
- `H_now -> H_hist` 是否允许

## 5. 新增 projector 和 embedding

### 文件

- [observation_encoder_autogaze.py](/home/rui/SimVLA+RMBench/SimVLA/models/observation_encoder_autogaze.py)

### 目标

- 用新 MLP projector 统一映射到 `SmolVLM text hidden`

### 流程

1. 新增 `ObservationProjector`。
2. 新增 `source_embed`。
3. 新增 `age_embed`。
4. 约定：
   - `hist_head = 0`
   - `cur_head = 1`
   - `cur_left = 2`
   - `cur_right = 3`

### 检查点

- projector 输出维度是否稳定为 `960`
- history token 的 `age embedding` 是否只加在历史部分

## 6. 替换 SimVLA 的视觉前端

### 文件

- [modeling_smolvlm_vla.py](/home/rui/SimVLA+RMBench/SimVLA/models/modeling_smolvlm_vla.py)

### 目标

- 用新的 observation encoder 替换当前 `forward_vlm_efficient()`

### 流程

1. 初始化 `self.obs_encoder`。
2. 新增 `forward_vlm_obs()`。
3. 用 `obs_tokens + text_embeds` 送入 `self.vlm.model.text_model`。
4. 保持后面的 action transformer / flow matching 逻辑不变。

### 检查点

- `vlm_features` 的 shape 是否符合后续 transformer 预期
- `generate_actions()` 和训练 `forward()` 是否都接到了新入口

## 7. 训练接线

### 文件

- [train_smolvlm.py](/home/rui/SimVLA+RMBench/SimVLA/train_smolvlm.py)
- [dataset_smolvlm.py](/home/rui/SimVLA+RMBench/SimVLA/datasets/dataset_smolvlm.py)

### 目标

- 训练 batch 能携带新的 observation 结构

### 流程

1. dataloader 接受新字段。
2. processor 不再只处理三视角当前帧。
3. 模型输入改成 observation dict。
4. 训练 loop 接 `forward_vlm_obs()`。

### 检查点

- batch collate 是否稳定
- history 部分是否在多 worker 下不乱序

## 8. 离线验证

### 必做

1. 先不跑全训练，只做一次 forward。
2. 打印 unified token 数。
3. 打印 custom mask 的 block 结构。
4. 检查 `left/right` 对 `history` 的 attention 值是否真的为 0。
5. 检查 `head` 对 `history` 的 attention 是否非 0。

### 通过标准

- forward 不报 shape error
- mask 行为符合设计
- text_model 输入长度可控

## 9. 全量微调

### 目标

- 直接按新 observation encoder 端到端训练

### 流程

1. 先小 batch 冒烟。
2. 再全量训练。
3. 优先看 memory-heavy tasks。

### 建议任务

- `observe_and_pickup`
- `battery_try`
- `blocks_ranking_try`

## 10. 在线推理验证

### 文件

- [server_adapter.py](/home/rui/SimVLA+RMBench/RMBench/policy/SimVLA/server_adapter.py)

### 目标

- 让在线 policy 跑通 history cache + new observation encoder

### 流程

1. `reset_model()` 清缓存。
2. `get_action()` 组 observation。
3. 先跑 smoke test。
4. 再进正式 eval。

### 检查点

- 首几步 history 不足是否稳定
- 长 episode 中 deque 是否正常滑动

## 最小落地顺序

如果要最快推进，严格按下面顺序：

1. `rmbench_hdf5.py`
2. `server_adapter.py`
3. `observation_encoder_autogaze.py`
4. `modeling_siglip.py`
5. `modeling_smolvlm_vla.py`
6. `train_smolvlm.py`
7. 小规模 forward 验证
8. 小规模训练
9. smoke test
10. 全量微调
