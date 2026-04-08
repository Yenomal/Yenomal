# OpenPI 接入 AutoGaze-SigLIP 计划

## 目标

在不修改 `openpi` 训练/模型 config、也不修改本地 `PYTHONPATH` 的前提下，把 `openpi` 当前的视觉前缀构造逻辑，从原生 `PaliGemma` 的 `embed_image()`，替换成 `AutoGaze + SigLIP` 风格的视觉 encoder。

本计划默认优先支持 `PyTorch OpenPI` 路径，也就是当前 `PI0Pytorch` 这条线。原因很直接：

- `src/common/vision/visual_gaze/visual_gaze_encoder.py` 是 `torch.nn.Module`
- `openpi` 当前既有 `JAX` 版，也有 `PyTorch` 版
- 你现在这条 `pi05_baseline_rmbench` 实际更适合先接到 `PI0Pytorch`

如果后面需要 `JAX Pi0` 也支持，需要单独再做一版，不建议这次一起做。

## 先说结论

推荐的接入点不是去替换 `PaliGemma` 内部的 `vision_tower`，也不是去改 `openpi` config，而是：

在 `third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py` 的 `embed_prefix()` 这一层，引入一个新的视觉前缀适配器，把：

- `openpi` 的 3 路当前图像
- 可选的 `head_history`

变成：

- `image_prefix_embs: [B, N, D]`
- `image_prefix_pad_mask: [B, N]`

然后再和语言 token embedding 直接拼接。

这条路最稳，因为 `openpi` 下游真正需要的只是“前缀 token embedding”，并不关心这些 token 是 `PaliGemma` 原生视觉塔产出的，还是 `AutoGaze-SigLIP` 产出的。

## 为什么不要去动 `PaliGemma` config

`openpi` 当前图像编码入口非常简单：

- `third_party/openpi/src/openpi/models_pytorch/gemma_pytorch.py` 里，`embed_image()` 只是调用 `self.paligemma.model.get_image_features(image)`
- `third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py` 里，`embed_prefix()` 逐路调用 `embed_image()`，然后把图像 token 和语言 token 拼起来

如果去改 `PaliGemma` 的 `vision_config` 或 `multi_modal_projector`，会立刻遇到这些问题：

- 预训练权重加载关系会变复杂
- `PaliGemma` 自带 projector 的输入输出维度要一起重配
- 很容易把原本已经跑通的 `openpi` checkpoint 破坏掉

而如果只替换 `embed_prefix()` 之前的视觉前缀构造，则：

- action expert、language model、diffusion / flow matching 主干都不用动
- 现有 checkpoint 的绝大部分加载逻辑不变
- 视觉 encoder 的替换范围被锁死在 prefix interface 上

## 两边接口怎么对齐

### `openpi` 现在的接口

`third_party/openpi/src/openpi/models/model.py`

- 图像 key 固定是 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`
- `Observation.from_dict()` 会把 `uint8` 图像转成 `[-1, 1]` 的 `float32`

`third_party/openpi/src/openpi/models_pytorch/preprocessing_pytorch.py`

- preprocessing 接受 `[B, C, H, W]` 和 `[B, H, W, C]`
- 最终 `embed_prefix()` 拿到的是按 view 拆开的 `images: list[Tensor]`

`third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py`

- `_preprocess_observation()` 返回 `list(observation.images.values())`
- `embed_prefix()` 当前逐张图编码
- 期望得到的是可直接喂给 `PaliGemma` language trunk 的 prefix embedding

### `VisualGazeEncoder` 的接口

`src/common/vision/visual_gaze/visual_gaze_encoder.py`

- 输入是 `sparse_video: [B, T, C, H, W]` 和 `dense_images: [B, V, C, H, W]`
- 输出是：
  - `tokens: [B, N, D]`
  - `attention_mask: [B, N]`
  - `metadata`

这和 `openpi` 的 prefix 需求其实天然接近。真正需要补的只有 4 件事：

1. 把 `openpi` 的 `list[3 x image]` 变成 `dense_images`
2. 给 `AutoGaze` 补 `head_history`
3. 把 `VisualGazeEncoder` 的输出维度改到 `openpi` 语言 hidden size
4. 处理 normalization mismatch

## 最大的接口问题其实不是 shape，而是 normalization

这是这次最关键的坑。

`VisualGazeEncoder._to_unit()` 假设输入已经按 `config.input.input_mean/std` 做过 normalize，然后它会把输入“反归一化”回 `[0, 1]`。

但 `openpi` 里的图像不是 ImageNet mean/std 归一化，而是 `[-1, 1]`。

所以如果把 `openpi` 的图直接喂给 `VisualGazeEncoder`，默认结果一定不对。

正确做法是：实例化 `VisualGazeEncoder` 时做 runtime override，而不是改 `openpi` config：

```python
overrides = {
    "input": {
        "input_mean": (0.5, 0.5, 0.5),
        "input_std": (0.5, 0.5, 0.5),
        "current_view_names": ("head", "left", "right"),
    },
    "projector": {
        "output_dim": paligemma_width,
    },
}
```

这里：

- `input_mean/std = 0.5/0.5` 是为了把 `[-1, 1]` 正确还原回 `[0, 1]`
- `current_view_names = ("head", "left", "right")` 用来和 `base / left_wrist / right_wrist` 的语义对齐
- `projector.output_dim = paligemma_width` 是为了让视觉 token 可以直接和语言 token 拼接

## 输出维度必须改成 `PaliGemma` hidden size

`src/common/vision/visual_gaze/config.yaml` 里默认 `projector.output_dim = 960`，这是给 SimVLA/SmolVLM 风格模型准备的，不适合直接进 `openpi`。

`openpi` 这边：

- `gemma_2b` / `gemma_2b_lora` 的 hidden size 是 `2048`
- `gemma_300m` 是 `1024`

所以这里不能直接复用默认 `960`。

推荐做法：

- 在 `PI0Pytorch.__init__()` 里根据 `paligemma_variant` 读出 `paligemma_width`
- 用 runtime override 把 `VisualGazeEncoder.projector.output_dim` 改成这个值

这样做的好处是：

- 不碰 `openpi` config
- 不碰 `src/common` 默认 yaml
- 同一套 `VisualGazeEncoder` 可以被不同 `paligemma_variant` 复用

## 推荐的实现结构

建议新增一个 `openpi` 内部适配层，例如：

- `third_party/openpi/src/openpi/models_pytorch/autogaze_visual_adapter.py`

职责只做一件事：

- 接收 `openpi` preprocess 后的 observation 图像
- 调 `VisualGazeEncoder`
- 返回 `openpi embed_prefix()` 需要的 prefix image embeddings 和 pad mask

建议接口长这样：

```python
class OpenPIAutoGazeVisualAdapter(nn.Module):
    def forward(
        self,
        *,
        images: list[torch.Tensor],      # [base, left, right], each [B,C,H,W]
        image_masks: list[torch.Tensor], # each [B]
        head_history: torch.Tensor | None,  # [B,T,C,H,W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        return image_embs, image_pad_mask
```

这样 `PI0Pytorch.embed_prefix()` 只需要在分支里写：

```python
if self.visual_adapter is not None:
    img_embs, img_pad_masks = self.visual_adapter(...)
else:
    # old paligemma embed_image path
```

## 为什么推荐复用 `src/common/vision/visual_gaze/visual_gaze_encoder.py`

仓里其实已经有两份思路：

- `src/common/vision/visual_gaze/visual_gaze_encoder.py`
- `third_party/SimVLA/models/observation_encoder_autogaze.py`

这两份的核心逻辑是同一路：

- `AutoGaze` 先决定 history token selection
- `SigLIP` 再按 unified selection 编码
- projector 映射到下游模型 hidden size
- 再补 `source_embed` 和 `age_embed`

这次更推荐复用 `src/common` 里的 `VisualGazeEncoder`，不要再在 `openpi` 里复制一份 SimVLA 版本，原因是：

- 你的 `PYTHONPATH` 现在已经包含 `${ROOT}/src`
- `VisualGazeEncoder` 已经把 tokenizer / encoder / projector / mask 逻辑封好了
- 后续如果 SimVLA 和 OpenPI 都走同一套 `src/common`，维护成本最低

`third_party/SimVLA/models/observation_encoder_autogaze.py` 更适合作为“接口参考”和“行为对照”，不建议直接复制过去。

这里要补一个当前环境下的重要注意点：

- 既然 `PYTHONPATH` 指到的是 `${ROOT}/src`，那 `src/common` 对外更自然的包名应该是 `common.*`
- 但当前 `src/common` 内部还有旧式绝对 import，比如 `from nn import ...`、`from vision...`
- 这类写法是基于旧的 `${ROOT}/src/common` 入口成立的

所以在真正落地前，建议把 `src/common` 内部 import 统一到：

- `from common.nn ...`
- `from common.vision ...`

或者至少先做一层兼容 shim。否则 IDE 索引可以对，但运行时 import 路径还是可能错位。

## `head_history` 怎么进来

这是第二个真正的关键点。

如果只看当前 `openpi` observation，只有 3 路当前帧，没有 history。那就算把 `dense_images` 接上了，也只是 “SigLIP current-view encoder”，还不是完整的 `AutoGaze-SigLIP`。

最终要做成你想要的版本，必须补一个 `head_history` 通道。

这里有两个方案。

### 方案 A：先做 inference-only，history 由 runner/cache 提供

适合先把接口跑通，改动最小。

做法：

- 在 `OpenPI` 的 server runner 或 `Policy` 外层维护一个 `deque`
- 每一步把当前 `base_0_rgb` 压进去
- 拼成 `[B, T, C, H, W]` 的 `head_history`
- 传给 `visual_adapter`

优点：

- 不用改 dataset
- 不用改训练 config
- 最快验证接口和 token shape

缺点：

- 这条路主要适合在线推理
- 训练时没有天然对应的 history 输入

### 方案 B：把 `head_history` 作为 observation 的可选字段

适合后面要训练/微调。

做法：

- 给 `Observation` 增加一个 optional `head_history`
- preprocessing 保持当前 3 路 view 逻辑不变，只额外透传 `head_history`
- dataset / transform 在 batch 里构造 `head_history`

优点：

- 训练和推理接口统一
- model 内部不需要维护状态

缺点：

- 会碰 `Observation` 数据结构
- 数据管线改动面比方案 A 大

### 这次推荐的顺序

建议先按 A 做接口验证，再决定要不要升到 B。

原因不是 A 更“正确”，而是它最能先回答一个问题：

`AutoGaze-SigLIP` 生成的 prefix token，只要维度和 mask 对上，能不能无痛接进 `openpi` 的 PaliGemma + action expert 主干？

这个问题先验证掉，后面再决定要不要把 history 贯穿到训练。

## `embed_prefix()` 应该怎么改

推荐保留现有函数签名，只在内部加分支。

现有逻辑：

1. 对每一路 image 调 `embed_image()`
2. 每路 image token 各自扩展 pad mask
3. 再拼语言 token

改完以后：

1. 如果没启用 autogaze visual adapter，走原逻辑
2. 如果启用了 adapter：
   - 先把 3 路当前图像 stack 成 `dense_images: [B, 3, C, H, W]`
   - 把 `head_history` 传给 adapter
   - adapter 返回：
     - `img_embs: [B, N, D]`
     - `img_pad_masks: [B, N]`
   - image prefix 的 `att_masks` 直接全 0
   - 再和 language embedding 拼接

也就是说，`openpi` 下游完全不用知道 image token 是怎么来的。

## `image_mask` 怎么处理

这件事要单独说一下。

`openpi` 原生接口是按 view 粒度有 `image_mask` 的。

但 `VisualGazeEncoder` 当前对 `dense_images` 只检查 view count，没有显式接受 `dense_view_mask`。也就是说它默认 dense view 都有效。

这会带来一个现实结论：

- 如果你当前 `RMBench / Aloha` 路径始终是 3 路图像齐全，那第一版可以先不处理
- 如果你想兼容缺失 wrist view 的通用场景，需要在 adapter 输出后，再把对应 group 的 token mask 置 0

第一版建议：

- 明确只支持 `base + left_wrist + right_wrist` 全有效
- 在 adapter 里加 assert

第二版再补：

- 根据 `group_roles` 和 `num_tokens_each_group`，把无效 view 对应 token span 整段 mask 掉

## 不改 config 的启用方式

因为你明确要求“不动 openpi 的 config”，所以启用方式不要走 `Pi0Config` 加字段。

建议走 runtime 开关：

- 环境变量
- 或单独的外部 yaml 路径

例如：

```bash
export OPENPI_VISUAL_ENCODER=autogaze_siglip
export OPENPI_VISUAL_GAZE_CONFIG=/home/rui/Yenomal/src/projects/.../autogaze_siglip_openpi.yaml
```

或者为了和你仓里现有脚本对齐，也可以兼容读取：

```bash
export MAEVLA_VISUAL_GAZE_CONFIG=...
```

重点是：

- `openpi` config 不加字段
- `PyTorch` model 初始化时自己读环境变量
- 没配环境变量就走原生 `embed_image()`，保证回退简单

## 不改 `PYTHONPATH` 这件事是否可行

可行。

你现在的：

```bash
export ROOT=/home/rui/Yenomal
export PYTHONPATH=${ROOT}/third_party:${ROOT}/src:$PYTHONPATH
```

这意味着 `openpi` 侧在接入时，应该优先按包根 `src` 来理解 import，也就是去 import `common` 这一层，而不是直接把 `vision` / `nn` 当成顶级包。

推荐的 import 目标应当是：

- `common.vision`
- `common.nn`

也就是说，`openpi` 代码里更推荐写成：

```python
from common.vision import VisualGazeEncoder
```

但要注意，当前 `src/common/vision/visual_gaze/visual_gaze_encoder.py` 和相关模块内部还在使用旧的 `from nn` / `from vision` 写法。所以“只改 `PYTHONPATH`”还不完全等于“这一套已经可运行”。

因此这部分建议拆成两个层次理解：

- 外部接入层：不需要再改本地 `PYTHONPATH`，直接基于 `${ROOT}/src` 即可
- `src/common` 内部实现层：需要补一次 import 风格统一，或者做兼容别名

## 建议的分阶段实施顺序

### Phase 1：只打通 prefix interface

- 新增 `OpenPIAutoGazeVisualAdapter`
- `PI0Pytorch.__init__()` 里按环境变量选择是否初始化 adapter
- `embed_prefix()` 增加 autogaze 分支
- 暂时允许 `head_history is None` 时走 dense-only smoke path

验收标准：

- `sample_actions()` 能跑通
- `prefix_embs.shape[-1] == paligemma_width`
- `prefix_pad_masks.shape[:2] == prefix_embs.shape[:2]`

### Phase 2：补上真实 `head_history`

- 先走 runner/cache 方案
- 保证 `head_history` 的 shape 是 `[B, T, C, H, W]`
- 明确 `T == history_len`

验收标准：

- `VisualGazeEncoder` 输出 token 数随 history 打开而变化
- `metadata["has_sparse"] == True`
- 推理结果稳定，不出现 shape / device mismatch

### Phase 3：如果要训练，再决定是否升级 observation interface

- 评估是否把 `head_history` 正式加入 observation 数据结构
- 如果需要训练，再补 dataset / transform

验收标准：

- train forward 和 sample forward 走同一套 adapter
- 不依赖 model 内部隐式状态缓存

## 我认为最需要盯紧的风险

### 风险 1：normalization 对不上

这是第一大风险，也是最容易让结果“能跑但全错”的地方。

必须确保：

- `openpi` 输入是 `[-1, 1]`
- `VisualGazeEncoder` 看到的 `input_mean/std` 被 override 成 `0.5/0.5`

### 风险 2：projector 输出维度不对

默认 `960` 不能直接进 `openpi` 的 language trunk。

必须按 `paligemma_variant` 改成对应 hidden size。

### 风险 3：view 顺序错位

一定要固定成：

- `base_0_rgb -> head`
- `left_wrist_0_rgb -> left`
- `right_wrist_0_rgb -> right`

不要偷懒按 dict 顺序“默认相信没问题”，最好显式按 key 取。

### 风险 4：history 来源不一致

如果 inference 用 runner cache，training 用别的定义，很容易出现 train/infer mismatch。

所以第一版就算只做 inference，也要把 `head_history` 的语义写死：

- 只来自 `base_0_rgb`
- newest 在最后一帧
- 长度固定 `history_len`

### 风险 5：mask 语义退化

第一版如果假设 3 路 view 永远有效，必须在文档和代码里写明，不然以后排查会很痛苦。

## 推荐的验证顺序

1. 先只验证 `dense_images -> VisualGazeEncoder -> tokens`
2. 再验证 `tokens -> openpi embed_prefix() -> sample_actions()`
3. 最后再打开 `head_history`

建议每一步都打印这些 shape：

```python
dense_images.shape
head_history.shape if head_history is not None else None
visual_tokens.shape
visual_attention_mask.shape
prefix_embs.shape
prefix_pad_masks.shape
```

如果第一轮只想快速 smoke test，我建议先用：

- 3 路当前图
- `head_history=None`
- `projector.output_dim=2048`
- `input_mean/std=0.5/0.5`

先证明“替换 visual prefix 不会破坏 openpi 主干”。

然后再补完整的 sparse history。

## 最后的推荐决策

如果只问“这件事最应该从哪里切进去”，我的答案很明确：

- 切 `PI0Pytorch.embed_prefix()`
- 复用 `src/common/vision/visual_gaze/visual_gaze_encoder.py`
- 不改 `PaliGemma` config
- 不改 `openpi` config
- 先做 `PyTorch` 路径
- 先把 prefix interface 跑通，再补 `head_history`

这条路线改动最小、回退最简单、也最符合你现在这套工程组织方式。
