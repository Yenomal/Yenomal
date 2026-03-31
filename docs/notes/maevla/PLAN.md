# maevla

## Background

1. Masked AE ，一种对样本进行类RAG的处理过程，通过对 Patch 每部分计算重建重要性，并做出针对性的Mask以降低模型计算，提高模型吞吐的方法，算法本质是一种RAG。

2. 在具身领域，视频流的输入是短期记忆的重要指标。PI公司的[MEM](https://www.arxiv.org/abs/2603.03596)提出的一种 Casual Temporal + Spatial Attention 保障了输入VLM的token量不变。但是其中的 Visual Encoder 依然需要处理大量的 Token，导致其使用的图像尺寸，历史窗口长度都不足。


## Target
我们打算在各种VLA上部署AutoGaze，提高VLA的模型输入，最后的目标是能保持30视频窗口+1080p图像输入，并且基本不增加模型计算量

## TODO

TODO —— RMBench分数

- [x]  SimVLA+RMBench —— 0
- [x]  SimVLA+AutoGaze+RMBench —— 0
- [ ]  Pi05+RMBench —— 16
- [ ]  Pi05+AutoGaze+RMBench —— 
- [ ]  Lingbot-VA+RMBench —— 
- [ ]  Lingbot-VA+AutoGaze+RMBench —— 

## Target

我们打算在各种VLA上部署AutoGaze，提高VLA的模型输入，最后的目标是能保持30视频窗口+1080p图像输入，并且基本不增加模型计算量