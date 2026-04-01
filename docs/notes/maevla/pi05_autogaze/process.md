## 实验目的

* 这版实验的主要目标是测试 AutoGaze 这种Sparse History是否对于短期记忆有效，最终结果应该是pi05在RMBench上battery-try分数提高

## 实验过程

这一版是pi05+autogaze，我们没有修改任何内容，只是将原本common里面的AutoGaze替换了pi05的paligemma visual encoder，但是我们在common里面修改了SigLIP，以支持双向attention，Casual attention在多视角下问题实在太多，这里就交给后人的智慧

这一次还尝试了codex的PLAN模式，我的评价是不太行，下一次可以尝试让他写出plan后写入plan.md而不是直接进行，多轮迭代后再进行

## TODO

我们这样的方法是比较解耦的方法，后续的任何调整都可以在common上进行更新迭代，具体的想法包括：

1. 输入 Multi-View 切换

2. 输入 Attention 设计

3. Encoder 切换为 V-JEPA 2.1 ，V-JEPA 训练是Mask训练，原生比较支持我们的想法

4. 学习率调度问题

5. AutoGaze训练，这是一个Mask Policy以及RAG问题

## Ques

1. 我们目前除了research，plan，list，run都没有模板，这个地方需要尽早确立规范，我的感觉是research需要明确数据流动情况，plan需要明确更改范围，list需要尽量简单按照待办清单格式，run应该简洁，只提出bash脚本

2. 我们对bash脚本是有要求的，需要参数可以在bash脚本下修改，而不是在命令上修改，所以这里需要注意让其执行前需要做出声明——prompt约束，这个prompt也可以写入或者我们单独写一个restrict.md脚本来说明各种约束

3. bash脚本的位置，和我们每次运行这个脚本的工作空间需要配置好，这里还需要想一下，sh的位置是不是可以放在原项目位置，外部只加小小的一层，还是sh脚本就暴露在外面，因为sh脚本是用户的调参口，所以我感觉还是放在外面比较好

4. 我们使用dev_shell.sh固定环境，和manifest是不是有重复的地方